# SOURCE: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/main/tiled_vae.py
# Modified for use in diffusers by Gábe.

import math
from typing import Tuple, List

from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.vae import Decoder, Encoder
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D, DownEncoderBlock2D
import torch
from torch.functional import F
from tqdm import tqdm
from PIL import Image

from src.config import Config, get_config


async def cheap_approximation(
    x: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: float,
    config: Config = None,
) -> List[Image.Image]:
    from src.pipeline import FluxPipeline

    if config == None:  # noqa
        config = get_config()

    coeffs = [
        [-0.0404, 0.0159, 0.0609],
        [0.0043, 0.0298, 0.0850],
        [0.0328, -0.0749, -0.0503],
        [-0.0245, 0.0085, 0.0549],
        [0.0966, 0.0894, 0.0530],
        [0.0035, 0.0399, 0.0123],
        [0.0583, 0.1184, 0.1262],
        [-0.0191, -0.0206, -0.0306],
        [-0.0324, 0.0055, 0.1001],
        [0.0955, 0.0659, -0.0545],
        [-0.0504, 0.0231, -0.0013],
        [0.0500, -0.0008, -0.0088],
        [0.0982, 0.0941, 0.0976],
        [-0.1233, -0.0280, -0.0897],
        [-0.0005, -0.0530, -0.0020],
        [-0.1273, -0.0932, -0.0680],
    ]
    latent_rgb_factors = torch.tensor(coeffs, dtype=x.dtype, device=x.device)
    x = FluxPipeline._unpack_latents(x, height, width, vae_scale_factor)
    images = []
    for z in x:
        latent_image = z.permute(1, 2, 0) @ latent_rgb_factors

        latents_ubyte = (
            ((latent_image + 1.0) / 2.0)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
        ).to(device="cpu", dtype=torch.uint8, non_blocking=True)

        image = Image.fromarray(latents_ubyte.numpy())
        if config.io.preview_size == "upscaled":
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        images.append(image)
    return images


def _attn(self: Attention, hidden_states: torch.Tensor) -> torch.Tensor:
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width
        ).transpose(1, 2)
    query: torch.Tensor = self.to_q(hidden_states)
    key: torch.Tensor = self.to_k(hidden_states)
    value: torch.Tensor = self.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads

    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, self.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width
        )
    return hidden_states


def inplace_nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x, inplace=True)


def attn2task(task_queue: List, attn: Attention):
    task_queue.append(("store_res", lambda x: x))
    task_queue.append(("pre_norm", attn.group_norm))
    task_queue.append(("attn", lambda x, attn=attn: _attn(attn, x)))
    task_queue.append(("add_res", None))


def resblock2task(task_queue: List, block: ResnetBlock2D):
    if block.use_in_shortcut:  # checks already for in != out
        task_queue.append(("store_res", block.conv_shortcut))
    else:
        task_queue.append(("store_res", lambda x: x))
    task_queue.append(("pre_norm", block.norm1))
    task_queue.append(("silu", inplace_nonlinearity))
    task_queue.append(("conv1", block.conv1))
    task_queue.append(("pre_norm", block.norm2))
    task_queue.append(("silu", inplace_nonlinearity))
    task_queue.append(("conv2", block.conv2))
    task_queue.append(("add_res", None))


def build_sampling(task_queue: List, network: Decoder | Encoder, is_decoder: bool):
    if is_decoder:
        network: Decoder
        resblock2task(task_queue, network.mid_block.resnets[0])
        attn2task(task_queue, network.mid_block.attentions[0])
        resblock2task(task_queue, network.mid_block.resnets[1])
        for up_block in network.up_blocks:
            up_block: UpDecoderBlock2D
            for resnet in up_block.resnets:
                resblock2task(task_queue, resnet)
            if up_block.upsamplers is not None:
                for upsample in up_block.upsamplers:
                    task_queue.append(("upsample", upsample))
    else:
        network: Encoder
        for down_block in network.down_blocks:
            down_block: DownEncoderBlock2D
            for resnet in down_block.resnets:
                resblock2task(task_queue, resnet)
            if down_block.downsamplers is not None:
                for downsample in down_block.downsamplers:
                    task_queue.append(("downsample", downsample))
        resblock2task(task_queue, network.mid_block.resnets[0])
        attn2task(task_queue, network.mid_block.attentions[0])
        resblock2task(task_queue, network.mid_block.resnets[1])


def build_task_queue(network: Decoder | Encoder, is_decoder: bool) -> List:
    task_queue = []
    task_queue.append(("conv_in", network.conv_in))
    build_sampling(task_queue, network, is_decoder)
    task_queue.append(("pre_norm", network.conv_norm_out))
    task_queue.append(("silu", inplace_nonlinearity))
    task_queue.append(("conv_out", network.conv_out))

    return task_queue


def clone_task_queue(task_queue):
    return [[item for item in task] for task in task_queue]


def get_var_mean(input, num_groups, eps=1e-6):
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c / num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:]
    )
    var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c / num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:]
    )

    out = F.batch_norm(
        input_reshaped,
        mean,
        var,
        weight=None,
        bias=None,
        training=False,
        momentum=0,
        eps=eps,
    )
    out = out.view(b, c, *input.size()[2:])

    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


class GroupNormParam:

    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(tile.shape[2] * tile.shape[3])
        if hasattr(layer, "weight"):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def summary(self, config: Config):
        if len(self.var_list) == 0:
            return None

        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels: torch.Tensor = (
            torch.tensor(
                self.pixel_list, dtype=torch.float32, device=config.compute.device
            )
            / max_value
        )
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(1) / sum_pixels
        var = torch.sum(var * pixels, dim=0)
        mean = torch.sum(mean * pixels, dim=0)
        return lambda x: custom_group_norm(x, 32, mean, var, self.weight, self.bias)

    @staticmethod
    def from_tile(tile, norm):
        """
        create a function from a single tile without summary
        """
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
            # if it is a macbook, we need to convert back to float16
            if var.device.type == "mps":
                # clamp to avoid overflow
                var = torch.clamp(var, 0, 60000)
                var = var.half()
                mean = mean.half()
        if hasattr(norm, "weight"):
            weight = norm.weight
            bias = norm.bias
        else:
            weight = None
            bias = None

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, 32, mean, var, weight, bias, 1e-6)

        return group_norm_func


class VAEHook:
    def __init__(
        self,
        network: Decoder | Encoder,
    ) -> None:
        self.network = network
        self.is_decoder = isinstance(network, Decoder)
        self.pad = 11 if self.is_decoder else 32  # magic number

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        config = get_config()
        # print(self.pad * 2 + config.vae.tile_size)
        # print(max(H, W))
        # print(config.vae.tile_size)

        if max(H, W) <= self.pad * 2 + config.vae.tile_size:
            return self.network.original_forward(x)
        else:
            return self.vae_tile_forward(x, config)

    def _tile_size(self, lower: int, upper: int) -> int:
        divider = 32
        while divider >= 2:
            remainer = lower % divider
            if remainer == 0:
                return lower
            candidate = lower - remainer + divider
            if candidate <= upper:
                return candidate
            divider //= 2
        return lower

    def _split(self, h: int, w: int, config: Config) -> Tuple[List[int], List[int]]:
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = config.vae.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # at least 1, since long and thin images are a possibility
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self._tile_size(real_tile_height, tile_size)
        real_tile_width = self._tile_size(real_tile_width, tile_size)

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # don't x // 8, since config handles that for us already.
                output_bbox = [x * 8 if self.is_decoder else x for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                tile_input_bboxes.append(
                    [
                        max(0, input_bbox[0] - pad),
                        min(w, input_bbox[1] + pad),
                        max(0, input_bbox[2] - pad),
                        min(h, input_bbox[3] + pad),
                    ]
                )
        return tile_input_bboxes, tile_output_bboxes

    @torch.no_grad()
    def _group_norm(self, z: torch.Tensor, task_queue: List, config: Config) -> bool:
        device = config.compute.device
        tile = z
        last_id = len(task_queue) - 1
        while last_id >= 0 and task_queue[last_id][0] != "pre_norm":
            last_id -= 1
        if last_id <= 0 or task_queue[last_id][0] != "pre_norm":
            raise ValueError("No group norm found in task queue.")
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == "pre_norm":
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ("apply_norm", group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == "store_res":
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != "add_res":
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == "add_res":
                tile += task[1].to(device=device)
                task[1] = None
            elif config.vae.tiling_encoder_color_fix and task[0] == "downsample":
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == "store_res":
                        task_queue[j] = ("store_res_cpu", task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                self._test_for_nans(tile)
            except:  # noqa
                print("Nan in fastpath. Reverting to slow path.")
                return False
        raise ValueError("Shouldn't reach this point.")

    @torch.no_grad()
    def vae_tile_forward(self, z: torch.Tensor, config: Config) -> torch.Tensor:
        device = config.compute.device
        dtype = z.dtype

        net = self.network
        tile_size = config.vae.tile_size
        is_decoder = self.is_decoder

        z = z.detach()
        N, _, height, width = z.shape

        in_bboxes, out_bboxes = self._split(height, width, config)
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[
                :, :, input_bbox[2] : input_bbox[3], input_bbox[0] : input_bbox[1]
            ].cpu()
            tiles.append(tile)
        num_tiles = len(tiles)
        num_completed = 0

        single_task_queue = build_task_queue(net, is_decoder)
        # print(list(map(lambda x: x[0], single_task_queue)))

        if config.vae.use_tiling_fastpath:
            scale_factor = tile_size / max(height, width)
            z = z.to(device=device)
            downsampled_z: torch.Tensor = F.interpolate(
                z, scale_factor=scale_factor, mode="nearest-exact"
            )

            # print(
            #     f"Fastpath enabled: estimating group norm on {downsampled_z.shape[3] * 8}x{downsampled_z.shape[2] * 8} image"
            # )

            _std, _mean = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
            std, mean = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean) / std * _std + _mean
            del _std, _mean, std, mean

            downsampled_z = torch.clamp_(downsampled_z, min=z.min(), max=z.max())
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self._group_norm(downsampled_z, estimate_task_queue, config):
                single_task_queue = estimate_task_queue
            del downsampled_z
        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]
        result = None
        # result_approx = torch.cat(
        #     [
        #         F.interpolate(
        #             cheap_approximation(x),
        #             scale_factor=8,
        #             mode="nearest-exact",
        #         )
        #         for x in z
        #     ],
        #     dim=0,
        # ).cpu()
        del z

        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc="Tiling")
        forward = True
        while True:
            group_norm_param = GroupNormParam()
            for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                tile = tiles[i].to(device=device)
                input_bbox = in_bboxes[i]
                task_queue = task_queues[i]

                while len(task_queue) > 0:
                    task = task_queue.pop(0)
                    if task[0] == "pre_norm":
                        group_norm_param.add_tile(tile, task[1])
                        break
                    elif task[0] == "store_res" or task[0] == "store_res_cpu":
                        task_id = 0
                        res = task[1](tile)
                        if (
                            not config.vae.use_tiling_fastpath
                            or task[0] == "store_res_cpu"
                        ):
                            res = res.cpu()
                        while task_queue[task_id][0] != "add_res":
                            task_id += 1
                        task_queue[task_id][1] = res
                    elif task[0] == "add_res":
                        tile += task[1].to(device=device)
                        task[1] = None
                    else:
                        tile = task[1](tile)
                    pbar.update(1)
                self._test_for_nans(tile)

                if len(task_queue) == 0:
                    tiles[i] = None
                    num_completed += 1
                    if result is None:
                        result = torch.zeros(
                            (
                                N,
                                tile.shape[1],
                                height * 8 if is_decoder else height,
                                width * 8 if is_decoder else width,
                            ),
                            device=device,
                            requires_grad=False,
                        )
                    result[
                        :,
                        :,
                        out_bboxes[i][2] : out_bboxes[i][3],
                        out_bboxes[i][0] : out_bboxes[i][1],
                    ] = self._crop(tile, in_bboxes[i], out_bboxes[i])
                    del tile
                elif i == num_tiles - 1 and forward:
                    forward = False
                    tiles[i] = tile
                elif i == 0 and not forward:
                    forward = True
                    tiles[i] = tile
                else:
                    tiles[i] = tile.cpu()
                    del tile

            if num_completed == num_tiles:
                break

            group_norm_func = group_norm_param.summary(config)
            if group_norm_func is not None:
                for i in range(num_tiles):
                    task_queue = task_queues[i]
                    task_queue.insert(0, ("apply_norm", group_norm_func))
        pbar.close()
        return (
            result.to(dtype=dtype, device=device)
            # if result is not None
            # else result_approx.to(dtype=dtype, device=device)
        )

    def _test_for_nans(self, x: torch.Tensor) -> None:
        if torch.all(torch.isnan(x)).item():
            raise ValueError("NaN in fastpath. Reverting to slow path.")

    def _crop(self, x, input_bbox, target_bbox):
        padded_bbox = [i * 8 if self.is_decoder else i // 8 for i in input_bbox]
        margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
        return x[
            :, :, margin[2] : x.size(2) + margin[3], margin[0] : x.size(3) + margin[1]
        ]
