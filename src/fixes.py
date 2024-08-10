from typing import TYPE_CHECKING
import torch

from .utils import is_available

if TYPE_CHECKING:
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.prototype.uintx import UintxTensor
    from torchao.prototype.quant_llm import QuantLlmLinearWeight


aten = torch.ops.aten


def _get_to_kwargs(self: torch.nn.Module, *args, **kwargs):
    device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    device = self.device if device is None else device
    dtype = self.dtype if dtype is None else dtype
    memory_format = (
        memory_format if memory_format is not None else torch.preserve_format
    )
    kwargs = {
        "device": device,
        "dtype": dtype,
        "memory_format": memory_format,
    }
    return kwargs


def apply_fixes():
    if is_available("torchao.prototype.quant_llm"):
        from torchao.prototype.quant_llm.quant_llm import (
            QuantLlmLinearWeight,
            _SPLIT_K_MAP,
            quant_llm_linear,
        )

        # Stinky fix for aten::to::dtype_layout on QuantLlmLinearWeight
        QuantLlmLinearWeight.to = move_fpx

        # Even stinkier fix for aten::_has[...] on QuantLlmLinearWeight
        @QuantLlmLinearWeight.implements(aten._has_compatible_shallow_copy_type.default)
        def _(f, types, *args, **kwargs):
            return False

        # With large enough tensors, this can become an issue, so let's fix it
        # May make certain things slower.
        @QuantLlmLinearWeight.implements(torch.nn.functional.linear)
        def _(func, types, args, kwargs):
            act: torch.Tensor = args[0]
            weight = args[1]
            bias = args[2] if len(args) >= 3 else None
            assert isinstance(weight, QuantLlmLinearWeight)

            out_dim, in_dim = weight.shape
            act_reshaped = act.contiguous().view(-1, in_dim).half()

            # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
            bsize = act_reshaped.shape[0]
            splitK = (
                _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1
            )

            out = quant_llm_linear(
                weight.ebits,
                weight.mbits,
                act_reshaped,
                weight.fpx_data,
                weight.scale,
                splitK=splitK,
            )

            if bias is not None:
                out += bias

            return out.view(*act.shape[:-1], out_dim).to(act.dtype)

    if is_available("torchao.prototype.uintx"):
        from torchao.prototype.uintx import UintxTensor

        UintxTensor.to = move_intx

        @UintxTensor.implements(aten._has_compatible_shallow_copy_type.default)
        def _(f, types, *args, **kwargs):
            return False

    if is_available("torchao.dtypes.affine_quantized_tensor"):
        from torchao.dtypes.affine_quantized_tensor import (
            AffineQuantizedTensor,
            TensorCoreTiledAQTLayout,
        )

        TensorCoreTiledAQTLayout.to = move_aqt

        @AffineQuantizedTensor.implements(
            aten._has_compatible_shallow_copy_type.default
        )
        def _(f, types, *args, **kwargs):
            return False


def move_aqt(self: "AffineQuantizedTensor", *args, **kwargs):
    kwargs = self._get_to_kwargs(*args, **kwargs)  # use builtin
    device = kwargs["device"]

    # couldn't care less, won't be executing on anything besides "cuda"
    # if not is_device("cuda", device):
    #     raise ValueError(f"TensorCoreTiledAQTLayout is only available for cuda device, can't convert to {device}")

    return self.__class__(
        self.packed_weight.to(device),
        self.scale_and_zero.to(device),
        self.transposed,
        self.layout_type,
    )


def move_fpx(self: "QuantLlmLinearWeight", *args, **kwargs):
    kwargs = _get_to_kwargs(self, *args, **kwargs)

    device = kwargs.pop("device")
    kwargs.pop("memory_format")  # unsupported

    return self.__class__(
        self.fpx_data.to(device=device),
        self.scale.to(device=device),
        self.ebits,
        self.mbits,
    )


def move_intx(self: "UintxTensor", *args, **kwargs):
    kwargs = _get_to_kwargs(self, *args, **kwargs)

    device = kwargs.pop("device")
    kwargs.pop("memory_format")  # unsupported

    return self.__class__(
        [x.to(device) for x in self.get_shards()],
        self.packed_shape,
        self.bit_size,
        self.pack_dim,
    )
