from pathlib import Path
import os
import gc

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from src import quantization, pipeline, fixes
from src.quantization import quantize_model

fixes.apply_fixes()  # fix for fp6
dtype = torch.bfloat16
transformer_qdtype = quantization.qfloatx(2, 2)
text_encoder_qdtype = quantization.qfloat8

can_offload = True
bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "refs/pr/3"

prompt = "An anime drawing of a tall woman with long white and green hair standing on a mountain ledge looking at a pine forest whilst it is snowing"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    bfl_repo, subfolder="scheduler", revision=revision, torch_dtype=dtype
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=dtype
)
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=dtype
)


def init_transformer() -> FluxTransformer2DModel:
    return FluxTransformer2DModel.from_pretrained(
        bfl_repo, revision=revision, subfolder="transformer", torch_dtype=dtype
    )


transformer = quantize_model(
    init_transformer,
    transformer_qdtype,
    skip=["proj_out", "x_embedder", "norm_out", "context_embedder"],
)
if can_offload:
    try:
        transformer.to("cpu")
    except:  # noqa
        can_offload = False
else:
    transformer.to("cuda")


def init_text_encoder():
    return T5EncoderModel.from_pretrained(
        bfl_repo,
        subfolder="text_encoder_2",
        revision=revision,
        torch_dtype=dtype,
    )


text_encoder_2 = quantize_model(init_text_encoder, text_encoder_qdtype)
if can_offload:
    try:
        text_encoder_2.to("cpu")
    except:  # noqa
        can_offload = False
else:
    text_encoder_2.to("cuda")
torch.cuda.empty_cache()

tokenizer_2 = T5TokenizerFast.from_pretrained(
    bfl_repo, subfolder="tokenizer_2", revision=revision, torch_dtype=dtype
)

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
    bfl_repo, subfolder="vae", revision=revision, torch_dtype=dtype
)
vae.enable_tiling()
pipe: pipeline.FluxPipeline = pipeline.FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)
if can_offload:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")

generator = torch.Generator().manual_seed(12345)
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=3.5,
        output_type="pil",
    ).images[0]
    gc.collect()
    torch.cuda.empty_cache()
    filename = len(
        [filename for filename in os.listdir(Path()) if filename.endswith(".png")]
    )
    image.save("{}/{:05d}.png".format(Path().as_posix(), filename))
