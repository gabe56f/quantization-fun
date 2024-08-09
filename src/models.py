from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from src import quantization
from src.config import get_config
from src.pipeline import FluxPipeline


def load_models() -> dict:
    config = get_config()
    device = "cpu" if config.offload else config.device

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="scheduler",
        torch_dtype=config.compute_dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=config.compute_dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=config.compute_dtype
    )

    transformer = quantization.quantize_model(
        _init_transformer,
        config.transformer_qdtype,
        device=device,
        skip=config.transformer_skip,
        strict_skip=config.transformer_strict_skip,
    )

    text_encoder_2 = quantization.quantize_model(
        _init_text_encoder,
        config.text_encoder_qdtype,
        device=device,
        skip=config.text_encoder_skip,
        strict_skip=config.text_encoder_strict_skip,
    )

    tokenizer_2 = T5TokenizerFast.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="tokenizer_2",
        torch_dtype=config.compute_dtype,
    )

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="vae",
        torch_dtype=config.compute_dtype,
    )

    return {
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "text_encoder_2": text_encoder_2,
        "tokenizer_2": tokenizer_2,
        "transformer": transformer,
        "vae": vae,
    }


def create_pipeline(models: dict) -> FluxPipeline:
    config = get_config()
    pipe = FluxPipeline(**models)
    if config.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(config.device)
    return pipe


def _init_transformer() -> FluxTransformer2DModel:
    config = get_config()
    return FluxTransformer2DModel.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="transformer",
        torch_dtype=config.compute_dtype,
    )


def _init_text_encoder():
    config = get_config()
    return T5EncoderModel.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="text_encoder_2",
        torch_dtype=config.compute_dtype,
    )
