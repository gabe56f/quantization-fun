import json
from pathlib import Path

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,
)

from src import quantization
from src.config import get_config
from src.pipeline import FluxPipeline


def load_models(file: Path = None) -> dict:
    config = get_config()
    device = "cpu" if config.offload else config.device

    if file is not None and file.exists():
        state_dict = torch.load(file, mmap=True)

        transformer = _init_transformer(state_dict["transformer"]).to(device)
        text_encoder_2 = _init_text_encoder(state_dict["text_encoder_2"]).to(device)
    else:
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


def save_models(file: Path, models: dict):
    transformer = models["transformer"].state_dict()
    text_encoder_2 = models["text_encoder_2"].state_dict()
    with open(file, "wb") as f:
        torch.save(
            {
                "transformer": transformer,
                "text_encoder_2": text_encoder_2,
            },
            f,
        )


def create_pipeline(models: dict) -> FluxPipeline:
    config = get_config()
    pipe = FluxPipeline(**models)
    if config.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(config.device)
    return pipe


def _init_transformer(state_dict=None) -> FluxTransformer2DModel:
    config = get_config()

    if state_dict is not None:
        with torch.device("meta"):
            transformer_config = Path(
                hf_hub_download(
                    config.repo,
                    revision=config.revision,
                    subfolder="transformer",
                    filename="config.json",
                )
            )
            transformer_config = json.loads(transformer_config.read_text())
            model = FluxTransformer2DModel.from_config(transformer_config)
        model.load_state_dict(state_dict, assign=True)
        return model

    return FluxTransformer2DModel.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="transformer",
        torch_dtype=config.compute_dtype,
    )


def _init_text_encoder(state_dict=None):
    config = get_config()
    if state_dict is not None:
        with torch.device("meta"):
            encoder_config = Path(
                hf_hub_download(
                    config.repo,
                    revision=config.revision,
                    subfolder="text_encoder_2",
                    filename="config.json",
                )
            )
            encoder_config = json.loads(encoder_config.read_text())
            model = T5EncoderModel(T5Config(**encoder_config))
        model.load_state_dict(state_dict, assign=True)
        return model

    return T5EncoderModel.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="text_encoder_2",
        torch_dtype=config.compute_dtype,
    )
