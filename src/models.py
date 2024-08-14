from dataclasses import dataclass, field
import json
from pathlib import Path

from dataclasses_json import dataclass_json
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

from src import quantization, fixes, utils
from src.config import get_config
from src.pipeline import FluxPipeline


def load_models(file: Path = None) -> dict:
    config = get_config()
    device = "cpu" if config.compute.offload != "none" else config.compute.device

    if file is not None and file.exists():
        state_dict = torch.load(file, mmap=True)

        metadata: Metadata = Metadata.from_json(state_dict.get("metadata", "{}"))
        requested_metadata: Metadata = Metadata.get_from_config()

        if (
            metadata.transformer_qdtype != requested_metadata.transformer_qdtype
            or metadata.transformer_skip != requested_metadata.transformer_skip
        ):
            print("Requantizing transformer, since saved model doesn't match config.")

            transformer = quantization.quantize_model(
                _init_transformer,
                config.transformer.qdtype,
                device=device,
                skip=config.transformer.skip,
                strict_skip=config.transformer.strict_skip,
            )
        else:
            transformer = _init_transformer(state_dict["transformer"]).to(device)

        if (
            metadata.text_encoder_qdtype != requested_metadata.text_encoder_qdtype
            or metadata.text_encoder_skip != requested_metadata.text_encoder_skip
        ):
            print("Requantizing text_encoder, since saved model doesn't match config.")

            text_encoder_2 = quantization.quantize_model(
                _init_text_encoder,
                config.text_encoder.qdtype,
                device=device,
                skip=config.text_encoder.skip,
                strict_skip=config.text_encoder.strict_skip,
            )
        else:
            text_encoder_2 = _init_text_encoder(state_dict["text_encoder_2"]).to(device)
    else:
        transformer = quantization.quantize_model(
            _init_transformer,
            config.transformer.qdtype,
            device=device,
            skip=config.transformer.skip,
            strict_skip=config.transformer.strict_skip,
        )

        text_encoder_2 = quantization.quantize_model(
            _init_text_encoder,
            config.text_encoder.qdtype,
            device=device,
            skip=config.text_encoder.skip,
            strict_skip=config.text_encoder.strict_skip,
        )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="scheduler",
        torch_dtype=config.compute.dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=config.compute.dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=config.compute.dtype
    )

    tokenizer_2 = T5TokenizerFast.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="tokenizer_2",
        torch_dtype=config.compute.dtype,
    )

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.repo,
        revision=config.revision,
        subfolder="vae",
        torch_dtype=config.compute.dtype,
        force_upcast=False,
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


def _apply_transformations(models: dict) -> dict:
    config = get_config()
    # models["vae"].enable_tiling()
    fixes.patch_vae(models["vae"], config)
    return models


def save_models(file: Path, models: dict):
    transformer = models["transformer"].state_dict()
    text_encoder_2 = models["text_encoder_2"].state_dict()
    metadata = Metadata.get_from_config()
    with open(file, "wb") as f:
        torch.save(
            {
                "metadata": metadata.to_json(),
                "transformer": transformer,
                "text_encoder_2": text_encoder_2,
            },
            f,
        )


def create_pipeline(models: dict) -> FluxPipeline:
    config = get_config()
    models = _apply_transformations(models)
    pipe = FluxPipeline(**models)
    if config.compute.offload == "model":
        pipe.enable_model_cpu_offload(device=config.compute.device)
    elif config.compute.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=config.compute.device)
    else:
        pipe.to(config.compute.device)
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
        torch_dtype=config.compute.dtype,
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
        torch_dtype=config.compute.dtype,
    )


@dataclass_json
@dataclass
class Metadata:
    transformer_qdtype: quantization.qdtype = field(
        default_factory=lambda: quantization.qfloatx(2, 2),
        metadata=utils.qdt_config(),
    )

    transformer_skip: list = field(
        default_factory=lambda: [
            "proj_out",
            "x_embedder",
            "norm_out",
            "context_embedder",
        ]
    )

    text_encoder_qdtype: quantization.qdtype = field(
        default_factory=lambda: quantization.qint4,
        metadata=utils.qdt_config(),
    )

    text_encoder_skip: list = field(default_factory=list)

    @staticmethod
    def get_from_config() -> "Metadata":
        config = get_config()
        return Metadata(
            transformer_qdtype=config.transformer.qdtype,
            transformer_skip=config.transformer.skip,
            text_encoder_qdtype=config.text_encoder.qdtype,
            text_encoder_skip=config.text_encoder.skip,
        )
