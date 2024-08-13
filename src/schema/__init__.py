import gc
import os
from pathlib import Path

from strawberry import type, field, mutation
import torch

from .generation import GenerationInput
from .generic import GenericResponse
from ..models import load_models, create_pipeline, FluxPipeline

_MODEL = None
_PIPELINE: FluxPipeline = None
_MODEL_FILE = Path("models.pt")
_IMAGES = Path("images/")

_IMAGES.mkdir(exist_ok=True, parents=True)


@type
class Mutations:
    @mutation
    def generate_image(self, input: GenerationInput) -> GenericResponse:
        if _PIPELINE is not None:
            with torch.inference_mode():
                image = _PIPELINE(
                    prompt=input.prompt,
                    neg_prompt=input.negative_prompt,
                    cfg=input.cfg,
                    guidance_scale=input.flux_cfg,
                    num_inference_steps=input.num_inference_steps,
                    width=input.width,
                    height=input.height,
                    generator=torch.Generator().manual_seed(input.seed),
                ).images[0]
                gc.collect()
                torch.cuda.empty_cache()
                filename = len(
                    [
                        filename
                        for filename in os.listdir(_IMAGES)
                        if filename.endswith(".png")
                    ]
                )
                image.save(_IMAGES / "{:05d}.png".format(filename))
                return GenericResponse(
                    message="localhost:8000/images/{:05d}.png".format(filename)
                )
        else:
            return GenericResponse(message="model not loaded.")


@type
class Queries:
    @field
    def model_loaded(self) -> GenericResponse:
        return GenericResponse(message="true" if _MODEL is not None else "false")

    @field
    def pipeline_loaded(self) -> GenericResponse:
        return GenericResponse(message="true" if _PIPELINE is not None else "false")

    @field
    def load_model(self) -> GenericResponse:
        global _MODEL, _PIPELINE
        if _PIPELINE is None:
            _MODEL = load_models(_MODEL_FILE)
            _PIPELINE = create_pipeline(_MODEL)
            gc.collect()
            torch.cuda.empty_cache()
            return GenericResponse(message="true")
        return GenericResponse(message="false")
