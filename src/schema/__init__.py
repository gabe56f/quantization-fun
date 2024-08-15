import gc
import os
from pathlib import Path
from typing import AsyncGenerator, TYPE_CHECKING

from strawberry import type, field, mutation, subscription
import torch

from .generation import GenerationInput, GenerationOutput
from .generic import GenericResponse

if TYPE_CHECKING:
    from ..models import FluxPipeline

_MODEL = None
_PIPELINE: "FluxPipeline" = None
_MODEL_FILE = Path("models.pt")
_IMAGES = Path("images/")

_IMAGES.mkdir(exist_ok=True, parents=True)


@type
class Mutations:
    @mutation
    async def generate_image(
        self, input: GenerationInput
    ) -> AsyncGenerator[GenericResponse, None]:
        if _PIPELINE is not None:
            files = len(os.listdir(_IMAGES))
            id = "{:05d}".format(files)
            with torch.inference_mode():
                async for x in _PIPELINE(
                    id,
                    prompt=input.prompt,
                    neg_prompt=input.negative_prompt,
                    cfg=input.cfg,
                    guidance_scale=input.flux_cfg,
                    num_inference_steps=input.num_inference_steps,
                    width=input.width,
                    num_images_per_prompt=input.batch_size,
                    height=input.height,
                    generator=torch.Generator().manual_seed(input.seed),
                ):
                    x: GenerationOutput
                    pass
                yield GenericResponse(message=x.images[0])
        else:
            yield GenericResponse(message="model not loaded.")


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
            from ..models import load_models, create_pipeline

            _MODEL = load_models(_MODEL_FILE)
            _PIPELINE = create_pipeline(_MODEL)
            gc.collect()
            torch.cuda.empty_cache()
            return GenericResponse(message="true")
        return GenericResponse(message="false")


@type
class Subscriptions:
    @subscription
    async def generate_image_and_watch(
        self, input: GenerationInput
    ) -> AsyncGenerator[GenerationOutput, None]:
        if _PIPELINE is not None:
            files = len(os.listdir(_IMAGES))
            id = "{:05d}".format(files)

            with torch.inference_mode():
                async for x in _PIPELINE(
                    id,
                    prompt=input.prompt,
                    neg_prompt=input.negative_prompt,
                    cfg=input.cfg,
                    guidance_scale=input.flux_cfg,
                    num_inference_steps=input.num_inference_steps,
                    width=input.width,
                    height=input.height,
                    num_images_per_prompt=input.batch_size,
                    generator=torch.Generator().manual_seed(input.seed),
                ):
                    x: GenerationOutput
                    yield x
                    pass
        else:
            print("load model!!!")
            yield GenerationOutput(
                images=[], id="", step=0, total_steps=input.num_inference_steps
            )


__all__ = ["Mutations", "Queries", "Subscriptions", "GenerationOutput", "_IMAGES"]
