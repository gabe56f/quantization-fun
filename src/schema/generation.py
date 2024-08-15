from typing import List, TYPE_CHECKING

from strawberry import type, field, input
from ..utils import convert_image_to_base64
import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers.image_processor import VaeImageProcessor


@input
@type
class GenerationInput:
    prompt: str = field(description="The prompt to generate an image from.")
    negative_prompt: str = field(
        description="The negative prompt to generate an image from. Only needed if cfg is above 1.",
        default="",
    )
    cfg: float = field(description="Real CFG value.", default=1.0)
    flux_cfg: float = field(
        description="CFG value for the Flux model. Only has a use on .dev models.",
        default=3.5,
    )
    num_inference_steps: int = field(
        description="Number of inference steps to take.", default=28
    )
    width: int = field(description="Width of the image.", default=1024)
    height: int = field(description="Height of the image.", default=1024)
    seed: int = field(description="Seed for the random number generator.", default=-1)
    batch_size: int = field(
        description="Number of images to generate per prompt.", default=1
    )


@type
class GenerationOutput:
    id: str = field(description="ID of the generation.")
    images: List[str] = field(
        description="List of image blobs in the case of an ongoing generation, otherwise URLs."
    )
    step: int = field(description="Current step of the generation.")
    total_steps: int = field(description="Total amount of steps for the generation.")

    @classmethod
    def create_from_pil(
        cls, id: str, image: Image.Image, step: int, total_steps: int
    ) -> "GenerationOutput":
        if not isinstance(image, list):
            image = [image]
        images = [convert_image_to_base64(i) for i in image]
        return cls(
            id=id,
            images=images,
            step=step,
            total_steps=total_steps,
        )

    @classmethod
    def create_from_tensors(
        cls,
        id: str,
        tensor: torch.Tensor,
        step: int,
        total_steps: int,
        image_processor: "VaeImageProcessor",
    ) -> "GenerationOutput":
        image = image_processor.postprocess(tensor, output_type="pil")
        return cls.create_from_pil(id, image, step, total_steps)
