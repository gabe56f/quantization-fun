from typing import Optional

from strawberry import type, field, input


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
    seed: Optional[int] = field(
        description="Seed for the random number generator.", default=-1
    )
