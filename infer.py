from pathlib import Path
import os
import gc

import torch

from src import fixes, models, config

prompt = "A town full of people dressed in Victorian era clothing."
neg_prompt = "Houses"

fixes.apply_fixes()  # fix for qdtypes
config = config.get_config()

model_file = Path("models.pt")

model = models.load_models(model_file)  # load models
model["vae"].enable_tiling()  # enable tiling

if not model_file.exists():
    models.save_models(model_file, model)

pipe = models.create_pipeline(model)  # create pipeline


def disable_cfg(step: int, max_steps: int, timestep: torch.Tensor) -> bool:
    return step > max_steps // 4 or timestep.item() < 0.1


generator = torch.Generator().manual_seed(123456)
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        neg_prompt=neg_prompt,
        width=1024,
        height=1024,
        num_inference_steps=28,
        cfg=8,
        cfg_disable=disable_cfg,
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
