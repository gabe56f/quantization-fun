from pathlib import Path
import os
import gc

import torch

from src import fixes, models, config

prompt = "An anime drawing of a cute shiba inu"

fixes.apply_fixes()  # fix for qdtypes
config = config.get_config()

model = models.load_models()  # load models
model["vae"].enable_tiling()  # enable tiling

pipe = models.create_pipeline(model)  # create pipeline

generator = torch.Generator().manual_seed(123456)
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
