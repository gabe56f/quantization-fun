from typing import Literal
import importlib.util
import base64
from io import BytesIO

from dataclasses_json import config
import torch
from PIL import Image

from . import quantization


ImageFormats = Literal["png", "webp", "jpg"]


def is_available(pkg):
    return importlib.util.find_spec(pkg) is not None


def uintx_version() -> int:
    """
    1 - no uintx
    2 - uintx-prototype
    3 - uintx-dtype
    """
    uintx_proto = is_available("torchao.prototype.uintx")
    uintx_dtype = is_available("torchao.dtypes.uintx")
    return 1 if not (uintx_dtype or uintx_proto) else 2 if uintx_proto else 3


def qdt_config() -> dict:
    def encode_qdt(qdt: quantization.qdtype) -> str:
        return qdt.q

    def decode_qdt(qdt: str) -> quantization.qdtype:
        if "_" in qdt:
            qdt = qdt.replace("_", "")
            if qdt.startswith("float"):
                qdt = qdt.replace("float", "")
                base = quantization.qfloatx
            else:
                qdt = qdt.replace("int", "")
                base = quantization.qintx
            return base(*list(map(int, qdt.split("x"))))
        return getattr(quantization, f"q{qdt}")

    return config(
        encoder=encode_qdt,
        decoder=decode_qdt,
    )


def dtype_config() -> dict:
    def encode_dtype(dtype: torch.dtype) -> str:
        return dtype.__repr__().replace("torch.", "")

    def decode_dtype(dtype: str) -> torch.dtype:
        return getattr(torch, dtype)

    return config(
        encoder=encode_dtype,
        decoder=decode_dtype,
    )


def device_config() -> dict:
    def encode_device(device: torch.device) -> str:
        return f"{device.type}:{device.index or 0}"

    def decode_device(device: str) -> torch.device:
        return torch.device(device)

    return config(
        encoder=encode_device,
        decoder=decode_device,
    )


def tile_config() -> dict:
    def encode_tile(tile: int) -> str:
        return str(tile * 8)

    def decode_tile(tile: str) -> int:
        return int(tile) // 8

    return config(
        encoder=encode_tile,
        decoder=decode_tile,
    )


def convert_image_to_stream(
    image: Image.Image, quality: int = 95, _format: ImageFormats = "webp"
) -> BytesIO:
    "Convert an image to a stream of bytes"

    stream = BytesIO()
    image.save(stream, format=_format, quality=quality)
    stream.seek(0)
    return stream


def convert_image_to_base64(
    image: Image.Image,
    quality: int = 95,
    image_format: ImageFormats = "webp",
    prefix_js: bool = True,
) -> str:
    "Convert an image to a base64 string"

    stream = convert_image_to_stream(image, quality=quality)
    if prefix_js:
        prefix = f"data:image/{image_format};base64,"
    else:
        prefix = ""
    return prefix + base64.b64encode(stream.read()).decode("utf-8")
