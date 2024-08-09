from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dataclasses_json import dataclass_json, config
import torch

from src import quantization


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


def encode_dtype(dtype: torch.dtype) -> str:
    return dtype.__repr__().replace("torch.", "")


def decode_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)


@dataclass_json
@dataclass
class Config:
    compute_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata=config(
            encoder=encode_dtype,
            decoder=decode_dtype,
        ),
    )

    offload: bool = True
    repo: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = "refs/pr/3"

    transformer_qdtype: quantization.qdtype = field(
        default=quantization.qfloatx(2, 2),
        metadata=config(
            encoder=encode_qdt,
            decoder=decode_qdt,
        ),
    )
    transformer_skip: List[str] = field(
        default_factory=lambda: [
            "proj_out",
            "x_embedder",
            "norm_out",
            "context_embedder",
        ]
    )
    transformer_strict_skip: bool = False

    text_encoder_qdtype: quantization.qdtype = field(
        default=quantization.qint4,
        metadata=config(
            encoder=encode_qdt,
            decoder=decode_qdt,
        ),
    )
    text_encoder_skip: List[str] = field(default_factory=list)
    text_encoder_strict_skip: bool = False

    device: str = "cuda"


def get_config() -> Config:
    p = Path("config.json")
    if p.exists():
        return Config.from_json(p.read_text("utf-8"))
    config = Config()
    config.transformer_skip = [
        "proj_out",
        "x_embedder",
        "norm_out",
        "context_embedder",
    ]
    config.text_encoder_skip = []
    return save_config(config)


def save_config(config: Config) -> Config:
    p = Path("config.json")
    p.write_text(config.to_json(indent=2), "utf-8")
    return config
