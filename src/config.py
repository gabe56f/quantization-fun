from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

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


def encode_device(device: torch.device) -> str:
    return f"{device.type}:{device.index or 0}"


def decode_device(device: str) -> torch.device:
    return torch.device(device)


@dataclass_json
@dataclass
class ComputeConfig:
    dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata=config(
            encoder=encode_dtype,
            decoder=decode_dtype,
        ),
    )
    offload: Literal["none", "model", "sequential"] = "model"
    device: torch.device = field(
        default=torch.device("cpu"),
        metadata=config(
            encoder=encode_device,
            decoder=decode_device,
        ),
    )


@dataclass_json
@dataclass
class ModelConfig:
    qdtype: quantization.qdtype = field(
        default=quantization.qfloatx(2, 2),
        metadata=config(
            encoder=encode_qdt,
            decoder=decode_qdt,
        ),
    )
    skip: List[str] = field(
        default_factory=lambda: [
            "proj_out",
            "x_embedder",
            "norm_out",
            "context_embedder",
        ]
    )
    strict_skip: bool = False


@dataclass_json
@dataclass
class Config:
    compute: ComputeConfig = ComputeConfig()
    repo: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None

    transformer: ModelConfig = field(default_factory=ModelConfig)
    text_encoder: ModelConfig = field(
        default_factory=lambda: ModelConfig(skip=[], qdtype=quantization.qint4)
    )


def get_config() -> Config:
    p = Path("config.json")
    if p.exists():
        return Config.from_json(p.read_text("utf-8"))

    config = Config()
    return save_config(config)


def save_config(config: Config) -> Config:
    p = Path("config.json")
    p.write_text(config.to_json(indent=4, sort_keys=True), "utf-8")
    return config
