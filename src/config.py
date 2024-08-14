from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from dataclasses_json import dataclass_json
import torch

from src import quantization
from .utils import qdt_config, dtype_config, device_config, tile_config

_CONFIG = None


def save_on_change(x=None):
    def wrap(cls):
        def setattr(self, name, value):
            object.__setattr__(self, name, value)
            save_config()

        cls.__setattr__ = setattr
        return cls

    if x is None:
        return wrap
    return wrap(x)


@save_on_change
@dataclass_json
@dataclass
class VAEConfig:
    use_tiling: bool = True
    tile_size: int = field(default=64, metadata=tile_config())
    use_tiling_fastpath: bool = True
    tiling_encoder_color_fix: bool = True
    upcast: bool = False


@save_on_change
@dataclass_json
@dataclass
class ComputeConfig:
    dtype: torch.dtype = field(
        default_factory=lambda: torch.bfloat16,
        metadata=dtype_config(),
    )
    offload: Literal["none", "model", "sequential"] = "model"
    device: torch.device = field(
        default_factory=lambda: torch.device("cpu"),
        metadata=device_config(),
    )


@save_on_change
@dataclass_json
@dataclass
class ModelConfig:
    qdtype: quantization.qdtype = field(
        default_factory=lambda: quantization.qfloatx(2, 2),
        metadata=qdt_config(),
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


@save_on_change
@dataclass_json
@dataclass
class Config:
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    repo: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None

    transformer: ModelConfig = field(default_factory=ModelConfig)
    text_encoder: ModelConfig = field(
        default_factory=lambda: ModelConfig(skip=[], qdtype=quantization.qint4)
    )
    vae: VAEConfig = field(default_factory=VAEConfig)


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        p = Path("config.json")
        if p.exists():
            _CONFIG = Config.from_json(p.read_text("utf-8"))
        else:
            _CONFIG = Config()
            save_config()
    return _CONFIG


def save_config(config: Config = None) -> Config:  # noqa
    if _CONFIG is not None and config is None:
        config = _CONFIG

    if config is None:
        return None
    p = Path("config.json")
    p.write_text(config.to_json(indent=4, sort_keys=True), "utf-8")
    return config
