# utils/configs.py

from typing import Literal
from dataclasses import dataclass
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MODEL_PRESETS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 512
    rope_theta: float = 10000
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32"] = "float32"


@dataclass
class AdamWConfig:
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    batch_size: int = 4
    seed: int = 42


@dataclass
class BenchmarkConfig:
    warm_up_steps: int = 5
    execution_steps: int = 20
    mode: Literal["fwd", "fwdbwd", "fwdbwdopt"] = "fwdbwdopt"
    sizes: tuple[str, ...] = ("small", "medium", "large", "xl", "10B")
