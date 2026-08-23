import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn as nn
from .linear import Linear
from cs336_basics.transformer.functions import silu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "d_model"]) -> Float[Tensor, "d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))
