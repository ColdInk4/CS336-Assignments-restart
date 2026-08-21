import torch
from torch import Tensor
import torch.nn as nn
from math import sqrt
from jaxtyping import Float
import einx


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        mean = 0
        std = sqrt(2.0 / (in_features + out_features))
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, dtype=dtype, device=device),
                mean,
                std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einx.dot("d_out [d_in], ... [d_in] -> ... d_out", self.weight, x)
