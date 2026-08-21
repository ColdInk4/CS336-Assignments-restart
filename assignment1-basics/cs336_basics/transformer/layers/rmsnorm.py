import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
import einx


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = (
            einx.sum("... [d_model] -> ... 1", x**2) / self.d_model + self.eps
        ).sqrt()
        result = einx.multiply(
            "... d_model, d_model -> ... d_model", x / rms, self.weight
        )

        return result.to(in_dtype)
