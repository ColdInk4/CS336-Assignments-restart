import torch.nn as nn
import torch
from torch import Tensor
from jaxtyping import Float, Int
import einx
from typing import cast


class RotaryPositionalEmbedding(nn.Module):
    pre_compute_sin: Float[Tensor, "max_seq_len half"]
    pre_compute_cos: Float[Tensor, "max_seq_len half"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        i_raw = torch.arange(max_seq_len, dtype=dtype, device=device)
        k_raw = torch.arange(d_k // 2, dtype=dtype, device=device)
        i = cast(Tensor, einx.id("max_seq_len -> max_seq_len 1", i_raw))
        k = cast(Tensor, einx.id("half -> 1 half", k_raw))
        angle = i / (theta ** (2 * k / d_k))
        pre_compute_sin = torch.sin(angle)
        pre_compute_cos = torch.cos(angle)
        self.register_buffer("pre_compute_sin", pre_compute_sin, persistent=False)
        self.register_buffer("pre_compute_cos", pre_compute_cos, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        x_reshaped = cast(
            Tensor,
            einx.id("... seq_len (half pair) -> ... seq_len half pair", x, pair=2),
        )
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        sin_vec: Float[Tensor, "... seq_len half"] = self.pre_compute_sin[
            token_positions
        ]
        cos_vec: Float[Tensor, "... seq_len half"] = self.pre_compute_cos[
            token_positions
        ]
        y0 = x0 * cos_vec - x1 * sin_vec
        y1 = x0 * sin_vec + x1 * cos_vec
        result_reshaped = cast(
            Tensor,
            einx.id(
                "... seq_len half, ... seq_len half -> ... seq_len half (1 + 1)",
                y0,
                y1,
            ),
        )
        result = cast(
            Tensor,
            einx.id(
                "... seq_len half pair -> ... seq_len (half pair)",
                result_reshaped,
                pair=2,
            ),
        )
        return result
