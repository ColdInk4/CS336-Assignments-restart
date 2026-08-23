import torch.nn as nn
import torch
from torch import Tensor
from jaxtyping import Float, Int
from math import sin, cos


class RotaryPositionalEmbedding(nn.Module):
    pre_compute_sin: Float[Tensor, "max_seq_len d_k/2"]
    pre_compute_cos: Float[Tensor, "max_seq_len d_k/2"]

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        pre_compute_sin = torch.zeros(max_seq_len, d_k // 2, device=device)
        pre_compute_cos = torch.zeros(max_seq_len, d_k // 2, device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angle = i * 1.0 / (theta ** (2 * k / d_k))
                pre_compute_sin[i][k] = sin(angle)
                pre_compute_cos[i][k] = cos(angle)
        self.register_buffer("pre_compute_sin", pre_compute_sin, persistent=False)
        self.register_buffer("pre_compute_cos", pre_compute_cos, persistent=False)
        self.d_k = d_k

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        result = torch.zeros_like(x)
        for k in range(self.d_k // 2):
            cur_cos = self.pre_compute_cos[token_positions][..., k]
            cur_sin = self.pre_compute_sin[token_positions][..., k]

            result[..., 2 * k] = x[..., 2 * k] * cur_cos - x[..., 2 * k + 1] * cur_sin
            result[..., 2 * k + 1] = (
                x[..., 2 * k] * cur_sin + x[..., 2 * k + 1] * cur_cos
            )

        return result
