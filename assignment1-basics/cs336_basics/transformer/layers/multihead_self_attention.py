import torch.nn as nn
from .linear import Linear
from .rope import RotaryPositionalEmbedding
import torch
from ..functions import scaled_dot_product_attention
from torch import Tensor
from jaxtyping import Float
import einx
from typing import cast


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        super().__init__()

        d_k = d_model // num_heads
        d_v = d_k

        self.num_heads = num_heads

        self.q_proj = Linear(d_model, num_heads * d_k, device, dtype)
        self.k_proj = Linear(d_model, num_heads * d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * d_v, device, dtype)
        self.output_proj = Linear(num_heads * d_v, d_model, device, dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)
        else:
            self.rope = None

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:

        seq_len = in_features.size(-2)

        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)

        Q_heads = cast(
            Tensor,
            einx.id(
                " ... sequence_length (num_heads d_k)-> ... num_heads sequence_length d_k",
                q,
                num_heads=self.num_heads,
            ),
        )
        K_heads = cast(
            Tensor,
            einx.id(
                " ... sequence_length (num_heads d_k)-> ... num_heads sequence_length d_k",
                k,
                num_heads=self.num_heads,
            ),
        )
        V_heads = cast(
            Tensor,
            einx.id(
                " ... sequence_length (num_heads d_v)-> ... num_heads sequence_length d_v",
                v,
                num_heads=self.num_heads,
            ),
        )

        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=in_features.device)
            Q_heads = self.rope(Q_heads, token_positions)
            K_heads = self.rope(K_heads, token_positions)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool)
        )

        head_outputs = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        merged_heads = cast(
            Tensor,
            einx.id(
                "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)",
                head_outputs,
            ),
        )

        result = self.output_proj(merged_heads)
        return result
