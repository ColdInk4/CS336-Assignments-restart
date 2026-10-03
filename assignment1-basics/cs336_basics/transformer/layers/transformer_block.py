import torch.nn as nn
from .multihead_self_attention import MultiheadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rope import RotaryPositionalEmbedding
import torch
from torch import Tensor
from jaxtyping import Float
import torch.cuda.nvtx as nvtx


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if num_heads <= 0:
            raise ValueError(f"num_heads must be greater than 0, but got {num_heads}")

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        super().__init__()

        d_k = d_model // num_heads
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be an even number, but got {d_k}")

        rope = RotaryPositionalEmbedding(
            theta, d_k, max_seq_len, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    @nvtx.range("Transformer Block")
    def forward(
        self, in_features: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        y = in_features + self.attn(self.ln1(in_features))
        z = y + self.ffn(self.ln2(y))
        return z
