import torch.nn as nn
from .multihead_self_attention import MultiheadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
import torch
from torch import Tensor
from jaxtyping import Float


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
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device, dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self, in_features: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        y = in_features + self.attn(self.ln1(in_features))
        z = y + self.ffn(self.ln2(y))
        return z
