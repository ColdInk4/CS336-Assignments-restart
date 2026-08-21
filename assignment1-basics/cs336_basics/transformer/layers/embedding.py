import torch
from torch import Tensor
import torch.nn as nn
from math import sqrt
from jaxtyping import Float, Int
import einx


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        mean = 0
        std = 1
        self.weight: Int[Tensor, "vocab_size d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
                mean,
                std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return einx.get_at(
            "[num_embeddings] embedding_dim, ... -> ... embedding_dim",
            self.weight,
            token_ids,
        )
