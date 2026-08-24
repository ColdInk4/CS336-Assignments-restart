import torch.nn as nn
from .layers import Embedding, TransformerBlock, RMSNorm, Linear
from torch import Tensor
from jaxtyping import Int, Float
import torch


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model, num_heads, d_ff, rope_theta, context_length, device, dtype
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self, in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

        hidden_states: Float[Tensor, " batch_size sequence_length d_model"] = (
            self.token_embeddings(in_indices)
        )
        for block in self.layers:
            hidden_states: Float[Tensor, " batch_size sequence_length d_model"] = block(
                hidden_states
            )
        hidden_states: Float[Tensor, " batch_size sequence_length d_model"] = (
            self.ln_final(hidden_states)
        )
        logits: Float[Tensor, " batch_size sequence_length vocab_size"] = self.lm_head(
            hidden_states
        )
        # to obtain an unnormalized distribution over the vocabulary (the logits).
        return logits
