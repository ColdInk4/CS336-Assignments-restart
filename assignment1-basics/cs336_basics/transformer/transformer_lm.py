import torch.nn as nn
from .layers import Embedding, TransformerBlock, RMSNorm, Linear
from .functions import softmax
from torch import Tensor
from jaxtyping import Int, Float
import torch
import os
from typing import BinaryIO, IO


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
        self.context_length = context_length

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

    @torch.no_grad()
    def generate_top_p(
        self,
        prompt_ids: Int[Tensor, "L"],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        special_token_ids: list[int],
    ) -> Int[Tensor, "L"]:
        was_training = self.training
        self.eval()
        try:
            ids = prompt_ids.unsqueeze(0)
            max_len = ids.shape[-1] + max_new_tokens
            while ids.shape[-1] < max_len:
                new_token = self._top_p_sample(ids, temperature, top_p)
                ids = torch.cat([ids, new_token], dim=-1)
                if new_token.item() in special_token_ids:
                    break
        finally:
            if was_training:
                self.train()
        return ids.squeeze(0)

    @torch.no_grad()
    def _top_p_sample(
        self,
        ids: Int[Tensor, "1 L"],
        temperature: float,
        top_p: float,
    ) -> Int[Tensor, "1 1"]:
        logits: Float[Tensor, "1 L V"] = self(ids[:, -self.context_length :])  # 1, L, V
        logits_last = logits[:, -1, :]  # 1, V
        if temperature == 0:
            return logits_last.argmax(dim=-1, keepdim=True)
        probs = softmax(logits_last / temperature, dim=-1)  # 1, V

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # 1, V

        cum_probs = torch.cumsum(sorted_probs, dim=-1)  # 1, V

        #    cum_probs: a1, a1 + a2, a1 + a2 + a3
        # sorted_probs: a1,      a2,           a3
        mask = cum_probs - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)  # 1, V

        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # 1, V
        choice = torch.multinomial(sorted_probs, num_samples=1)  # 1, 1
        next_token = sorted_idx.gather(-1, choice)  # 1, 1

        return next_token

    @classmethod
    def from_checkpoint(
        cls,
        src: str | os.PathLike | BinaryIO | IO[bytes],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        map_location: torch.device | str = "cpu",
    ):
        state = torch.load(src, map_location=map_location)
        model_cfg = state["config"]["model"]
        model = cls(
            vocab_size=model_cfg["vocab_size"],
            context_length=model_cfg["context_length"],
            d_model=model_cfg["d_model"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            d_ff=model_cfg["d_ff"],
            rope_theta=model_cfg["rope_theta"],
            device=device,
            dtype=dtype,
        )
        model.load_state_dict(state["model"])
        model.eval()
        return model
