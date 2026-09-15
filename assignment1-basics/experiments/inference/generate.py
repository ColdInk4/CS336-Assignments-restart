from cs336_basics.transformer import TransformerLM
from cs336_basics.transformer.functions import softmax
from jaxtyping import Int
import torch
from torch import Tensor


@torch.no_grad()
def generate(
    model,
    prompt_ids: Int[Tensor, "1 L"],
    max_new_tokens,
    temperature,
    top_p,
    special_token_ids: list,
):
    model.eval()
    max_len = prompt_ids.shape[-1] + max_new_tokens
    while prompt_ids.shape[-1] < max_len:
        new_token = top_p_sample(model, prompt_ids, temperature, top_p)
        prompt_ids = torch.cat([prompt_ids, new_token], dim=-1)
        if new_token.item() in special_token_ids:
            break
    return prompt_ids


def top_p_sample(
    model,
    prompt_ids: Int[Tensor, "1 L"],
    temperature: float,
    top_p: float,
):
    logits = model(prompt_ids)  # 1, L, V
    logits_last = logits[:, -1, :]  # 1, V
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
