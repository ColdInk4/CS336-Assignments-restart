from torch import Tensor
from jaxtyping import Float, Int
import einx
from typing import cast


def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]
) -> Float[Tensor, ""]:
    row_max_logits: Float[Tensor, " ... 1"] = logits.amax(dim=-1, keepdim=True)
    shifted_logits: Float[Tensor, " ... vocab_size"] = logits - row_max_logits
    exp_shifted_logits = shifted_logits.exp()
    target_logits: Float[Tensor, " ..."] = einx.get_at(
        " ... [vocab_size], ... -> ...", logits, targets
    )
    per_example_max: Float[Tensor, " ..."] = cast(
        Tensor, einx.id(" ... 1 -> ...", row_max_logits)
    )
    per_example_loss = (
        exp_shifted_logits.sum(dim=-1).log() - target_logits + per_example_max
    )
    return per_example_loss.mean()
