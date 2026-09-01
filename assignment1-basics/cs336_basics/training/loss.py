from torch import Tensor
from jaxtyping import Float, Int
import einx
from typing import cast


def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]
) -> Float[Tensor, ""]:
    row_max_logits: Float[Tensor, " ... 1"] = einx.max(
        " ... [vocab_size] -> ... 1", logits
    )
    shifted_logits: Float[Tensor, " ... vocab_size"] = logits - row_max_logits
    exp_shifted_logits: Float[Tensor, " ... vocab_size"] = shifted_logits.exp()
    sum_exp_shifted_logits = einx.sum(" ... [vocab_size] -> ...", exp_shifted_logits)
    log_sum_exp_shifted_logits = sum_exp_shifted_logits.log()
    target_logits: Float[Tensor, " ..."] = einx.get_at(
        " ... [vocab_size], ... -> ...", logits, targets
    )
    per_example_max: Float[Tensor, " ..."] = cast(
        Tensor, einx.id(" ... 1 -> ...", row_max_logits)
    )
    per_position_loss = log_sum_exp_shifted_logits - target_logits + per_example_max
    return per_position_loss.mean()
