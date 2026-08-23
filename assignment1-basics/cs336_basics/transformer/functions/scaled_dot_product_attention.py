from torch import Tensor
from jaxtyping import Float, Bool
import einx
from math import sqrt
from cs336_basics.transformer.functions import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    pre_softmax_values = einx.dot(
        " ... queries [d_k], ... keys [d_k] -> ... queries keys", Q, K
    ) / sqrt(d_k)
    if mask is not None:
        pre_softmax_values.masked_fill_(~mask, -float("inf"))
    after_softmax_values = softmax(pre_softmax_values, -1)
    result = einx.dot(
        " ... queries [keys], ... [keys] d_v -> ... queries d_v",
        after_softmax_values,
        V,
    )
    return result
