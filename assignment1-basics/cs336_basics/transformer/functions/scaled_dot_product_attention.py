from torch import Tensor
from jaxtyping import Float, Bool
import einx
from math import sqrt
import torch.cuda.nvtx as nvtx
from .softmax import softmax


@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    with nvtx.range("computing attention scores[matmul]"):
        pre_softmax_values = einx.dot(
            " ... queries [d_k], ... keys [d_k] -> ... queries keys", Q, K
        )

    with nvtx.range("computing attention scores[element divide]"):
        pre_softmax_values = pre_softmax_values / sqrt(d_k)

    with nvtx.range("mask fill"):
        if mask is not None:
            pre_softmax_values.masked_fill_(~mask, -float("inf"))

    with nvtx.range("computing softmax"):
        after_softmax_values = softmax(pre_softmax_values, -1)

    with nvtx.range("final matmul"):
        result = einx.dot(
            " ... queries [keys], ... [keys] d_v -> ... queries d_v",
            after_softmax_values,
            V,
        )
    return result
