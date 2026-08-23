import torch
from torch import Tensor
from jaxtyping import Float


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_values = in_features.amax(dim=dim, keepdim=True)
    shifted_inputs = in_features - max_values
    exp_inputs = shifted_inputs.exp()
    sum_exp_inputs = exp_inputs.sum(dim=dim, keepdim=True)
    return exp_inputs / sum_exp_inputs
