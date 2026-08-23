import torch
from jaxtyping import Float
from torch import Tensor


def silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return in_features * torch.sigmoid(in_features)
