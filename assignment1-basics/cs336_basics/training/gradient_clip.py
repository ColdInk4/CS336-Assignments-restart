from collections.abc import Iterable
import torch.nn as nn


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    grads_list = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not grads_list:
        return

    grad_squared_sum = (grads_list[0] ** 2).sum()
    for grad in grads_list[1:]:
        grad_squared_sum += (grad**2).sum()

    l2_norm = grad_squared_sum.sqrt()
    if l2_norm > max_l2_norm:
        for grad in grads_list:
            grad *= max_l2_norm / (l2_norm + 1e-6)
