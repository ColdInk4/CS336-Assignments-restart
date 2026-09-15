from collections.abc import Callable
from typing import Optional
import torch
from math import sqrt


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        if not isinstance(lr, float) or lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple of length 2")
        if not (
            isinstance(betas[0], float)
            and 0 < betas[0] < 1
            and isinstance(betas[1], float)
            and 0 < betas[1] < 1
        ):
            raise ValueError(f"Invalid betas: {betas}")
        if not isinstance(eps, float) or eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if not isinstance(weight_decay, float) or weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                adjusted_lr = lr * sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data *= 1 - lr * weight_decay
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2
                p.data -= adjusted_lr * m / (v.sqrt() + eps)
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
