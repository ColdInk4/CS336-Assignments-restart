import numpy.typing as npt
import torch
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    if len(dataset) <= context_length:
        raise ValueError(
            f"dataset length {len(dataset)} must be > context_length {context_length}"
        )

    max_start = len(dataset) - context_length
    starts = np.random.randint(0, len(dataset) - max_start, size=batch_size)  # (B, )
    offsets = np.arange(context_length + 1)  # (L + 1, )

    idx = starts[:, None] + offsets  # (B, L + 1)
    windows = dataset[idx]  # (B, L + 1)

    inputs = windows[:, :-1]  # (B, L)
    targets = windows[:, 1:]  # (B, L)

    return (
        torch.as_tensor(inputs, device=device),
        torch.as_tensor(targets, device=device),
    )
