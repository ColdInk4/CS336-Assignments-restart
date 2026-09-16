import numpy.typing as npt
import torch
import numpy as np
from jaxtyping import Int


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[Int[torch.Tensor, "B L"], Int[torch.Tensor, "B L"]]:

    if len(dataset) <= context_length:
        raise ValueError(
            f"dataset length {len(dataset)} must be > context_length {context_length}"
        )

    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)  # (B, )
    offsets = np.arange(context_length + 1)  # (L + 1, )

    idx = starts[:, None] + offsets  # (B, L + 1)
    windows = dataset[idx]  # (B, L + 1)

    inputs = windows[:, :-1]  # (B, L)
    targets = windows[:, 1:]  # (B, L)

    # 磁盘存 uint16，进模型前转 LongTensor，Embedding.forward 期望 torch.LongTensor
    return (
        torch.as_tensor(inputs, device=device, dtype=torch.long),
        torch.as_tensor(targets, device=device, dtype=torch.long),
    )
