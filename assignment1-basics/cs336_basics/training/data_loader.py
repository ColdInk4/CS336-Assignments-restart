import numpy.typing as npt
import torch
import numpy as np
from jaxtyping import Int
import einx
from typing import cast


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_idx: Int[npt.NDArray, " batch_size"] = np.random.randint(
        0, len(dataset) - context_length, size=batch_size
    )
    start_idx_reshaped: Int[npt.NDArray, " batch_size 1"] = cast(
        npt.NDArray, einx.id("batch_size -> batch_size 1", start_idx)
    )
    start_offsets = np.arange(context_length)
    end_offsets = start_offsets + 1
    inputs = dataset[start_idx_reshaped + start_offsets]
    targets = dataset[start_idx_reshaped + end_offsets]

    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)

    return (inputs_tensor, targets_tensor)
