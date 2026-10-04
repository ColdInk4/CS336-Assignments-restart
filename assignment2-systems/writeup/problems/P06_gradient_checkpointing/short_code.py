import torch
from cs336_basics.transformer.layers import TransformerBlock
from torch.utils.checkpoint import checkpoint
import torch.nn as nn

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 32, 2048
device = torch.device("cuda")


# Fuse as much torch.compile will allow
def make_block():
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        theta=10000,
        max_seq_len=context_length,
        device=device,
    )
    block.compile(fullgraph=True)

    return block


layers = nn.ModuleList([make_block() for _ in range(32)])

x = torch.randn((4, context_length, d_model), requires_grad=True, device=device)

# Now logs the number of bytes saved
total_size_bytes = 0


def pack_hook(t):
    if isinstance(
        t, torch.nn.Parameter
    ):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def recursive_binary_checkpoint(layers, x):
    n = len(layers)

    if n <= 1:
        for block in layers:
            x = block(x)
        return x
    mid = n // 2

    def left(x):
        return recursive_binary_checkpoint(layers[:mid], x)

    def right(x):
        return recursive_binary_checkpoint(layers[mid:], x)

    leftout = checkpoint(left, x, use_reentrant=False)
    return right(leftout)


# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = recursive_binary_checkpoint(layers, x)


print(
    f"Total size of saved tensors in 32 TransformerBlocks with recursive_binary_checkpoint: {total_size_bytes / (1024**2):.2f} MiB"
)
