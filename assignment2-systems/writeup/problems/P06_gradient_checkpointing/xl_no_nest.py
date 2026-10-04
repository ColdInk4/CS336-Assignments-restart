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


def checkpoint_blocks(cur_layers, x):
    for layer in cur_layers:
        x = layer(x)
    return x


def checkpoint_layers(layers, x, k):
    step = len(layers) // k
    for idx in range(k):
        cur_layers = layers[step * idx : step * (idx + 1)]
        x = checkpoint(checkpoint_blocks, cur_layers, x, use_reentrant=False)
    return x


y = checkpoint_layers(layers, x, 32)
y.sum().backward()

base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = checkpoint_layers(layers, x, 32)
    y.sum().backward()

print(
    f"Total size of saved tensors in 32 TransformerBlocks with 32 checkpoints: {total_size_bytes / (1024**2):.2f} MiB"
)


print(
    f"max memory allocated in 32 TransformerBlocks with 32 checkpoints: {(torch.cuda.max_memory_allocated() - base)/ (1024**2):.2f} MiB"
)
