import torch
from cs336_basics.transformer.layers import TransformerBlock
from torch.utils.checkpoint import checkpoint

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 32, 2048
device = torch.device("cuda")
block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    theta=10000,
    max_seq_len=context_length,
    device=device,
)

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
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


# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / 
(1024**2):.2f} MiB")
