#import "../../template.typ": solution, deliverable

#solution(
  "fsdp",
  "Fully-Sharded Data Parallel",
  "15 points",
  [
    Implement a Python class for fully-sharded data parallel training. The class should wrap an arbitrary PyTorch #raw("nn.Module") (your full model) and hook into or wrap any #raw("Linear") or #raw("Embedding") layer within it. We recommend the following public interface:

    #raw("def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None): Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP module that will handle weight all-gathers and gradient reduce-scatters. Make sure that your hooks or your module wrappers all-gather the weights in time for the forward pass. To limit memory use, only start gathering after the layer two before the current one has completed its forward pass. In the backward pass, your hooks or module wrappers should all-gather to have the weights available for the computation. When the gradients are available, they should be reduce-scattered to the appropriate ranks. Make sure to free the gathered weights after use. When compute_dtype is provided, cast the weights to that dtype before communicating or using them for compute, while keeping master weights and optimizer updates in FP32.\n\ndef forward(self, *inputs, **kwargs): Calls the wrapped module's forward() method with the provided positional and keyword arguments.\n\ndef finish_gradient_synchronization(self): When called, wait for asynchronous communication calls to finish on the GPU.", block: true, lang: "python")

    #deliverable[Implement a container class to handle fully sharded data parallel training. Each shard of this container should be compatible with the standard AdamW implementation from assignment 1. To test your FSDP implementation, implement the adapter #raw("[adapters.get_fsdp]"). Run the tests with #raw("uv run pytest tests/test_fsdp.py"). We recommend running the tests multiple times (e.g., 5) to catch any race conditions.]
  ],
  answer: [
  ],
)