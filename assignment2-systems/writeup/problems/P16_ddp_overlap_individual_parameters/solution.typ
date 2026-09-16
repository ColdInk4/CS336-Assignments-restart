#import "../../template.typ": solution, deliverable

#solution(
  "ddp_overlap_individual_parameters",
  "DDP with Overlapping Individual Parameters",
  "5 points",
  [
    Implement a Python class to handle distributed data parallel training. The class should wrap an arbitrary PyTorch #raw("nn.Module") and take care of broadcasting the weights before training (so all ranks have the same initial parameters) and issuing communication calls for gradient averaging. We recommend the following public interface:

    #raw("def __init__(self, module: torch.nn.Module): Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks.\n\ndef forward(self, *inputs, **kwargs): Calls the wrapped module's forward() method with the provided positional and keyword arguments.\n\ndef finish_gradient_synchronization(self): When called, wait for asynchronous communication calls to finish on the GPU.", block: true, lang: "python")

    To use this class to perform distributed training, we'll pass it a module to wrap, and then add a call to #raw("finish_gradient_synchronization()") before we run #raw("optimizer.step()") to ensure that the optimizer step, the operation that depends on the gradients, can be safely queued.

    #raw("model = ToyModel().to(device)\nddp_model = DDP(model)\n\nfor _ in range(train_steps):\n    x, y = get_batch()\n    logits = ddp_model(x)\n    loss = loss_fn(logits, y)\n    loss.backward()\n    ddp_model.finish_gradient_synchronization()\n    optimizer.step()", block: true, lang: "python")

    #deliverable[Implement a container class to handle distributed data parallel training. This class should overlap gradient communication and the computation of the backward pass. To test your DDP class, first implement the adapters #raw("[adapters.get_ddp]") and #raw("[adapters.ddp_on_after_backward]") (the latter is optional, depending on your implementation you may not need it).

    Then, to execute the tests, run #raw("uv run pytest tests/test_ddp.py"). We recommend running the tests multiple times (e.g., 5) to ensure that it passes reliably.]
  ],
  answer: [
  ],
)