#import "../../template.typ": solution, deliverable

#solution(
  "optimizer_state_sharding",
  "Optimizer State Sharding",
  "15 points",
  [
    Implement a Python class to handle optimizer state sharding. The class should wrap an arbitrary input PyTorch #raw("optim.Optimizer") and take care of synchronizing updated parameters after each optimizer step. We recommend the following public interface:

    #raw("def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter groups, in case the user wants to use different hyperparameters, such as learning rates, for different parts of the model); these parameters will be sharded across all the ranks. The optimizer_cls parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW). Finally, any remaining keyword arguments are forwarded to the constructor of the optimizer_cls. Make sure to call the torch.optim.Optimizer super-class constructor in this method.\n\ndef step(self, closure, **kwargs): Calls the wrapped optimizer's step() method with the provided closure and keyword arguments. After updating the parameters, synchronize with the other ranks.\n\ndef add_param_group(self, param_group: dict[str, Any]): This method should add a parameter group to the sharded optimizer. This is called during construction of the sharded optimizer by the super-class constructor and may also be called during training (e.g., for gradually unfreezing layers in a model). As a result, this method should handle assigning the model's parameters.", block: true, lang: "python")

    #deliverable[Implement a container class to handle optimizer state sharding. To test your sharded optimizer, first implement the adapter #raw("[adapters.get_sharded_optimizer]"). Then, to execute the tests, run #raw("uv run pytest tests/test_sharded_optimizer.py"). We recommend running the tests multiple times (e.g., 5) to ensure that they pass reliably.]
  ],
  answer: [
  ],
)