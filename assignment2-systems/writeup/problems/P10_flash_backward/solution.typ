#import "../../template.typ": solution, deliverable

#solution(
  "flash_backward",
  "FlashAttention-2 Backward Pass",
  "5 points",
  [
    Implement the backward pass for your FlashAttention-2 #raw("autograd.Function") using PyTorch (not Triton) and #raw("torch.compile"). Your implementation should take the #raw("Q, K, V, O, dO"), and #raw("L") tensors as inputs and return #raw("dQ, dK") and #raw("dV"). Remember to compute and use the #raw("D") vector. You may follow along the computations of Equation 13 to Equation 19.

    #deliverable[To test your implementation, run #raw("uv run pytest -k test_flash_backward").]
  ],
  answer: [
  ],
)