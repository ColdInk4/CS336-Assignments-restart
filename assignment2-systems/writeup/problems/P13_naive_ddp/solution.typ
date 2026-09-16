#import "../../template.typ": solution, deliverable

#solution(
  "naive_ddp",
  "Naïve DDP",
  "5 points",
  [
    #deliverable[Implement a naïve form of distributed data parallel training that all-reduces individual parameter gradients after the backward pass. To test your implementation, implement #raw("[adapters.get_ddp]") and (optionally) #raw("[adapters.ddp_on_after_backward]"), then run #raw("uv run pytest tests/test_ddp.py").]
  ],
  answer: [
  ],
)