#import "../../template.typ": solution, deliverable

#solution(
  "minimal_ddp_flat_benchmarking",
  "Minimal DDP with Flat Gradients Benchmarking",
  "2 points",
  [
    Modify your minimal DDP implementation to communicate a tensor with flattened gradients from all parameters. Compare its performance with the minimal DDP implementation that issues an all-reduce for each parameter tensor under the previously-used conditions (1 node x 2 GPUs, #raw("xl") model size as described in Section 2.1.2).

    #deliverable[The measured time per training iteration and time spent communicating gradients under distributed data parallel training with a single batched all-reduce call. 1-2 sentences comparing the results when batching vs. individually communicating gradients.]
  ],
  answer: [
  ],
)