#import "../../template.typ": solution, deliverable

#solution(
  "ddp_overlap_individual_parameters_benchmarking",
  "DDP Overlapping Individual Parameters Benchmarking",
  "1 point",
  [
    (a) Benchmark the performance of your DDP implementation when overlapping backward pass computation and communication of individual parameter gradients. Compare its performance with our previously-studied settings (the minimal DDP implementation that either issues an all-reduce for each parameter tensor, or a single all-reduce on the concatenation of all parameter tensors) with the same setup: 1 node, 2 GPUs, and the #raw("xl") model size described in Section 2.1.2.

    #deliverable[The measured time per training iteration when overlapping the backward pass with communication of individual parameter gradients, with 1-2 sentences comparing the results.]

    (b) Instrument your benchmarking code (using the 1 node, 2 GPUs, #raw("xl") model size setup) with the Nsight profiler, comparing the initial DDP implementation with this overlapped implementation. Visually compare the two traces, and provide a profiler screenshot demonstrating that one implementation overlaps compute with communication while the other doesn't.

    #deliverable[2 screenshots (one from the initial DDP implementation, and another from this DDP implementation that overlaps compute with communication) that visually show that communication is or isn't overlapped with the backward pass.]
  ],
  answer: [
  ],
)