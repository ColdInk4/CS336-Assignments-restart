#import "../../template.typ": solution, deliverable

#solution(
  "naive_ddp_benchmarking",
  "Naïve DDP Benchmarking",
  "3 points",
  [
    In this naïve DDP implementation, parameter gradients are individually all-reduced across ranks after each backward pass. To better understand the overhead of data parallel training, create a script to benchmark your previously-implemented language model when trained with this naïve implementation of DDP. Measure the total time per training step and the proportion of time spent on communicating gradients. Collect measurements in the single-node setting (1 node x 2 GPUs) for the #raw("xl") model size described in Section 2.1.2.

    #deliverable[A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients for each setting.]
  ],
  answer: [
  ],
)