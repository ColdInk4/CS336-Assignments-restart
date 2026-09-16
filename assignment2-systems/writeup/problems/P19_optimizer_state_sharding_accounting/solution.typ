#import "../../template.typ": solution, deliverable

#solution(
  "optimizer_state_sharding_accounting",
  "Optimizer State Sharding Accounting",
  "5 points",
  [
    (a) Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, #raw("xl") model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. The results align with your expectations? Break down the memory usage in each setting (e.g., how much memory for parameters, how much for optimizer states, etc).

    #deliverable[2-3 sentence response with peak memory usage results and a breakdown of how the memory was divided between different model and optimizer components.]

    (b) How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration (1 node, 2 GPUs, #raw("xl") model size).

    #deliverable[2-3 sentence response with your timings.]

    (c) How does our approach to optimizer state sharding differ from ZeRO stage 1 (described as ZeRO-DP $P_"os"$ in S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He [5])?

    #deliverable[2-3 sentence summary of any differences, especially those related to memory and communication volume.]
  ],
  answer: [
  ],
)