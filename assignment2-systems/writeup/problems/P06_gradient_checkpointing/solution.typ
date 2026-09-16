#import "../../template.typ": solution, deliverable

#solution(
  "gradient_checkpointing",
  "Memory-Optimal Gradient Checkpointing",
  "4 points",
  [
    Consider a Transformer with #raw("N") identical blocks stacked sequentially. Without any checkpointing, all #raw("N") blocks' worth of residuals are kept alive simultaneously, giving #raw("O(N)") peak activation memory. We have a free hand to wrap any subset of the forward pass in #raw("checkpoint"), including nesting #raw("checkpoint") calls inside one another.

    (a) What checkpointing strategy minimizes peak activation memory, ignoring the compute cost? Describe how you would arrange the #raw("checkpoint") calls (a code sketch is fine), and give the asymptotic peak activation memory and compute cost of your strategy as a function of #raw("N"). Assume the residuals saved by a single block dominate any per-checkpoint bookkeeping.

    #deliverable[A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.]

    (b) Consider the #raw("xl") model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest #raw("checkpoint") calls), what is the best checkpointing strategy to reduce peak memory? Profile your run's peak memory to validate your hypothesis. Compare the peak memory of the next smaller and larger checkpointing block sizes to be sure.

    #deliverable[A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.]
  ],
  answer: [
  ],
)