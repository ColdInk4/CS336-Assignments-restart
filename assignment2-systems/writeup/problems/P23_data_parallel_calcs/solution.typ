#import "../../template.typ": solution, deliverable

#solution(
  "data_parallel_calcs",
  "Data parallel calculations",
  "3 points",
  [
    We now have everything we need to calculate when data parallelism becomes communication bottlenecked. Let C (in FLOP/s) denote the device accelerator speed, and W (in bytes per second) denote each device's egress bandwidth. We can then compute the computation time and communication time. Because computation and communication can be overlapped, we are bottlenecked when communication time becomes larger than computation time. We'll assume that all weights and activations are in FP16 (i.e. two bytes).

    (a) How many FLOPs are required to compute the backward pass, with $N_"DP"$ data parallelism? You can ignore all non-matmul operations. Recall that a matmul $(A, B)(B, C) arrow (A, C)$ takes $2 A B C$ flops.

    #deliverable[An answer in terms of B, D, $D_"FF"$, and $N_"DP"$, along with a one-sentence justification.]

    (b) How much communication time is required in the backward pass, with $N_"DP"$ data parallelism?

    #deliverable[An answer in terms of a subset of B, D, $D_"FF"$, $N_"DP"$, and W, along with a one-sentence justification.]

    (c) Fixing the other parameters, how large can $N_"DP"$ become before we're communication bottlenecked?

    #deliverable[An inequality with $N_"DP"$ on one side, and an expression in terms of a subset of B, D, $D_"FF"$, C, and W on the other, along with a one-sentence justification.]
  ],
  answer: [
  ],
)