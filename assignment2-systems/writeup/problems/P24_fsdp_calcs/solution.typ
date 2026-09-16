#import "../../template.typ": solution, deliverable

#solution(
  "fsdp_calcs",
  "Fully sharded data parallel calculations",
  "3 points",
  [
    Under the same setting as the data parallel calculation, let's calculate when FSDP becomes communication bottlenecked.

    (a) How many FLOPs are required to compute the backward pass, with $N_"FSDP"$ FSDP? What about the forward pass?

    #deliverable[Two answers in terms of B, D, $D_"FF"$, and $N_"FSDP"$, along with two one-sentence justifications.]

    (b) How much communication time is required in the backward pass, with $N_"FSDP"$ FSDP? What about the forward pass?

    #deliverable[Two answers in terms of a subset of B, D, $D_"FF"$, $N_"FSDP"$, and W, along with two one-sentence justifications.]

    (c) Fixing the other parameters, how large can $N_"FSDP"$ become before the backward pass is communication bottlenecked? What about the forward pass?

    #deliverable[Two inequalities with $N_"FSDP"$ on one side, and an expression in terms of a subset of B, D, $D_"FF"$, C, and W on the other, along with two one-sentence justifications.]
  ],
  answer: [
  ],
)