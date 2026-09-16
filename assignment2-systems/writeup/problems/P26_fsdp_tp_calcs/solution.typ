#import "../../template.typ": solution, deliverable

#solution(
  "fsdp_tp_calcs",
  "2D parallelism calculations",
  "6 points",
  [
    Under the same setting as the calculations so far, let's calculate when 2D parallelism becomes communication bottlenecked.

    (a) How many FLOPs are required to compute the forward pass, with $N_"FSDP"$ FSDP + $N_"TP"$ TP?

    #deliverable[An answer in terms of B, D, $D_"FF"$, $N_"FSDP"$, and $N_"TP"$, along with a one-sentence justification.]

    (b) How much communication time is required for the forward pass, with $N_"FSDP"$ FSDP + $N_"TP"$ TP? Assume that the communication along each axis can be overlapped (in other words, the collectives along the FSDP axis can be overlapped with the collectives along the TP axis).

    #deliverable[An answer in terms of a subset of B, D, $D_"FF"$, $N_"FSDP"$, $N_"TP"$, and W, along with a one-sentence justification. Hint: the answer should be expressed as a max between two quantities (the FSDP and TP collective costs), since the two can be overlapped.]

    (c) Under the optimal setting of $N_"TP"$ and $N_"FSDP"$, how large can $N = N_"TP" N_"FSDP"$ become before the forward pass is communication bottlenecked?

    #deliverable[An inequality with $N$ on one side, and an expression in terms of a subset of B, D, $D_"FF"$, C, and W on the other, along with a few sentences and equations for justification.]

    (d) Now suppose the FSDP-axis and TP-axis collectives can not be overlapped because they share the same network resources. Under the optimal setting of $N_"TP"$ and $N_"FSDP"$, how large can $N = N_"TP" N_"FSDP"$ become before the forward pass is communication bottlenecked? Don't worry about truncating $N_"TP"$ and $N_"FSDP"$ to integers.

    #deliverable[An inequality with $N$ on one side, and an expression in terms of a subset of B, D, $D_"FF"$, C, and W on the other, along with a few sentences and equations as justification.]
  ],
  answer: [
  ],
)