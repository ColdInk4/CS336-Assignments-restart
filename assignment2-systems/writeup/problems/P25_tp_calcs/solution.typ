#import "../../template.typ": solution, deliverable

#solution(
  "tp_calcs",
  "Tensor parallel calculations",
  "4 points",
  [
    Under the same setting as the DP and FSDP calculations, let's calculate when TP becomes communication bottlenecked.

    (a) Given input $d y$ of size $(B, D)$, write out the backward pass of the tensor parallel strategy described above (where $W_1^{(i)}$ and $W_2^{(i)}$ have shape $(D, D_"FF"/N_"TP")$, and $W_3^{(i)}$ has shape $(D_"FF"/N_"TP", D)$).

    #deliverable[A series of equations describing the backward pass, in terms of $d y$, sharded weights $(W_1^{(i)}, W_2^{(i)}, W_3^{(i)})$, activations saved from the forward pass $(x, x_1^{(i)}, x_2^{(i)}, z^{(i)})$, communication primitives, and any intermediate variables you'd like to define. The equations should produce each device's gradients $d W_1^{(i)}, d W_2^{(i)}, d W_3^{(i)}$ and the backward pass output $d x$. Feel free to reference the non-sharded backward pass in Section 8.2 and modify it.]

    (b) How many FLOPs are required to compute the forward pass, with $N_"TP"$ TP? What about the backward pass?

    #deliverable[Two answers in terms of B, D, $D_"FF"$, and $N_"TP"$, along with two one-sentence justifications.]

    (c) How much communication time is required in the forward pass, with $N_"TP"$ TP? What about the backward pass?

    #deliverable[Two answers in terms of a subset of B, D, $D_"FF"$, $N_"TP"$, and W, along with two one-sentence justifications.]

    (d) Fixing the other parameters, how large can $N_"TP"$ become before the backward pass is communication bottlenecked? What about the forward pass?

    #deliverable[Two inequalities with $N_"TP"$ on one side, and an expression in terms of a subset of B, D, $D_"FF"$, C, and W on the other, along with two one-sentence justifications.]
  ],
  answer: [
  ],
)