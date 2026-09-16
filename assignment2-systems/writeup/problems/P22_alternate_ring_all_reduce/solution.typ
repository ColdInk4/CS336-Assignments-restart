#import "../../template.typ": solution, deliverable

#solution(
  "alternate_ring_all_reduce",
  "Alternate ring all-reduce",
  "1 point",
  [
    Instead of implementing all-reduce as a ring reduce-scatter followed by a ring all-gather, let's use the following algorithm:

    For step $t = 1, ..., N-1$, device $i$ does the following:
    - If $t = 1$, initialize $y arrow x^{(i)}$, which stores the partial sum so far
    - Send $x^{(i - t + 1) mod N}$ to device $(i + 1) mod N$
    - Receive $x^{(i - t) mod N}$ from device $(i - 1) mod N$
    - Update your copy of the partial sum: $y arrow y + x^{(i - t) mod N}$

    In the same setting as above (W egress bandwidth per device, each $x^{(i)}$ is of size S), how long does this algorithm take?

    #deliverable[An answer in terms of S, N, and W, along with a one-sentence justification.]
  ],
  answer: [
  ],
)