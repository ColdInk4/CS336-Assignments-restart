#import "../../template.typ": solution, deliverable

#solution(
  "fsdp_accounting",
  "FSDP Accounting",
  "5 points",
  [
    (a) Given your analysis in Section 6, how much memory do you expect to save from the peak by implementing FSDP? You can ignore the size of the preallocated buffers needed to all-gather weights to each GPU in your calculation.

    #deliverable[2-3 sentence response with your findings.]

    (b) Profile the #raw("xl") model on two GPUs and pay attention to the all-gather of weights. Does the communication finish in time for the forward pass?

    #deliverable[2-3 sentence response with your timings. Include screenshots of Nsight to back up your claims.]
  ],
  answer: [
  ],
)