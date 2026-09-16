#import "../../template.typ": solution, deliverable

#solution(
  "torch_compile",
  "Torch Compile",
  "2 points",
  [
    (a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the #raw("pytorch_attention") problem above.

    #deliverable[A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version from the #raw("pytorch_attention") problem above.]

    (b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

    #deliverable[A table comparing your vanilla and compiled Transformer model.]
  ],
  answer: [
  ],
)