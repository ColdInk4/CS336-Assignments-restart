#import "../../template.typ": solution, deliverable

#solution(
  "flash_benchmarking",
  "FlashAttention-2 Benchmarking",
  "5 points",
  [
    (a) Write a benchmarking script using #raw("triton.testing.do_bench") that compares the performance of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with a regular PyTorch implementation (i.e., not using FlashAttention).

    Specifically, you will report a table that includes latencies for forward, backward, and the end-to-end forward-backward pass, for both Triton and PyTorch implementations. Randomly generate any necessary inputs before you start benchmarking, and run the benchmark on a single B200. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence lengths of various powers of 2 from 128 up to 65536, embedding dimension dims of various powers of 2 from 16 up to size 128, and precisions of #raw("torch.bfloat16") and #raw("torch.float32"). You will likely need to adjust tile sizes depending on the input sizes.

    #deliverable[A table of results comparing your implementation of FlashAttention-2 with the PyTorch implementation, using the settings above and reporting forward, backward, and end-to-end latencies.]
  ],
  answer: [
  ],
)