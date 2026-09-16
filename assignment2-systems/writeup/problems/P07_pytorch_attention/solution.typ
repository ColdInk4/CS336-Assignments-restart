#import "../../template.typ": solution, deliverable

#solution(
  "pytorch_attention",
  "PyTorch Attention Benchmarking",
  "2 points",
  [
    (a) Benchmark your attention implementation at different scales. Write a script that will:
    - (i) Fix the batch size to 8 and don't use multihead attention (i.e. remove the head dimension).
    - (ii) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension #raw("d_model"), and [256, 1024, 4096, 8192, 16384] for the sequence length.
    - (iii) Create random inputs #raw("Q, K, V") for the appropriate size.
    - (iv) Time 100 forward passes through attention using the inputs.
    - (v) Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
    - (vi) Make sure to warm up, and to call #raw("torch.cuda.synchronize()") after each forward/backward pass.

    Depending on your GPU, some of these configurations are expected to run out of memory. Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers in Section 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost?

    #deliverable[A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response.]
  ],
  answer: [
  ],
)