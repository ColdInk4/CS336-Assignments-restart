#import "../../template.typ": solution, deliverable

#solution(
  "nsys_profile",
  "Nsight Systems Profiling",
  "5 points",
  [
    Profile your forward pass, backward pass, and optimizer step using #raw("nsys") with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length that can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:

    (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

    #deliverable[A 1-2 sentence response.]

    (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times does this kernel get invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the "CUDA GPU Kernel Summary" under "Stats System View", and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

    #deliverable[A 1-2 sentence response.]

    (c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

    #deliverable[A 1-2 sentence response.]

    (d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

    #deliverable[A 1-2 sentence response.]

    (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

    #deliverable[A 1-2 sentence response.]
  ],
  answer: [
  ],
)