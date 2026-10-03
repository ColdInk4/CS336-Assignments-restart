#import "../../template.typ": solution, deliverable

#solution(
  "memory_profiling",
  "Memory Profiling",
  "4 points",
  [
    Profile your complete training step of forward pass, backward pass, and optimizer step of the #raw("xl") model from Table 1 with context lengths of 128 and 2048.

    (a) Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the #raw("xl") model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?

    #deliverable[Two images of the "Active memory timeline" of an #raw("xl") model, from the #raw("memory_viz") tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.]

    (b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

    #deliverable[A table with two numbers per context length.]

    (c) Find the peak memory usage of the #raw("xl") model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?

    #deliverable[A 2-3 sentence response.]

    (d) Consider the #raw("xl") model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by $1024^2$).

    #deliverable[A 1-2 sentence response with your derivation.]

    (e) Now look closely at the "Active Memory Timeline" from #raw("pytorch.org/memory_viz") of a memory snapshot of the #raw("xl") model doing a forward pass. When you reduce the "Detail" level, the tool hides the smallest allocations to the corresponding level (e.g., putting "Detail" at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

    #deliverable[A 1-2 sentence response.]

    (f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model's lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single #raw("TransformerBlock") in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute.

    During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every #raw("TransformerBlock") in the backward pass, calculate how much memory the produced gradient tensors for a #raw("TransformerBlock") take. Does the result match what you expect?

    #deliverable[Screenshots from Nsight Systems and a 1-2 paragraph response.]
  ],
  answer: [
    由于 xl + ctx(2048) 在本机OOM，所以决定采用 large+ctx (128/1024) 来作答。
  
  
  (a) 
  ],
)