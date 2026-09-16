#import "../../template.typ": solution, deliverable

#solution(
  "flash_forward",
  "FlashAttention-2 Forward Pass",
  "15 points",
  [
    (a) Write a pure PyTorch (no Triton) #raw("autograd.Function") that implements the FlashAttention-2 forward pass. This will be a lot slower than the regular PyTorch implementation, but will help you debug your Triton kernel.

    Your implementation should take input #raw("Q, K, V"), as well as a flag #raw("is_causal") and produce the output #raw("O") and the logsumexp value #raw("L"). You can ignore the #raw("is_causal") flag for this task. The #raw("autograd.Function") forward should then save #raw("L, Q, K, V, O") for the backward pass and return #raw("O"). Remember that the implementation of the forward method of #raw("autograd.Function") always takes the context as its first parameter. Any #raw("autograd.Function") class needs to implement a #raw("backward") method, but for now you can make it just raise #raw("NotImplementedError"). If you need something to compare against, you can implement Equation 4 to Equation 6 and Equation 12 in PyTorch and compare your outputs.

    The interface is then #raw("def forward(ctx, Q, K, V, is_causal=False)"). Determine your own tile sizes, but make sure they are at least of size $16 times 16$. We will always test your code with dimensions that are powers of 2 and at least 16, so you don't need to worry about out-of-bounds accesses.

    #deliverable[A #raw("torch.autograd.Function") subclass that implements FlashAttention-2 in the forward pass. To test your code, implement #raw("adapters.get_flashattention_autograd_function_pytorch"). Then, run the test with #raw("uv run pytest -k test_flash_forward_pass_pytorch") and make sure your implementation passes it.]

    (b) Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then, write another subclass of #raw("torch.autograd.Function") that calls this (fused) kernel for the forward pass, instead of computing the result in PyTorch. A few technical tips:
    - To debug, we suggest comparing the results of each Triton operation you perform with the tiled PyTorch implementation you wrote in part (a).
    - Your launch grid should be set as (#raw("T_q, batch_size")), meaning each Triton program instance will load only elements from a single batch index, and only read/write to a single query tile of #raw("Q"), #raw("O"), and #raw("L").
    - The kernel should only have a single loop, which will iterate key tiles $1 lt.eq j lt.eq T_k$.
    - Advance block pointers at the end of the loop.
    - Use the function declaration below (using the block pointer we give you, you should be able to infer the setup of the rest of the pointers):

    #raw("@triton.jit\ndef flash_fwd_kernel(\n    Q_ptr, K_ptr, V_ptr,\n    O_ptr, L_ptr,\n    stride_qb, stride_qq, stride_qd,\n    stride_kb, stride_kk, stride_kd,\n    stride_vb, stride_vk, stride_vd,\n    stride_ob, stride_oq, stride_od,\n    stride_lb, stride_lq,\n    N_QUERIES, N_KEYS,\n    scale,\n    D: tl.constexpr,\n    Q_TILE_SIZE: tl.constexpr,\n    K_TILE_SIZE: tl.constexpr,\n):\n    # Program indices\n    query_tile_index = tl.program_id(0)\n    batch_index = tl.program_id(1)\n\n    # Offset each pointer with the corresponding batch index\n    # multiplied with the batch stride for each tensor\n    Q_block_ptr = tl.make_block_ptr(\n        Q_ptr + batch_index * stride_qb,\n        shape=(N_QUERIES, D),\n        strides=(stride_qq, stride_qd),\n        offsets=(query_tile_index * Q_TILE_SIZE, 0),\n        block_shape=(Q_TILE_SIZE, D),\n        order=(1, 0),\n    )\n    ...", block: true, lang: "python")

    where #raw("scale = 1/sqrt(d)") and #raw("Q_TILE_SIZE") and #raw("K_TILE_SIZE") are $B_q$ and $B_k$ respectively. You can tune these later.

    These additional guidelines may help you avoid precision issues:
    - The on chip buffers (#raw("O, l, m")) should have dtype #raw("tl.float32"). If you're accumulating into an output buffer, use the #raw("acc") argument (#raw("acc = tl.dot(..., acc=acc)")).
    - Cast $hat(P)_i^{(j)}$ to the dtype of $V^{(j)}$ before multiplying them, and cast $O_i$ to the appropriate dtype before writing it to global memory. Casting is done with #raw("tensor.to"). You can get the dtype of a tensor with #raw("tensor.dtype"), and the dtype of a block pointer/pointer with #raw("*_block_ptr.type.element_ty").

    #deliverable[A #raw("torch.autograd.Function") subclass that implements FlashAttention-2 in the forward pass using your Triton kernel. Implement #raw("adapters.get_flashattention_autograd_function_triton"). Then, run the test with #raw("uv run pytest -k test_flash_forward_pass_triton") and make sure your implementation passes it.]

    (c) Add a flag as the last argument to your #raw("autograd.Function") implementation for causal masking. This should be a boolean flag that, when set to #raw("True"), enables an index comparison for causal masking. Your Triton kernel should have a corresponding additional parameter #raw("is_causal: tl.constexpr") (this is a required type annotation). In Triton, construct appropriate index vectors for queries and keys, and compare them to form a square mask of size $B_q times B_k$. For elements that are masked out, add the constant value of #raw("-1e6") to the corresponding elements of the attention score matrix $hat(S)_i^{(j)}$. Make sure to save the mask flag for backward using #raw("ctx.is_causal = is_causal").

    #deliverable[An additional flag for your #raw("torch.autograd.Function") subclass that implements the FlashAttention-2 forward pass with causal masking using your Triton kernel. Make sure that the flag is optional and defaults to #raw("False") so the previous tests still pass.]
  ],
  answer: [
  ],
)