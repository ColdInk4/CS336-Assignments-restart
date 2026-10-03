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
  在不使用混合精度的情况下：

  large + ctx(1024) 的 fwd 阶段如下：
  #image("/assets/image-1.png")
  
  fwd 因为我使用了`no_grad()`，所以都是平坦的。一共有36个突起，正好对应large 的 `num_layers`。


  large + ctx(1024) 的 fwdbwdopt 阶段如下：
  #image("/assets/image.png")

  正向传播这里会存储各个为算梯度而保存下来的张量，所以会呈现爬坡。下降是反向传播，右侧平的阶段是正在优化

  (b) 在不使用混合精度的情况下：
  #table(
  columns: 3,
  [model/ctx], [fwd], [fwdbwdopt],

  [large(128)], [3.76 GiB], [15.1 GiB],
  [large(1024)], [5.13 GiB], [67.0 GiB],
)

  (c)在使用混合精度的情况下：
    #table(
  columns: 3,
  [model/ctx], [fwd], [fwdbwdopt],

  [large(128)], [5.52 GiB], [15.1 GiB],
  [large(1024)], [6.51 GiB], [49.8 GiB],
)

注意到对于较大ctx的fwdbwdopt，内存会有较大的降低。而对于fwd阶段，由于autocast有缓存机制，来避免在同一个 autocast 上下文中重复进行类型转换，所以会有所增加。

(d)

  #table(
  columns: 7,
  [size], [d_model], [d_ff], [num_layers],[num_heads], [batch], [vocab_size],

  [large], [1280], [5120], [36], [20], [4],[10000]
)

  residual中张量的大小为`batch sequence_length d_model`，在 large 下即为：$4 times "ctx" times 1280$

  - large(ctx = 128): $4 times 128 times 1280 times 4B = 2.5"MiB"$
  - large(ctx = 1024): $4 times 1024 times 1280 times 4B = 20"MiB"$
  
(e) 
在不混合精度的large(ctx = 128)的forward里（参数这类在开始记录前就已存在的 ghost 块不算），最大的size为19.5MiB，按 fp32 每个元素 4 字节算，一共5,120,000个元素。 调用栈里的 Python 帧是 transformer_lm.py:50 → linear.py:31，也就是所有层都跑完之后的那一次线性层，也就是logits 张量。

其张量大小为：[batch, context_length, vocab_size]，即 $4 times 128 times 10000 = 5,120,000$，对应上述元素数。

在不混合精度的large(ctx = 1024)的forward里，最大的size为320MiB，按 fp32 每个元素 4 字节算，一共83,886,080个元素。

其320MiB的各个操作为
#table(
  columns: 3,
  [编号], [op], [代码行],
  [557], [bmm（einsum 内部）], [scaled_dot_product_attention.py:18],
  [558], [div], [scaled_dot_product_attention.py:23],
  [561], [sub], [softmax.py:7],
  [562], [exp], [softmax.py:8],
  [564], [div], [softmax.py:10],
)
而`scaled_dot_product_attention`第一步得到的张量大小为：[batch, num_heads, context_length, context_length]，即 $4 times 20 times 1,024 times 1,024 = 83,886,080$，对应上述元素数。

ctx 较小时，最大的块是 logits，因为它的形状里有 vocab_size 这个很大的维度；ctx 变大后 attention 反超，因为它的形状里 ctx 出现了 2 次，大小随 ctx 按 平方 增长。

(f)
我选取的模型是large(ctx=1024)

#image("/assets/image-2.png")
在forward阶段，我选取一个block，它从TransformerBlock开始的17.44GiB 到结束的18.97GiB，净变化1.53GiB。

#image("/assets/image-3.png")
在backward阶段，我选取两个SigmoidBackward0来读取，从第一个开始的55.89GiB 到下一个开始的54.46GiB，净变化-1.43GiB。

梯度约为：-1.43+1.53 = 0.1GiB

一个TransformerBlock里面有
  - 2 个 RMSNorm
    - $2 times "d_model"$ 
  - MultiheadSelfAttention
    - $4 times "d_model" times "d_model"$
  - SwiGLU
    - $3 times "d_model" times "d_ff"$

总参数为：$2 times "d_model" + 3 times "d_model" times "d_ff" + 4 times "d_model" times "d_model" = 2 times 1280 + 3 times 1280 times 5120 + 4 times 1280 times 1280 = 26,216,960$

总字节数为总参数 $times 4$，约为0.0977 GiB，因为读图有误差，所以大致对的上这个梯度值。
  
找到的前五个（含并列）的最大几块如下（用 memory_viz 训练 snapshot 的调用栈认出，同样是 
large、ctx1024、fp32）：
                                                                                       
#table(
  columns: 6,
  [编号], [代码行], [op], [张量含义], [大小], [占比],
  [1602], [scaled_dot_product_attention.py:23], [div], [pre softmax values], [320 MiB], [20.42%],
  [1606], [softmax.py:8], [exp], [exp(x − max)], [320 MiB], [20.42%],
  [1608], [softmax.py:10], [div], [after softmax values], [320 MiB], [20.42%],
  [1621], [linear.py:31（swiglu.py:23）], [bmm], [swiglu里的`self.w1(x)`], [80 MiB], [5.11%],
  [1622], [silu.py:7], [sigmoid], [silu里面的sigmoid计算], [80 MiB], [5.11%],
  [1623], [silu.py:7], [mul], [silu里面的mul计算], [80 MiB], [5.11%],
  [1624], [linear.py:31（swiglu.py:23）], [bmm], [swiglu里的`self.w3(x)`], [80 MiB], [5.11%],
  [1625], [swiglu.py:23], [mul], [swiglu里的 `silu(self.w1(x)) * self.w3(x)`], [80 MiB], [5.11%],
)

第 4、5 名有 5 块 80 MiB 并列，所以全部列出。占比 = 块大小 ÷ 单个 block 的 residuals（1.53 GiB ≈ 1567 MiB）。注意力里的 3 块合计占 61.26%，FFN 里的 5 块合计占 25.53 %，8 块一共占 86.79%。剩下的 13.21% 主要是 残差流（[batch, ctx, d_model]，20 MiB） 大小的张量。

  ],
)