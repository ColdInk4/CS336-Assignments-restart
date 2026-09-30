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
    在这里卡了挺久，主要是之前不了解 `CUDA HW` 和下面 `Threads → python` 的关系，导致上下两边的 NVTX 标签和时间总是对不上。

    后来弄清楚后，理解为：`Threads → python` 反映的是 CPU/Python 线程上 NVTX range 的实际持续时间；`CUDA HW` 则是这个 NVTX range 所对应的 GPU workload 在 GPU timeline 上的执行范围。因此两边的时间本来就不一定相同。

    之前一直对不上，主要是因为 CUDA 默认是异步执行的，标签内部没有 `sync`。CPU 把 CUDA 工作提交出去之后就可以继续执行，因此 CPU 侧的 `forward` range 可能已经结束了，但对应的 GPU kernels 还在继续执行，所以 `Threads → python` 中的 `forward` 会比 `CUDA HW` 中投影出的 `forward` 更短，边界也会看起来对不上。（总之还是要加 sync!）

    我选的 model sizes 是 small(context_length = 256, 512, 2048) 和 large(context_length = 256, 512, 1024)。
  
    #table(
  columns: 5,
  [size], [d_model], [d_ff], [num_layers],[num_heads],

  [small], [768], [3072], [12], [12],
  [large], [1280], [5120], [36], [20],
)

(a) 

在 forward 模式下：

   #table(
  columns: 5,
  [context_length], [256], [512], [1024], [2048],
  [size], [], [], [],[],

  [small], [38.396], [57.742],  [], [319.943], 
  [large], [182.279], [358.668], [794.869], [], 
)

在 context length = 512 时，Nsight Systems 测得 small 和 large 模型的 forward 时间分别为 57.742 ms 和 358.668 ms，而之前 Python `timeit` 的结果分别为 57.95 ms 和 360.57 ms。差不多。

(b) 
#table(
columns: 4,
[Model],
[Context length],
[Forward dominant kernel],
[Forward invocations / time],

[small],
[256],
[`ampere_sgemm_32x128_tn`],
[60 / 9.943 ms],

[small],
[512],
[`ampere_sgemm_128x32_tn`],
[24 / 17.985 ms],

[small],
[2048],
[`ampere_sgemm_128x64_tn`],
[25 / 74.756 ms],

[large],
[256],
[`ampere_sgemm_128x64_tn`],
[217 / 109.709 ms],

[large],
[512],
[`ampere_sgemm_128x64_tn`],
[253 / 281.491 ms],

[large],
[1024],
[`ampere_sgemm_128x64_tn`],
[253 / 557.831 ms],
)

列出最大时长 kernel 如下：

#table(
  columns: 3,
  [mode], [forward], [forward-and-backward],

  [small(256)], [`ampere_sgemm_32x128_tn`], [`ampere_sgemm_64x32_sliced1x4_nt`], 
  [small(512)], [`ampere_sgemm_128x32_tn`], [`ampere_sgemm_64x32_sliced1x4_nt`], 
  [small(2048)], [`ampere_sgemm_128x64_tn`], [`ampere_sgemm_128x128_nt`], 
  [large(256)], [`ampere_sgemm_128x64_tn`], [`ampere_sgemm_128x64_tn`], 
  [large(512)], [`ampere_sgemm_128x64_tn`], [`ampere_sgemm_128x64_tn`], 
  [large(1024)], [`ampere_sgemm_128x64_tn`], [`ampere_sgemm_128x64_tn`], 
)

在所有 profile 中，forward 阶段累计 GPU 时间最多的 kernel 都是 GEMM kernel，但具体实现会随模型大小和 context length 改变。small 模型中，forward 与 forward+backward 阶段的 dominant kernel 不同；large 模型中二者均由 `ampere_sgemm_128x64_tn` 主导。

(c) 除矩阵乘法外，forward pass 中还有一些 kernel 占用了较为明显的 CUDA runtime。这里将不同实现但执行相同操作的 kernel 合并统计：

#table(
  columns: 3,
  [Model], [Context], [主要的非-matmul kernel（同类合并后占比）],

  [small], [256],
  [elementwise mul (5.7%)、add (2.5%)、copy (2.1%)、div (1.9%)、reduce-sum (1.3%)、reduce-max (0.6%)],

  [small], [512],
  [elementwise mul (5.7%)、add (2.9%)、div (2.6%)、masked_fill (1.8%)、reduce-sum (1.5%)、copy (1.5%)、reduce-max (1.1%)],

  [small], [2048],
  [elementwise mul (6.1%)、div (5.9%)、add (5.3%)、masked_fill (4.6%)、exp (3.5%)、reduce-sum (1.9%)、reduce-max (1.8%)],

  [large], [256],
  [elementwise mul (3.9%)、add (1.7%)、div (1.3%)、copy (1.3%)、reduce-sum (0.8%)、reduce-max (0.4%)],

  [large], [512],
  [elementwise mul (4.0%)、add (2.1%)、div (2.0%)、masked_fill (1.3%)、copy (1.0%)、reduce-sum (0.9%)、reduce-max (0.7%)],

  [large], [1024],
  [elementwise mul (4.4%)、div (3.2%)、add (2.9%)、masked_fill (2.3%)、exp (1.8%)、reduce-sum (1.1%)、reduce-max (1.0%)],
)

(百分比为该配置下同类 kernel 合并后占 forward 总 GPU 时间的比例；表中仅列出最显著的几类，故各行列出的项目不完全一致。)

总体来看，除 GEMM 外，forward pass 中占比较明显的 kernel 包括 elementwise
mul、add、div、masked_fill、copy、exp 以及 reduce-sum / reduce-max。这些非
矩阵乘法 kernel 的合计占比随 context length 明显上升：small 模型从 ~14%
(256) 增至 ~29% (2048)，large 模型从 ~9% (256) 增至 ~17% (1024)。这反映
出长 context 下 attention 相关的 elementwise、mask 与 reduction 操作（而非
GEMM 本身）成为更重要的开销来源。


(d)

#table(
  columns: 5,
  [size], [context], [forward-only GEMM], [full train GEMM], [change],

  [small], [256], [81.8%], [67.2%], [-14.6 pp],
  [small], [512], [79.8%], [70.2%], [-9.6 pp],
  [small], [2048], [69.0%], [63.1%], [-5.9 pp],

  [large], [256], [88.2%], [73.2%], [-15.0 pp],
  [large], [512], [85.9%], [76.9%], [-9.0 pp],
  [large], [1024], [81.5%], [75.5%], [-6.0 pp],
)

完整训练步中，矩阵乘法占比相比 forward-only 下降约 6–15 个百分点：small 从约 81.8/79.8/69.0% 降到 67.2/70.2/63.1%，large 从约 88.2/85.9/81.5% 降到 73.2/76.9/75.5%。不过 GEMM 依然是占比最大的 kernel 类别（63–77%）；非 GEMM kernel 合计占比相应上升，主要来自 backward 的 `neg`、`sigmoid_backward`、`copy`、`reduction`，以及 AdamW 的 `pow`、`sqrt`、`div`、`mul`、`add`、`fill` 等 elementwise/reduction kernel。

(e)
]
)