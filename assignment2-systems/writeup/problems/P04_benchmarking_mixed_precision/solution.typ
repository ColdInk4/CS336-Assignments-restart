#import "../../template.typ": solution, deliverable

#solution(
  "benchmarking_mixed_precision",
  "Benchmarking Mixed Precision",
  "2 points",
  [
    (a) Consider the following model:

    #raw("class ToyModel(nn.Module):\n    def __init__(self, in_features: int, out_features: int):\n        super().__init__()\n        self.fc1 = nn.Linear(in_features, 10, bias=False)\n        self.ln = nn.LayerNorm(10)\n        self.fc2 = nn.Linear(10, out_features, bias=False)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        x = self.relu(self.fc1(x))\n        x = self.ln(x)\n        x = self.fc2(x)\n        return x", block: true, lang: "python")

    Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We'd like to use autocasting mixed precision with FP16. What are the data types of:
    - the model parameters within the autocast context?
    - the output of the first feed-forward layer (#raw("ToyModel.fc1"))?
    - the output of layer norm (#raw("ToyModel.ln"))?
    - the model's predicted logits?
    - the loss?
    - the model's gradients?

    #deliverable[The data types for each of the components listed above.]

    (b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

    #deliverable[A 2-3 sentence response.]

    (c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in Section 2.1.2. Compare the results of using full precision versus mixed precision, and comment on how the model size changes. You may find the #raw("nullcontext") no-op context manager to be useful.

    #deliverable[A 2-3 sentence response with your timings and commentary.]
  ],
  answer: [
    (a) 
    - the model parameters within the autocast context: FP32
    - the output of the first feed-forward layer (ToyModel.fc1): FP16
    - the output of layer norm (ToyModel.ln): FP32
    - the model's predicted logits: FP16
    - the loss: FP32
    - the model's gradients: FP32
    
    (b) 最敏感的部分是均值与方差的计算（沿特征维度的 reduction）：方差需要累加 $x^2$，FP16 只有 5 个指数位（最大约 65504），很容易溢出成 inf 进而在反向传播中产生 NaN；同时 var = E[$x^2$] − (E[$x$])² 的相减在 10 位尾数下会发生灾难性抵消，且 eps = 1e-5 低于 FP16 的最小正规数而被直接吞掉（除法和开方本身反而不是瓶颈）。

    BF16 的指数位与 FP32 相同（8 位），动态范围的问题彻底消失，$x^2$ 不会溢出、eps 也能表示；但 BF16 尾数只有 7 位，mean/var 的量化误差和抵消问题依然存在。

    因此改用 BF16 后仍然应当区别对待。

    (c)

    在BF16下：
    #table(
  columns: 3,
  [mode], [fwd], [fwdbwd],
  [size], [], [],

  [small], [35.50 ± 3.20], [84.19 ± 3.62],
  [medium], [63.29 ± 0.95], [173.28 ± 5.98],
  [large], [95.56 ± 1.64], [293.52 ± 8.38],
  [xl], [146.84 ± 0.27], [532.34 ± 4.69],
  [10B], [422.55 ± 0.90], [OOM],
)

  在FP32下：
#table(
  columns: 3,
  [mode], [fwd], [fwdbwd],
  [size], [], [],

  [small], [58.18 ± 0.45], [181.53 ± 0.45],
  [medium], [172.95 ± 0.70], [539.24 ± 0.31],
  [large], [360.57 ± 0.89], [1144.65 ± 0.54],
  [xl], [1094.66 ± 0.71], [3386.89 ± 0.88],
  [10B], [OOM], [OOM],
)

在所有模型规模上，BF16 混合精度都更快，且规模越大加速越明显：前向加速比从 1.6#sym.times（small，35.5 对 58.2 ms）上升到 7.5#sym.times（XL，146.8 对 1094.7 ms），前向加反向则从 2.2#sym.times（84.2 对 181.5 ms）上升到 6.4#sym.times（532.3 对 3386.9 ms）。

大模型收益更高，是因为其运行时间主要消耗在矩阵乘（GEMM）上，而 GEMM 的吞吐随低精度张量核心的使用而提升；小模型则受 kernel 启动和逐元素算子开销限制，这部分混合精度无法压缩。

此外，BF16 将激活内存占用减半，使 10B 模型在前向能放进显存（422.6 ms）而 FP32 直接 OOM；但前向加反向在 10B 下仍然 OOM，因为梯度和优化器状态仍保持 FP32。
  ],
)