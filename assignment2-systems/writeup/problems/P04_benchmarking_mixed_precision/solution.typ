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
    - the cross-entropy loss (computed inside the autocast context): FP32
    - the model's gradients: FP32

    (b) 敏感点有两层。

第一层是输入表示。低精度输入可能在进入 kernel 之前就已经丢失波动：$mu = 50, sigma = 0.01$ 时 FP16 只剩 3 档、BF16 整批舍到 50.0（std 归零），LN 输出恒 0——波动在进 kernel 前就没了，内部再用 fp32 累加也救不回来。

第二层是归约与归一化本身。累加误差随维度增长；$op("rstd") = 1 \/ sqrt(op("var") + epsilon)$ 把方差误差缩到约一半（相对形式 $delta_(op("rstd")) \/ op("rstd") approx -1/2 dot delta_(op("var")) \/ (op("var") + epsilon)$）；而当 var 被舍成 0 或负数时，$epsilon = 10^(-5)$ 是避免 inf/NaN 的唯一护栏。

区分 FP16 与 BF16 的分界行是 $mu = 50, sigma = 0.2$：FP16 此时 42 档、std 完好，BF16 只剩 7 档——50 附近的量化步长是 FP16 的 8 倍。

#h(0.6em)（注：公式本身也得选对。即使 fp32 输入、表示无问题，naive 一阶式 $E[x^2] - mu^2$ 也会因灾难性抵消被完全抵消成 $0$，而两遍式给出 $9.75 times 10^(-5)$；实测 kernel 与两遍式在输出空间 max 差 $1.29 times 10^(-5)$，与 naive 差 $7.32$。）

换 BF16 后结论不变、只会更严重：尾数从 10 位降到 7 位，同样 $mu$ 附近的量化步长是 FP16 的 8 倍；BF16 唯一的优势是指数域与 FP32 相同、不溢出，解决的是动态范围而非精度。所以统计量与归一化仍须以 FP32 计算（实测 BF16 autocast 下 layer_norm 输出即 float32）。

    (c)

  *表 1：BF16 mixed precision（ms，mean ± std）*
    #table(
  columns: 3,
  [mode], [fwd], [fwdbwd],

  [small], [35.50 ± 3.20], [84.19 ± 3.62],
  [medium], [63.29 ± 0.95], [173.28 ± 5.98],
  [large], [95.56 ± 1.64], [293.52 ± 8.38],
  [xl], [146.84 ± 0.27], [532.34 ± 4.69],
  [10B], [422.55 ± 0.90], [OOM],
)

  *表 2：FP32 full precision（ms，mean ± std）*
#table(
  columns: 3,
  [mode], [fwd], [fwdbwd],

  [small], [58.18 ± 0.45], [181.53 ± 0.45],
  [medium], [172.95 ± 0.70], [539.24 ± 0.31],
  [large], [360.57 ± 0.89], [1144.65 ± 0.54],
  [xl], [1094.66 ± 0.71], [3386.89 ± 0.88],
  [10B], [OOM], [OOM],
)

 BF16 混合精度在所有规模上都更快，且规模越大收益越大：前向加速比从 1.6×（small，35.5 vs FP32 58.2 ms）升到 7.5×（xl，146.8 vs 1094.7 ms），前向加反向从 2.2×（84.2 vs 181.5 ms）升到 6.4×（532.3 vs 3386.9 ms）
  ],
)