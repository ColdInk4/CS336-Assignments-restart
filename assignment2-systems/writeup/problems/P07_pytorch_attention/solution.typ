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

  测试环境：单张 A800 80GiB，FP32，batch size = 8，无 head 维度。时间为 100 次的单次平均值，内存为 backward 开始前的 `torch.cuda.memory_allocated()`。
  
  题目要求的 20 个配置：全部成功，没有 OOM,额外测了 seq=32768：4 个 d 全部 OOM,所以最小的 OOM 配置是 (32768, 16)

  #table(
  columns: 5,
  align: center,
  table.header(
    [seq len], [$d$], [forward (ms)], [backward (ms)], [backward 前内存 (MiB)],
  ),
  [256], [16], [0.65], [0.94], [23.27],
  [256], [32], [0.67], [0.93], [24.27],
  [256], [64], [0.53], [0.86], [26.27],
  [256], [128], [0.67], [0.88], [30.27],
  [1024], [16], [0.76], [1.61], [116.31],
  [1024], [32], [0.72], [1.63], [120.31],
  [1024], [64], [0.68], [1.67], [128.31],
  [1024], [128], [0.76], [1.74], [144.31],
  [4096], [16], [6.33], [18.61], [1568.50],
  [4096], [32], [6.68], [19.22], [1584.50],
  [4096], [64], [7.22], [19.96], [1616.50],
  [4096], [128], [8.22], [21.12], [1680.50],
  [8192], [16], [22.70], [69.48], [6192.75],
  [8192], [32], [23.98], [71.66], [6224.75],
  [8192], [64], [26.22], [74.57], [6288.75],
  [8192], [128], [30.20], [78.89], [6416.75],
  [16384], [16], [92.29], [276.38], [24657.25],
  [16384], [32], [97.72], [282.87], [24721.25],
  [16384], [64], [105.86], [292.58], [24849.25],
  [16384], [128], [121.53], [310.46], [25105.25],
  [32768], [16], table.cell(colspan: 3)[OOM],
  [32768], [32], table.cell(colspan: 3)[OOM],
  [32768], [64], table.cell(colspan: 3)[OOM],
  [32768], [128], table.cell(colspan: 3)[OOM],
)

考虑反向传播要保留哪些量:

对于输入量 $["batch", "context_length","d_model"]$:
- QK: 矩阵相乘运算要保留两个输入
  -  Q: $["batch", "context_length","d_model"]$
  -  K: $["batch", "context_length","d_model"]$
- softmax:
  - max: 要保留输入和输出
    - 输入: $["batch", "context_length","context_length"]$
    - 输出: $["batch", "context_length","1"]$
  - sub: 导数为 1,无需保留
  - exp: 导数为自己,保留输出
    - 输出: $["batch", "context_length","context_length"]$
  - sum: 对每个 $x_i$ 的导数都是 1,无需保留
  - div: 要保留两个输入
    - 输入(与exp的输出共享): $["batch", "context_length","context_length"]$ 
    - 输入: $["batch", "context_length",1]$ 
- AV: 矩阵相乘运算要保留两个输入
  - A: $["batch", "context_length","context_length"]$ 
  - V: $"batch" times "context_length" times "d_model"$

总元素量为 $ &3 times "batch" times "context_length"times "context_length" \ + & 3 times "batch" times "context_length" times "d_model" \ + & 2 times "batch" times "context_length" \ = & "batch" times "context_length" times(3 times "context_length" + 3  times "d_model" + 2) $

当`batch_size` = 8 时,总元素量为 $ 8 times "context_length" times(3 times "context_length" + 3  times "d_model" + 2) $

即字节数为$ 4 times 8 times "context_length" times(3 times "context_length" + 3  times "d_model" + 2) $, 随seq平方级增长

当ctx_len = 16384, d_model = 128时,
字节数为$ 4 times 8  times 16384 times(3 times 16384 + 3  times 128 + 2)  = 24769"MiB"$

实测25105.25MiB,差值为336.25MiB.还剩下Q.grad、K.grad、V.grad、gradient、y, 这些合计 $ & 4 times 5 times "batch" times "context_length" times "d_model" \ =& 4 times 5 times 8 times 16384 times 128 \ =& 320"MiB" $

剩下约 16.25MiB 为 cuBLAS workspace(用同样的方法检查全部 20 个配置，残差都是 16.25 MiB),它们都不是 saved tensors.

接下来看OOM的最小参数:

当ctx_len = 32768, d_model = 16时,需要的内存即为:

$ &4 times 8 times "context_length" times(3 times "context_length" + 3  times "d_model" + 2) \ +& 4 times 5 times 8 times "context_length" times "d_model"\ +&16.25("MiB")\ =&4 times 8 times 32768 times(3 times 32768 + 3  times 16 + 2) \ +& 4 times 5 times 8 times 32768 times 16\ +&16.25("MiB")\ =& 98450.25"MiB" $ 

光是一份 $"seq"^2$ 张量,就已经有32GiB了,三份已经超出我们的显存 80 GiB, OOM在 forward 阶段就已经发生了.

想办法消除 $"seq"^2$? 单独softmax里面就占了所有 `context_length` 平方的内容.可以考虑不存这些张量,重新分块来算.我们可以保存max和sum这些相较小的张量
  ],
)