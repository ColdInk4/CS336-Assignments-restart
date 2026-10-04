#import "../../template.typ": solution, deliverable

#solution(
  "torch_compile",
  "Torch Compile",
  "2 points",
  [
    (a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the #raw("pytorch_attention") problem above.

    #deliverable[A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version from the #raw("pytorch_attention") problem above.]

    (b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

    #deliverable[A table comparing your vanilla and compiled Transformer model.]
  ],
  answer: [

测试环境同 P07：单张 A800 80GB，FP32（未开 TF32），batch size = 8，无 head 维度。compiled 版对每个配置先 `torch._dynamo.reset()` 再 `torch.compile`，编译耗时被 warmup 吸收。时间为 100 次的单次平均值，内存为 backward 开始前的 `torch.cuda.memory_allocated()`。

    (a)#table(
  columns: 8,
  align: center,
  table.header(
    table.cell(rowspan: 2)[seq len], table.cell(rowspan: 2)[$d$],
    table.cell(colspan: 2)[forward (ms)],
    table.cell(colspan: 2)[backward (ms)],
    table.cell(colspan: 2)[backward 前内存 (MiB)],
    [eager], [compiled], [eager], [compiled], [eager], [compiled],
  ),
  [256], [16], [0.65], [0.42], [0.94], [0.60], [23.27], [21.27],
  [256], [32], [0.67], [0.37], [0.93], [0.57], [24.27], [22.27],
  [256], [64], [0.53], [0.37], [0.86], [0.53], [26.27], [24.27],
  [256], [128], [0.67], [0.42], [0.88], [0.59], [30.27], [28.27],
  [1024], [16], [0.76], [0.53], [1.61], [1.05], [116.31], [84.31],
  [1024], [32], [0.72], [0.54], [1.63], [1.09], [120.31], [88.31],
  [1024], [64], [0.68], [0.52], [1.67], [1.08], [128.31], [96.31],
  [1024], [128], [0.76], [0.59], [1.74], [1.17], [144.31], [112.31],
  [4096], [16], [6.33], [3.73], [18.61], [9.04], [1568.50], [1056.50],
  [4096], [32], [6.68], [3.98], [19.22], [9.32], [1584.50], [1072.50],
  [4096], [64], [7.22], [4.49], [19.96], [9.94], [1616.50], [1104.50],
  [4096], [128], [8.22], [5.47], [21.12], [11.10], [1680.50], [1168.50],
  [8192], [16], [22.70], [12.40], [69.48], [31.28], [6192.75], [4144.75],
  [8192], [32], [23.98], [13.46], [71.66], [32.74], [6224.75], [4176.75],
  [8192], [64], [26.22], [15.58], [74.57], [35.17], [6288.75], [4240.75],
  [8192], [128], [30.20], [19.61], [78.89], [39.68], [6416.75], [4368.75],
  [16384], [16], [92.29], [53.52], [276.38], [142.05], [24657.25], [16465.25],
  [16384], [32], [97.72], [57.63], [282.87], [146.83], [24721.25], [16529.25],
  [16384], [64], [105.86], [66.33], [292.58], [156.01], [24849.25], [16657.25],
  [16384], [128], [121.53], [81.46], [310.46], [173.08], [25105.25], [16913.25],
  [32768], [16], table.cell(colspan: 6)[OOM（eager 与 compiled 均是）],
  [32768], [32], table.cell(colspan: 6)[OOM（eager 与 compiled 均是）],
  [32768], [64], table.cell(colspan: 6)[OOM（eager 与 compiled 均是）],
  [32768], [128], table.cell(colspan: 6)[OOM（eager 与 compiled 均是）],
)

大 seq 时 forward 约 1.5–1.7×，backward 约 2×

 因为seq = [256,1024,4096,8192,16384], [batch_size, seq, seq] 的大小分别为 [2MiB,32MiB,512MiB,2048MiB,8192MiB]

注意到每一个都正好少了一份 [batch_size, seq, seq],

(b)

测试环境：单张 A800 80GB，FP32（未开 TF32），batch size = 4，context length = 512，warmup 5 次，计时 10 次。compiled 版只编译模型（`torch.compile(model)`），optimizer step 仍为 eager。时间单位为 ms。

标准差均 < 1%，略去.
#table(
  columns: 7,
  align: center,
  table.header(
    table.cell(rowspan: 2)[size],
    table.cell(colspan: 3)[forward],
    table.cell(colspan: 3)[forward + backward + optimizer],
    [eager], [compiled], [加速比], [eager], [compiled], [加速比],
  ),
  [small], [44.93], [37.48], [1.20×], [165.44], [131.97], [1.25×],
  [xl], [883.97], [831.51 ], [1.06×], [3122.23], [2835.73 ], [1.10×],
  [10B], [3258.89], [3132.01 ], [1.04×], table.cell(colspan: 3)[OOM（两者均是）],
)
  ],
)