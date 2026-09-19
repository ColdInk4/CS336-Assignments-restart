#import "../../template.typ": solution, deliverable

#solution(
  "benchmarking_script",
  "Benchmarking Script",
  "4 points",
  [
    (a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following:
    - Given hyperparameters (e.g., number of layers), initialize a model.
    - Generate a random batch of data.
    - Run #raw("w") warm-up steps (before you start measuring time), then time the execution of #raw("n") steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python #raw("timeit") module (e.g., either using the #raw("timeit") function, or using #raw("timeit.default_timer()"), which gives you the system's highest resolution clock, thus a better default for benchmarking than #raw("time.time()")).
    - Call #raw("torch.cuda.synchronize()") after each step.

    #deliverable[A script that will initialize a `basics` Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-and-backward, and full training steps that include the optimizer step.]

    (b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

    #deliverable[A 1-2 sentence response with your timings.]

    (c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

    #deliverable[A 2-3 sentence response.]
  ],
  answer: [
    (a) 见`scripts/benchmark.py`

    (b) 以下单位如无特别注明，均为毫秒
  
  #table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],

  [small], [45.50 ± 0.16], [149.17 ± 4.83], [159.12 ± 0.81],
  [medium], [138.97 ± 1.81], [444.64 ± 0.48], [479.85 ± 0.64],
  [large], [288.27 ± 1.25], [951.34 ± 1.61], [1031.19 ± 1.02],
  [xl], [891.64 ± 0.78], [2819.82 ± 0.62], [3080.10 ± 1.56],
  [10B], [OOM], [OOM], [OOM],
)

以 medium 为例，forward 约 139 ms，backward（fwdbwd − fwd）约 306 ms，optimizer step（fwdbwdopt − fwdbwd）约 35 ms；$"fwdbwd" approx 3.2 times "fwd"$，符合反向需要额外两次矩阵乘的理论预期。跨 10 次测量的标准差都在均值的 1% 以内，说明稳态下计时非常稳定。

(c) 
当 warmup 为 0 时：

#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],

  [small], [128.20 ± 248.50], [181.87 ± 97.83], [176.64 ± 43.76],
  [medium], [192.26 ± 162.39], [453.18 ± 43.00], [504.50 ± 74.00],
  [large], [362.23 ± 223.86], [946.60 ± 24.41], [1039.24 ± 15.93],
  [xl], [941.67 ± 171.08], [2787.55 ± 8.57], [3093.56 ± 21.13],
  [10B], [OOM], [OOM], [OOM],
)

不进行 warmup 时，第一次迭代可能包含 CUDA 运行时的惰性初始化、显存分配等一次性开销，因此会显著拉高平均时间，并导致标准差增大。例如 small 的 forward 从 45.50 ms 增加到 128.20 ms，标准差也从 0.16 ms 增加到 248.50 ms。加入 warmup 后，这些一次性开销被排除，测量结果因此稳定得多。

当 warmup 为1时：
#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],
  [small], [45.94 ± 0.43], [148.37 ± 0.29], [161.53 ± 3.19],
  [medium], [140.09 ± 0.87], [444.72 ± 0.34], [480.15 ± 0.76],
  [large], [291.75 ± 1.22], [948.69 ± 0.88], [1033.31 ± 1.18],
  [xl], [895.31 ± 0.58], [2803.91 ± 3.13], [3091.76 ± 1.26],
  [10B], [OOM], [OOM], [OOM],
)

当 warmup 为 2 时：

#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],

  [small], [45.66 ± 0.38], [149.05 ± 2.45], [160.58 ± 0.27],
  [medium], [139.26 ± 2.87], [444.36 ± 0.61], [480.47 ± 0.64],
  [large], [288.50 ± 0.81], [949.39 ± 1.79], [1032.54 ± 0.89],
  [xl], [889.51 ± 1.26], [2817.13 ± 2.00], [3078.77 ± 2.31],
  [10B], [OOM], [OOM], [OOM],
)

当 warmup 为 1 或 2 时，结果已经与 warmup=5 非常接近，说明 1--2 次 warmup 已经基本消除了首次运行开销。剩余的细微差异可能来自少量初始化/分配开销以及 GPU 测量本身的波动。

  ],
)