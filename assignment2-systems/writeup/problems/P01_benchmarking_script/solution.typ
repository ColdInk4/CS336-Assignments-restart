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

  [small], [57.95 ± 0.31], [180.79 ± 0.31], [194.43 ± 1.64],
  [medium], [172.72 ± 0.87], [539.48 ± 0.49], [580.15 ± 1.84],
  [large], [360.57 ± 0.91], [1144.19 ± 0.52], [1231.76 ± 1.07],
  [xl], [1094.99 ± 0.79], [3387.57 ± 1.17], [3663.73 ± 2.87],
  [10B], [OOM], [OOM], [OOM],
)

以 medium 为例，forward 约 172ms，backward（fwdbwd − fwd）约 367 ms，optimizer step（fwdbwdopt − fwdbwd）约 40 ms；$"backward" approx 2.1 times "forward"$，符合反向需要额外两次矩阵乘的理论预期。跨 10 次测量的标准差都在均值的 1% 以内，说明稳态下计时非常稳定。

(c) 
当 warmup 为 0 时：

#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],

  [small], [123.74 ± 196.02], [218.04 ± 106.22], [207.88 ± 26.37],
  [medium], [217.55 ± 134.48], [544.26 ± 11.12], [584.12 ± 13.58],
  [large], [403.28 ± 128.47], [1163.20 ± 57.32], [1248.76 ± 40.78],
  [xl], [1142.49 ± 142.61], [3387.72 ± 3.07], [3664.13 ± 8.49],
  [10B], [OOM], [OOM], [OOM],
)

不进行 warmup 时，第一次迭代可能包含 CUDA 运行时的惰性初始化、显存分配等一次性开销，因此会显著拉高平均时间，并导致标准差增大。例如 small 的 forward 从 57.95 ms 增加到 123.74 ms，标准差也从 0.31 ms 增加到 196.02 ms。加入 warmup 后，这些一次性开销被排除，测量结果因此稳定得多。

当 warmup 为1时：
#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],
  [small], [58.11 ± 0.96], [182.36 ± 1.81], [201.08 ± 5.89],
  [medium], [172.77 ± 0.63], [540.76 ± 0.84], [581.43 ± 2.20],
  [large], [359.73 ± 0.32], [1145.49 ± 1.31], [1234.01 ± 0.79],
  [xl], [1094.67 ± 0.68], [3387.93 ± 1.94], [3662.70 ± 2.82],
  [10B], [OOM], [OOM], [OOM],
)

当 warmup 为 2 时：

#table(
  columns: 4,
  [mode], [fwd], [fwdbwd], [fwdbwdopt],
  [size], [], [], [],

  [small], [57.76 ± 0.16], [180.98 ± 0.43], [198.07 ± 4.90],
  [medium], [172.62 ± 0.23], [539.98 ± 0.62], [580.17 ± 1.14],
  [large], [360.43 ± 0.85], [1144.63 ± 1.32], [1233.48 ± 0.64],
  [xl], [1094.85 ± 0.69], [3388.84 ± 1.04], [3662.90 ± 3.47],
  [10B], [OOM], [OOM], [OOM],
)

当 warmup 为 1 或 2 时，结果已经与 warmup=5 非常接近，说明 1--2 次 warmup 已经基本消除了首次运行开销。剩余的细微差异可能来自少量初始化/分配开销以及 GPU 测量本身的波动。

  ],
)