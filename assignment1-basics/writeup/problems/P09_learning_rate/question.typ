#import "../../template.typ": problem, deliverable, note

#problem("learning_rate", "Tune the learning rate (2 B200 hrs)", "3 points")[
The learning rate is one of the most important hyperparameters to tune. Taking the base model you've trained, answer the following questions.

(a) Perform a hyperparameter sweep over the learning rates and report the final losses, or note divergence if the optimizer diverges.

#deliverable[Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.]

#deliverable[A model with validation loss, per token, on TinyStories of at most `1.45`.]

#note[Low-Resource Tip: Train for a few steps on CPU or Apple Silicon

If you are running on `cpu` or `mps`, you should instead reduce the total tokens processed count to `40,000,000`, which will be sufficient to produce reasonably fluent text. You may also increase the target validation loss from `1.45` to `2.00`.

Running our solution code with a tuned learning rate on an M4 Max chip and `36 GB` of RAM, we use `batch size × total step count × context length = 32 × 5000 × 256 = 40,960,000` tokens, which takes `1 hour and 22 minutes` on `cpu` and `36 minutes` on `mps`. At step `5000`, we achieve a validation loss of `1.80`.]

Some Additional Tips

- When using `N` training steps, we suggest adjusting the cosine learning rate decay schedule to terminate its decay, that is, reach the minimum learning rate, at precisely step `N`.

- When using `mps`, do not use TF32 kernels. In particular, do not set:

#raw("torch.set_float32_matmul_precision(\"high\")", block: true, lang: "python")

as you might with CUDA devices. We tried enabling TF32 kernels with `mps`, using `torch` version `2.9.0`, and found the backend sometimes uses silently broken kernels that cause unstable training.

- You can speed up training by JIT-compiling your model with `torch.compile`.

Specifically:

#raw("# On cpu\nmodel = torch.compile(model)\n\n# On mps\nmodel = torch.compile(model, backend=\"aot_eager\")", block: true, lang: "python")

Compilation with Inductor is not supported on `mps` as of `torch` version `2.9.0`.

(b) Folk wisdom is that the best learning rate is "at the edge of stability." Investigate how the point at which learning rates diverge is related to your best learning rate.

#deliverable[Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.]
]
