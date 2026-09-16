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
  ],
)