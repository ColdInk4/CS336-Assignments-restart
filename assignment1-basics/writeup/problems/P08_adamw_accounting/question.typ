#import "../../template.typ": problem, deliverable

#problem("adamw_accounting", "Resource accounting for training with AdamW", "2 points")[
Let us compute how much memory and compute running AdamW requires. Assume we are using `float32` for every tensor.

(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of `batch_size` and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume `d_ff = 8 / 3 × d_model`.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block: RMSNorm(s)
- Transformer block: multi-head self-attention sublayer (`QKV` projections, `QK^T` matrix multiply, softmax, weighted sum of values, output projection)
- Transformer block: position-wise feed-forward (SwiGLU): `W1`, `W2`, SiLU on the gate branch, element-wise product, `W3`
- Final RMSNorm
- Output embedding
- Cross-entropy on logits

#deliverable[An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.]

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on `batch_size`. What is the maximum batch size you can use and still fit within `80 GB` memory?

#deliverable[An expression that looks like `a * batch_size + b` for numerical values `a`, `b`, and a number representing the maximum batch size.]

(c) How many FLOPs does running one step of AdamW take?

#deliverable[An algebraic expression, with a brief justification.]

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput, tokens per second, relative to the hardware's theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of `495 teraFLOP/s` for "float32", actually TensorFloat-32, which in reality is "bfloat19", operations. Assuming you are able to get `50% MFU`, how long would it take to train a GPT-2 XL for `400K` steps and a batch size of `1024` on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.

#deliverable[The number of hours training would take, with a brief justification.]
]
