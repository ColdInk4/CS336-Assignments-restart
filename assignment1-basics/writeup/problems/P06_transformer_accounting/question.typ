#import "../../template.typ": problem, deliverable

#problem("transformer_accounting", "Transformer LM resource accounting", "5 points")[
(a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

- `vocab_size: 50,257`
- `context_length: 1,024`
- `num_layers: 48`
- `d_model: 1,600`
- `num_heads: 25`
- `d_ff: 4,288` (the nearest multiple of 64 to `8 / 3 × 1,600`)

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

#deliverable[A one-to-two sentence response.]

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that your input sequence has `context_length` tokens.

#deliverable[A list of matrix multiplies, with descriptions, and the total number of FLOPs required.]

(c) Based on your analysis above, which parts of the model require the most FLOPs?

#deliverable[A one-to-two sentence response.]

(d) Repeat your analysis with GPT-2 small (`12` layers, `768 d_model`, `12` heads), GPT-2 medium (`24` layers, `1024 d_model`, `16` heads), and GPT-2 large (`36` layers, `1280 d_model`, `20` heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

#deliverable[For each model, provide a breakdown of model components and their associated FLOPs as a proportion of the total FLOPs required for a forward pass. In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.]

(e) Take GPT-2 XL and increase the context length to `16,384`. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

#deliverable[A one-to-two sentence response.]
]
