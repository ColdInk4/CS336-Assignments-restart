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
  ],
)