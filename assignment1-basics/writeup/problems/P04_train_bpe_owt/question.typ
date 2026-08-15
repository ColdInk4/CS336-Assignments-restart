#import "../../template.typ": problem, deliverable, note

#problem("train_bpe_expts_owt", "BPE Training on OpenWebText", "2 points")[
(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

#note[Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM]

#deliverable[A one-to-two sentence response.]

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

#deliverable[A one-to-two sentence response.]
]
