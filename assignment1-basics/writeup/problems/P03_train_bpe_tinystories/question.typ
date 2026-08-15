#import "../../template.typ": problem, deliverable, note

#problem("train_bpe_tinystories", "BPE Training on TinyStories", "2 points")[
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

#note[Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM]

#note[Hint: You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:

(a) The `<|endoftext|>` token delimits documents in the data files.

(b) The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.]

#deliverable[A one-to-two sentence response.]

(b) Profile your code. What part of the tokenizer training process takes the most time?

#deliverable[A one-to-two sentence response.]
]
