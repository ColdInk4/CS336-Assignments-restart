#import "../../template.typ": problem, deliverable

#problem("tokenizer_experiments", "Experiments with tokenizers", "4 points")[
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary sizes, respectively), encode these sampled documents into integer ids. What is each tokenizer's compression ratio, measured in bytes per token?

#deliverable[A one-to-two sentence response.]

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

#deliverable[A one-to-two sentence response.]

(c) Estimate the throughput of your tokenizer, for example in bytes per second. About how long would it take to tokenize the Pile dataset (825 GB of text)?

#deliverable[A one-to-two sentence response.]

(d) Using your TinyStories and OpenWebText tokenizers, encode the corresponding training and development datasets into sequences of integer token ids. We recommend storing the token ids as a NumPy array with dtype `uint16`. Why is `uint16` an appropriate choice?

#deliverable[A one-to-two sentence response.]
]
