#import "../../template.typ": problem, deliverable

#problem("batch_size_experiment", "Batch size variations (1 B200 hr)", "1 point")[
Vary your batch size all the way from `1` to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like `64` and `128`.

#deliverable[Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.]

#deliverable[A few sentences discussing your findings on batch sizes and their impacts on training.]
]
