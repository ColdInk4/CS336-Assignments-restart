#import "../../template.typ": problem, deliverable

#problem("main_experiment", "Experiment on OWT (2 B200 hrs)", "2 points")[
Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. You may need to re-tune hyperparameters such as learning rate or batch size.

#deliverable[A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories, and how those losses should be interpreted.]

#deliverable[Generated text from the OpenWebText language model, in the same format as the TinyStories outputs. Comment on the fluency of this text and why the output quality is worse even with the same model and compute budget as TinyStories.]
]
