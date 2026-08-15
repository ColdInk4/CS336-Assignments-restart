#import "../../template.typ": problem, deliverable

#problem("layer_norm_ablation", "Remove RMSNorm and train (0.5 B200 hrs)", "1 point")[
Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?

#deliverable[A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.]

#deliverable[A few sentences of commentary on the impact of RMSNorm.]
]
