#import "../../template.typ": answer
#answer[
  (a) 为了确定合适的学习率，我首先进行了一个粗粒度搜索。固定模型结构、batch size、优化器等其他超参数，仅改变 AdamW 的初始最大学习率，分别测试：$10^(-4), 3 times 10^(-4), 10^(-3), 3 times 10^(-3), 10^(-2)$，每组实验训练 $1000$ steps。

粗搜索结果如下：
#image("/assets/image.png")

#table(
columns: 3,
align: center,
[学习率], [Train Loss], [Validation Loss],
[$10^(-4)$], [2.8791], [2.7990],
[$3 times 10^(-4)$], [2.3151], [2.2591],
[$10^(-3)$], [1.8797], [1.8376],
[$3 times 10^(-3)$], [*1.7586*], [*1.7287*],
[$10^(-2)$], [2.7169], [2.6335],
)

在Step = 1000下对 $2 times 10^(-3)$，$4 times 10^(-3)$训练

#image("/assets/image-1.png")

#table(
columns: 3,
align: center,
[学习率], [Train Loss], [Validation Loss],
[$2 times 10^(-3)$], [1.7600], [1.7299],
[$3 times 10^(-3)$], [*1.7586*], [*1.7287*],
[$4 times 10^(-3)$], [1.8136], [1.7780],
)
在这组实验中，$3 times 10^(-3)$ 仍然取得了最低的 Train Loss 和 Validation Loss。因此，在当前实验预算下选择 $3 times 10^(-3)$ 作为后续完整训练的学习率。

最终训练得到的`train_loss`为 1.3836，`val_loss`为 1.3369，

#figure(
  grid(
    columns: (1fr, 1fr),   
    gutter: 1em,          
    image("/assets/image-2.png", width: 100%),
    image("/assets/image-3.png", width: 100%),
  ),
)

(b)从大于$10^(-2)$的lr开始寻找稳定边界。固定训练 1000 steps，逐渐增大学习率，观察 loss 曲线变化。

首先测试较大的学习率：

- lr 从 $0.02$ 到 $4$
#image("/assets/image-4.png", width: 70%)

可以看到，虽然这些学习率相比最优学习率 $3 times 10^(-3)$ 已经明显增大，但部分实验仍然能够下降，因此仅观察最终是否 NaN 并不能准确刻画稳定边界。

进一步增大学习率：

- lr 从 $10$ 到 $200$
#image("/assets/image-5.png", width: 70%)

可以观察到 loss 开始出现明显震荡，训练过程不再表现为稳定下降，但仍未立即产生 NaN。

当学习率继续增大到$300$ 以上后
#image("/assets/image-6.png", width: 70%)

训练出现出现NaN，说明已经进入严重发散区域。

但是 NaN 并不是 stability edge 的位置。因为在这之前已经有很多训练是无法stable的了

在较小范围内进一步观察：

- lr 从 $0.001$ 到 $0.5$


#image("/assets/image-7.png", width: 70%)

可以看到：

- $lr <= 4 times 10^(-3)$ 时，loss 持续下降；
- $lr approx 10^(-2)$ 附近开始出现较高 loss plateau；
- $lr = 0.02 ~ 0.05$ 时出现明显震荡。
更大的学习率导致更强的不稳定。

这说明最佳学习率并不等于稳定边界，但位于其附近的稳定区域内。该结果与 “best learning rate is often close to the edge of stability” 的经验规律一致。

]
