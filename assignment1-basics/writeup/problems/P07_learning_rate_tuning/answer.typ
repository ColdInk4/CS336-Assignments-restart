#import "../../template.typ": answer
#answer[
  对于相同的初始 weights：

  - 1e1: loss从 27.13 降到 3.64
  - 1e2: loss从 27.13 降到 6.32e-23
  - 1e3: loss从 27.1 3升到 2.51e18

  可以看到，学习率从 1e1 改到 1e2 时 loss 下降地更快，而 1e3 时训练发散


]
