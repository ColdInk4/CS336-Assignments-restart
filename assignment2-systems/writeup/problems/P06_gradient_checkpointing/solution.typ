#import "../../template.typ": solution, deliverable

#solution(
  "gradient_checkpointing",
  "Memory-Optimal Gradient Checkpointing",
  "4 points",
  [
    Consider a Transformer with #raw("N") identical blocks stacked sequentially. Without any checkpointing, all #raw("N") blocks' worth of residuals are kept alive simultaneously, giving #raw("O(N)") peak activation memory. We have a free hand to wrap any subset of the forward pass in #raw("checkpoint"), including nesting #raw("checkpoint") calls inside one another.

    (a) What checkpointing strategy minimizes peak activation memory, ignoring the compute cost? Describe how you would arrange the #raw("checkpoint") calls (a code sketch is fine), and give the asymptotic peak activation memory and compute cost of your strategy as a function of #raw("N"). Assume the residuals saved by a single block dominate any per-checkpoint bookkeeping.

    #deliverable[A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.]

    (b) Consider the #raw("xl") model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest #raw("checkpoint") calls), what is the best checkpointing strategy to reduce peak memory? Profile your run's peak memory to validate your hypothesis. Compare the peak memory of the next smaller and larger checkpointing block sizes to be sure.

    #deliverable[A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.]
  ],
  answer: [
    (a)
    
    记  C 为一个 checkpoint 入口存下的张量，也就是一份 residual stream 激活 [B, T, d_model]。xl、B=4、T=2048 时是 80 MiB。
    
    记  R 是单个 block 前向时 autograd 存下的全部张量，包括 block 的输入、不含参数

    策略为不断二分checkpoint，峰值为 $C log_2N +R$，总计算量为 $O(N log N)$
    
    
    不嵌套时，若平均分为 $k$ 段，则有 $k$ 个checkpoint和 $N/k$ 个residual，由于峰值时刻总有一段正在重算。这一段的 checkpoint 入口必须在显存里，因为重算要从它出发；而重算出来的第一个 block 要存的输入，正是这个入口张量。所以只要按"所有 checkpoint 入口 + 一段完整的 R"来算，这一个张量就会被算两次。
    
    peak activation memory 为 $ (k-1) C + N/k R $，即当k = $sqrt((N R)/C)$ 有最小值 $ 2 sqrt(N R C) -C ~ O(sqrt(N)) $

    因为此时有 $1<=k<=N$ 的限制

    易得$ min = cases(
  (N-1)C   +R & <=> k =N & "if" C N < R ,
  2 sqrt(N R C) -C ~ O(sqrt(N)) & <=> k = sqrt((N R)/C) & "if" C N >= R,
) $


    根据均值不等式 $ (a_1+a_2+...+a_n)/n>=root(n, a_1 a_2 ... a_n) $

    嵌套两层，若平均分为$k_1$,$k_2$ 段，则peak activation memory 为 $ &(k_1+k_2-1)C+N/(k_1 k_2)R -C \ >=& 2sqrt(k_1 k_2)C + N/(k_1 k_2)R -2C \ =& sqrt(k_1 k_2)C + sqrt(k_1 k_2)C + N/(k_1 k_2)R -2C\ >=&3root(3, C^2 N R) - 2C ~ O(root(3,N)) $，当且仅当 $k_1 = k_2 = root(3,(N R)/C)$ 时取到。

    嵌套 $L$ 层时，若嵌套中分别平均分为 $k_1, k_2, ...,k_L$，则peak activation memory 为 $ &(k_1+k_2+...+k_L-(L-1))C+N/(k_1 k_2 ... k_L)R -C\
    >=& L root(L, k_1 k_2 ... k_L)C + N/(k_1 k_2  ... k_L)R -L C \
    =& root(L, k_1 k_2 ... k_L)C +...+ root(L, k_1 k_2 ... k_L)C + N/(k_1 k_2... k_L)R -L C \
    >=&(L+1)root(L+1, C^L N R) -L C ~ O(N^(1/(L+1))) $
    当且仅当 $k_1 = k_2 = ... = k_L = root(L+1,(N R)/C)$ 时取到。

    记 $x = (L+1), M = (N R)/C$， 最小值可转化为：

    $ &x M^(1/x) C - (x-1)C \
    = &(x M^(1/x) - x + 1)C $

    对 $f(x) = x M^(1/x) - x$ 求导得 
    $ f'(x) = M^(1/x)-(ln M)/x M^(1/x) - 1 $

    易证 $f'(x) <=0$ ，即 $L$ 越大，最小值越小

    为了让峰值最小，所以要确保 $L$ 最大，显然 $L$ 最大时 $k_1 = k_2 = ... = k_L = 2$，且 $ N/2^L >= 1 $
    
    
    则 peak activation memory 为 
    $   &(k_1+k_2+...+k_L-(L-1))C+N/(k_1 k_2 ... k_L)R - C\ 
      = &(2L-L)C+N/(2^L) R \
      = &L C+N/(2^L) R 
    $

    通过求导得到，当 $ L = log_2((N R ln 2)/C ) $ 时，有最小值 $ C log_2((N R ln 2)/C ) + C/ (ln 2)$

    为满足$ N/2^L >= 1 $的约束，则有

    $ min = cases(
  C log_2N  +R & <=> L = log_2 N & "if" C<=R ln 2 ,
  (log_2((N R ln 2)/C ) + 1/ (ln 2))C & <=> L = log_2((N R ln 2)/C ) & "if" C > R ln 2,
) ~ O(log_2 N) $

    由题意知 $R>>C$，所以总是取第一种情况，即 $min = C log_2N  +R <=> L = log_2 N $

    综上所述，策略为不断二分checkpoint，峰值为 $C log_2N  +R$

    递归一共 $log_2 N$层，每一层所有段的重算加起来最多涉及 $N$ 个 block，所以总计算量是 $O(N log N)$

#table(
  columns: 4,
  align: center,
  table.header(
    [], [不做 checkpoint], [一层 checkpoint], [二分递归],
  ),
  [峰值内存], [$O(N)$], [$O(sqrt(N))$], [$O(log N)$],
  [计算量], [$O(N)$], [$O(N)$], [$O(N log N)$],
)

code sketch见 `writeup/problems/P06_gradient_checkpointing/short_code.py`

对于我们的xl，跑出来一个Block 的 residual 约为 5,702.69MiB，一个 checkpoint 的大小约为 80MiB。通过二分策略后的 peak activation memory即为 $5 times 80 + 5,702.69 = 6102.69 "MiB"$


(b) 不嵌套时，由于 $C times N = 80 times 32 = 2560 < 5702.69 = R$ ，由(a)得到： peak activation memory 为 

$ &(N-1) C + R  \ =& (32-1) times 80 + 5,702.69\ =& 8,182.69 "MiB" $

此时 $k = 32$。只要有一段至少2个block，峰值至少$2R$，比$31C+R$大。

测得峰值-基线为 9756.06 MiB。

继续测其他组合：

#table(
  columns: 5,
  align: center,
  table.header(
    [$k$], [段长 $N/k$], [式子 $(k-1)C + N/k R$ (MiB)], [实测：峰值 − 基线 (MiB)], [实测 − 式子 (MiB)],
  ),
  [32], [1], [8182.69], [9756.06], [1573.37],
  [16], [2], [12605.38], [14179.13], [1573.75],
  [8], [4], [23370.76], [24945.25], [1574.49],
)

各个都大致为预测值加上一个常数项，因为式子只算了 saved tensors，常数是 backward 时一个 block 的工作内存。

验证得到 k=32 在当前状态（xl，fp32、B=4、T=2048）下最好

  ],

)