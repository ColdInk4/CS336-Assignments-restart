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
    策略为不断二分checkpoint，峰值为 $(log_2 N +1) C+R$，总计算量为 $O(N log N)$
    
    
     不嵌套时，若平均分为$k$段，则有$k$个checkpoint和$N/k$个residual，peak activation memory 为 $ k C + N/k R $，即当k = $sqrt((N R)/C)$ 有最小值 $ 2 sqrt(N R C) ~ O(sqrt(N)) $

    根据均值不等式 $ (a_1+a_2+...+a_n)/n>=root(n, a_1 a_2 ... a_n) $

    嵌套两层，若平均分为$k_1$,$k_2$ 段，则peak activation memory 为 $ &(k_1+k_2-1)C+N/(k_1 k_2)R \ >=& 2sqrt(k_1 k_2)C + N/(k_1 k_2)R -C \ =& sqrt(k_1 k_2)C + sqrt(k_1 k_2)C + N/(k_1 k_2)R -C\ >=&3root(3, C^2 N R) - C ~ O(root(3,N)) $，当且仅当 $k_1 = k_2 = root(3,(N R)/C)$ 时取到。

    嵌套 $L$ 层时，若嵌套中分别平均分为 $k_1, k_2, ...,k_L$，则peak activation memory 为 $ &(k_1+k_2+...+k_L-(L-1))C+N/(k_1 k_2 ... k_L)R\
    >=& L root(L, k_1 k_2 ... k_L)C + N/(k_1 k_2  ... k_L)R -(L-1)C \
    =& root(L, k_1 k_2 ... k_L)C +...+ root(L, k_1 k_2 ... k_L)C + N/(k_1 k_2... k_L)R -(L-1)C \
    >=&(L+1)root(L+1, C^L N R) -(L-1)C ~ O(N^(1/(L+1))) $
    当且仅当 $k_1 = k_2 = ... = k_L = root(L+1,(N R)/C)$ 时取到。

    记 $x = (L+1), M = (N R)/C$， 最小值可转化为：

    $ &x M^(1/x) C - (x-2)C \
    = &(x M^(1/x) - x + 2)C $

    对 $f(x) = x M^(1/x) - x$ 求导得 
    $ f'(x) = M^(1/x)-(ln M)/x M^(1/x) - 1 $

    易证 $f'(x) <=0$ ，即 $L$ 越大，最小值越小

    为了让峰值最小，所以要确保 $L$ 最大，显然 $L$ 最大时 $k_1 = k_2 = ... = k_L = 2$，且 $ N/2^L >= 1 $
    
    
    则 peak activation memory 为 
    $   &(k_1+k_2+...+k_L-(L-1))C+N/(k_1 k_2 ... k_L)R\ 
      = &(2L-(L-1))C+N/(2^L) R \
      = &L C+N/(2^L) R  +C
    $

    通过求导得到，当 $ L = log_2((N R ln 2)/C ) $ 时，有最小值 $ C log_2((N R ln 2)/C ) + C/ (ln 2) + C$

    为满足$ N/2^L >= 1 $的约束，则有

    $ min = cases(
  (log_2N +1) C+R & <=> L = log_2 N & "if" C<=R ln 2 ,
  (log_2((N R ln 2)/C ) + 1/ (ln 2) + 1)C & <=> L = log_2((N R ln 2)/C ) & "if" C > R ln 2,
) ~ O(log_2 N) $

    由题意知 $R>>C$，所以总是取第一种情况，即 $min = (log_2 N +1) C+R <=> L = log_2 N $

    综上所述，策略为不断二分checkpoint，峰值为 $(log_2 N +1) C+R$

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
  ],
)