#import "../../template.typ": answer
#answer[
  (a)

Transformer LM

assume $B = "batch_size", T = "context_length", L = "num_layers", H = "num_heads", V = "vocab_size", d = d_"model"$

- Token Embedding
  - Parameter: $V times d$
- $L times $Transformer Block
  - Total:
    - Parameter: $2 times L times d times (1 + 6 times d)$
    - Activation: $2/3 times B times T times L times(3+3 times T times H + 28 times d)$
  - Each Part:
    - RMSNorm(ln1)
      - Parameter: $d$
      - Activation:
        - rms: batch_size context_length 1
          - $B times T$
        - output: batch_size context_length d_model
          - $B times T times d$
    - Causal Multi-Head Self-Attention (attn)
      - QKV projections
        - Parameter: $3 times d^2$
        - Activation:
          - batch_size context_length d_model -> batch_size context_length d_model
          - $3 times B times T times d$
      - $Q times K^T$ matrix multiply
        - Activation:
          - [batch_size context_length d_model] [batch_size context_length d_model] -> batch_size num_heads context_length context_length
          - $B times H times T times T$
      - softmax
        - Activation:
          - batch_size num_heads context_length context_length
          - $B times H times T times T$
      - weighted sum of values
        - Activation:
          - batch_size context_length d_model
          - $B times T times d$
      - output projections
        - Parameter: $d^2$
        - Activation:
          - batch_size context_length d_model -> batch_size context_length d_model
          - $B times T times d$
    - RMSNorm(ln2)
      - Parameter: $d$
      - Activation:
        - rms: batch_size context_length 1
          - $B times T$
        - output: batch_size context_length d_model
          - $B times T times d$
    - SwiGLU(ffn)
      - Parameter($W_1,W_2,W_3$): $3 times d times d_"ff"$
      - Activation:
        - $W_1$ projection:
          - batch_size context_length d_model -> batch_size context_length d_ff
          - $B times T times d_"ff"$
        - $W_3$ projection:
          - batch_size context_length d_model -> batch_size context_length d_ff
          - $B times T times d_"ff"$
        - SiLU:
          - batch_size context_length d_ff
          - $B times T times d_"ff"$
        - element-wise product:
          - batch_size context_length d_ff
          - $B times T times d_"ff"$
        - $W_2$ projection:
          - batch_size context_length d_ff -> batch_size context_length d_model
          - $B times T times d_"model"$
- RMSNorm(ln_final)
  - Parameter: $d$  
  - Activation:
    - rms: batch_size context_length 1
      - $B times T$
    - output: batch_size context_length d_model
      - $B times T times d$
- Linear(lm_head)
  - Parameter: $d  times  V$
  - Activation:
    - batch_size context_length d_model -> batch_size context_length vocab_size
    - $B times T times V$
- cross-entropy on logits
  - Activation: $B times T$

Elements:
  - Parameter: $d times (1+2 times V+2 times L+12 times L times d)$
  - Activation: $1/3 times B times T times (6+3 times V+6 times T times L times H+6 times L+56 times L times d+3 times d)$
  - Gradients: $d times (1+2 times V+2 times L+12 times L times d)$
  - Optimizer state: $2 times d times (1+2 times V+2 times L+12 times L times d)$
  - Total: $1/3 times (3 times B times V times T+6 times B times T+6 times B times T times L+56 times B times T times L times d+3 times B times T times d+6 times B times T^2 times L times H+24 times V times d+24 times L times d+144 times L times d^2+12 times d)$

Bytes:
  - Parameter: $4 times d times (1+2 times V+2 times L+12 times L times d)$
  - Activation: $4/3 times B times T times (6+3 times V+6 times T times L times H+6 times L+56 times L times d+3 times d)$
  - Gradients: $4 times d times (1+2 times V+2 times L+12 times L times d)$
  - Optimizer state: $8 times d times (1+2 times V+2 times L+12 times L times d)$
  - Total: $4/3 times (3 times B times V times T+6 times B times T+6 times B times T times L+56 times B times T times L times d+3 times B times T times d+6 times B times T^2 times L times H+24 times V times d+24 times L times d+144 times L times d^2+12 times d)$

(b) 
此处使用 GPT-2 XL 的实际对齐值 $d_"ff" = 4288$

In GPT-2 XL-shaped model:
- Parameters($1,640,452,800$ elements): $6,561,811,200$ bytes.
- Activations($4,041,985,024 times B$ elements) : $16,167,940,096 B$ bytes.
- Gradients($1,640,452,800$ elements): $6,561,811,200$ bytes.
- Optimizer states($3,280,905,600$ elements): $13,123,622,400$ bytes.
- Total($4,041,985,024 times B + 6,561,811,200 $ elements):$16,167,940,096B+26,247,244,800$ bytes (about $15.06B+24.44$ GiB)

可以适配的最大 batch size 是 3

(c) 记总参数量为 $P$
- Apply weight decay: $2P$
- Update the first moment estimate: $3P$
- Update the second moment estimate: $4P$
- Apply moment-adjusted weight updates: $5P$

所以总 FLOP 为 $14P$

(d)$"Nvidia H100 GPU peak" =  495 "teraFLOP/s" = 495 times 10^12 "FLOP/s"$

$"observed throughput" = "MFU" times "peak" = 50% times 495 times 10^12 "FLOP/s"$

$"forward total FLOPs" = 400,000 times 1,024 times 3,516,769,894,400 = 1,440,468,948,746,240,000,000  "FLOPs"$

$"backward total FLOPs" = 2 times 400,000 times 1,024 times 3,516,769,894,400 = 2,880,937,897,492,480,000,000 "FLOPs"$

$"AdamW total FLOPs" = 400,000 times 14 times P = 400,000 times 14 times 1,640,452,800 = 9,186,535,680,000,000 "FLOPs"$

$"time" = ("forward total FLOPs"+"backward total FLOPs"+"AdamW total FLOPs")/"observed throughput" = (4,321,416,032,774,400,000,000)/(50% times 495 times 10^12) approx 17,460,267 "s" approx 4,850 "hours"$
]
