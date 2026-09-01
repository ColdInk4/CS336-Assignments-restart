#import "../../template.typ": answer
#answer[
  对于我们的模型，可训练的参数为 
  $2 times "vocab_size" times d_"model"+"num_layers" times (2 times d_"model"+3 times d_"model" times d_"ff"+4 times d_"model"^2)+d_"model"$
  
  总 FLOPs 为 $2 times "vocab_size" times "context_length" times d_"model" + "num_layers" times (6 times "context_length" times d_"model" times d_"ff"+8 times "context_length" times d_"model"^2+4 times "context_length"^2 times d_"model")$

  (a) GPT-2 XL has 1,640,452,800 trainable parameters, which is 6,561,811,200 bytes, about 6.11 GiB.

  (b) assuming batch size 1, GPT-2 XL has 3,516,769,894,400 FLOPs. 
  `
Total: 3,516,769,894,400 FLOPs
|- layers: 3,352,087,756,800 FLOPs (95.32%)
  |- attn: 1,328,755,507,200 FLOPs (37.78%)
    |- q_proj: 251,658,240,000 FLOPs (7.16%)
    |- k_proj: 251,658,240,000 FLOPs (7.16%)
    |- v_proj: 251,658,240,000 FLOPs (7.16%)
    |- output_proj: 251,658,240,000 FLOPs (7.16%)
    |- `$Q K^T$`: 161,061,273,600 FLOPs (4.58%)
    |- `$"softmax"((Q K^T)/ sqrt(d_k)) V$`: 161,061,273,600 FLOPs (4.58%)
  |- FFN: 2,023,332,249,600 FLOPs (57.53%)
    |- w1: 674,444,083,200 FLOPs (19.18%)
    |- w2: 674,444,083,200 FLOPs (19.18%)
    |- w3: 674,444,083,200 FLOPs (19.18%)
|- LM head: 164,682,137,600 FLOPs (4.68%)
`
  (c) The FFN requires the most FLOPs.

  (d) 可以注意到，随着参数量的增长，Transformer blocks 总占比从 72.90% 升至 92.55%；attention 从 33.13% 升至 38.25%，FFN 从 39.76% 升至 54.30%；LM head 从 27.10% 降至 7.45%。
  `
GPT-2 small has 291,648,307,200 FLOPs
Total: 291,648,307,200 FLOPs
|- layers: 212,600,881,152 FLOPs (72.90%)
  |- attn: 96,636,764,160 FLOPs (33.13%)
    |- q_proj: 14,495,514,624 FLOPs (4.97%)
    |- k_proj: 14,495,514,624 FLOPs (4.97%)
    |- v_proj: 14,495,514,624 FLOPs (4.97%)
    |- output_proj: 14,495,514,624 FLOPs (4.97%)
    |- `$Q K^T$`: 19,327,352,832 FLOPs (6.63%)
    |- `$"softmax"((Q K^T)/ sqrt(d_k)) V$`: 19,327,352,832 FLOPs (6.63%)
  |- FFN: 115,964,116,992 FLOPs (39.76%)
    |- w1: 38,654,705,664 FLOPs (13.25%)
    |- w2: 38,654,705,664 FLOPs (13.25%)
    |- w3: 38,654,705,664 FLOPs (13.25%)
|- LM head: 79,047,426,048 FLOPs (27.10%)
`
`
GPT-2 medium has 830,172,299,264 FLOPs
Total: 830,172,299,264 FLOPs
|- layers: 724,775,731,200 FLOPs (87.30%)
  |- attn: 309,237,645,312 FLOPs (37.25%)
    |- q_proj: 51,539,607,552 FLOPs (6.21%)
    |- k_proj: 51,539,607,552 FLOPs (6.21%)
    |- v_proj: 51,539,607,552 FLOPs (6.21%)
    |- output_proj: 51,539,607,552 FLOPs (6.21%)
    |- `$Q K^T$`: 51,539,607,552 FLOPs (6.21%)
    |- `$"softmax"((Q K^T)/ sqrt(d_k)) V$`: 51,539,607,552 FLOPs (6.21%)
  |- FFN: 415,538,085,888 FLOPs (50.05%)
    |- w1: 138,512,695,296 FLOPs (16.68%)
    |- w2: 138,512,695,296 FLOPs (16.68%)
    |- w3: 138,512,695,296 FLOPs (16.68%)
|- LM head: 105,396,568,064 FLOPs (12.70%)
`
`
GPT-2 large has 1,768,530,903,040 FLOPs
Total: 1,768,530,903,040 FLOPs
|- layers: 1,636,785,192,960 FLOPs (92.55%)
  |- attn: 676,457,349,120 FLOPs (38.25%)
    |- q_proj: 120,795,955,200 FLOPs (6.83%)
    |- k_proj: 120,795,955,200 FLOPs (6.83%)
    |- v_proj: 120,795,955,200 FLOPs (6.83%)
    |- output_proj: 120,795,955,200 FLOPs (6.83%)
    |- `$Q K^T$`: 96,636,764,160 FLOPs (5.46%)
    |- `$"softmax"((Q K^T)/ sqrt(d_k)) V$`: 96,636,764,160 FLOPs (5.46%)
  |- FFN: 960,327,843,840 FLOPs (54.30%)
    |- w1: 320,109,281,280 FLOPs (18.10%)
    |- w2: 320,109,281,280 FLOPs (18.10%)
    |- w3: 320,109,281,280 FLOPs (18.10%)
|- LM head: 131,745,710,080 FLOPs (7.45%)
`

(e) 总 FLOPs 从 3.52e12 增加到 1.34e14，约增加38倍。attention 从 37.78% 升至 73.79%；FFN 从 57.53% 降至 24.24%；LM head 从 4.68% 降至 1.97%。
其中 attention 部分的比重很高，主要是 $Q K^T$和 $"softmax"((Q K^T)/ sqrt(d_k)) V$
`
Total: 133,577,729,638,400 FLOPs
|- layers: 130,942,815,436,800 FLOPs (98.03%)
  |- attn: 98,569,499,443,200 FLOPs (73.79%)
    |- q_proj: 4,026,531,840,000 FLOPs (3.01%)
    |- k_proj: 4,026,531,840,000 FLOPs (3.01%)
    |- v_proj: 4,026,531,840,000 FLOPs (3.01%)
    |- output_proj: 4,026,531,840,000 FLOPs (3.01%)
    |- `$Q K^T$`: 41,231,686,041,600 FLOPs (30.87%)
    |- `$"softmax"((Q K^T)/ sqrt(d_k)) V$`: 41,231,686,041,600 FLOPs (30.87%)
  |- FFN: 32,373,315,993,600 FLOPs (24.24%)
    |- w1: 10,791,105,331,200 FLOPs (8.08%)
    |- w2: 10,791,105,331,200 FLOPs (8.08%)
    |- w3: 10,791,105,331,200 FLOPs (8.08%)
|- LM head: 2,634,914,201,600 FLOPs (1.97%)
`
]
