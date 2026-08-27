from symbolica import S, Expression
from decimal import Decimal

# model
# vocab_size, context_length, num_layers, d_model, num_heads, d_ff
vocab_size, context_length, num_layers, d_model, num_heads, d_ff = S(
    "vocab_size", "context_length", "num_layers", "d_model", "num_heads", "d_ff"
)

GPT2_XL: dict[Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 48,
    d_model: 1600,
    num_heads: 25,
    d_ff: 4288,
}

# Trainable parameters
## token_embeddings
TP_token_embeddings = vocab_size * d_model

## layers
### ln1
TP_block_ln1 = d_model

### attn
#### q_proj
TP_block_attn_q_proj = d_model * d_model
#### k_proj
TP_block_attn_k_proj = d_model * d_model
#### v_proj
TP_block_attn_v_proj = d_model * d_model
#### output_proj
TP_block_attn_output_proj = d_model * d_model

TP_block_attn = (
    TP_block_attn_q_proj
    + TP_block_attn_k_proj
    + TP_block_attn_v_proj
    + TP_block_attn_output_proj
)

### ln2
TP_block_ln2 = d_model

### ffn
#### w1
TP_block_ffn_w1 = d_model * d_ff
#### w2
TP_block_ffn_w2 = d_model * d_ff
#### w3
TP_block_ffn_w3 = d_model * d_ff

TP_block_ffn = TP_block_ffn_w1 + TP_block_ffn_w2 + TP_block_ffn_w3

TP_layers = (TP_block_ln1 + TP_block_attn + TP_block_ln2 + TP_block_ffn) * num_layers

## ln_final
TP_ln_final = d_model

## lm_head
TP_lm_head = d_model * vocab_size

TP = TP_token_embeddings + TP_layers + TP_ln_final + TP_lm_head

print(
    f"GPT-2 XL has {int(TP.evaluate(GPT2_XL).real)} trainable parameters, which is {int(TP.evaluate(GPT2_XL).real)*4} bytes, about {int(TP.evaluate(GPT2_XL).real)*4/1024/1024/1024} GiB."
)
