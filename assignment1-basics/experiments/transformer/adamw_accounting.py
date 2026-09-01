from symbolica import S, Expression, Replacement
from decimal import Decimal
from typing import cast

# model
batch_size = S("B")
# vocab_size, context_length, num_layers, d_model, num_heads, d_ff
vocab_size, context_length, num_layers, d_model, num_heads, d_ff = S(
    "V", "T", "L", "d", "H", "d_ff"
)


# token_embeddings
TP_token_embeddings = vocab_size * d_model

# layers (each block)
## ln1/RMSNorm
TP_block_ln1 = d_model

## attn
### q_proj
TP_block_attn_q_proj = d_model * d_model
### k_proj
TP_block_attn_k_proj = d_model * d_model
### v_proj
TP_block_attn_v_proj = d_model * d_model
### output_proj
TP_block_attn_output_proj = d_model * d_model

TP_block_attn = (
    TP_block_attn_q_proj
    + TP_block_attn_k_proj
    + TP_block_attn_v_proj
    + TP_block_attn_output_proj
)

## ln2
TP_block_ln2 = d_model

## ffn
### w1
TP_block_ffn_w1 = d_model * d_ff
### w2
TP_block_ffn_w2 = d_model * d_ff
### w3
TP_block_ffn_w3 = d_model * d_ff

TP_block_ffn = TP_block_ffn_w1 + TP_block_ffn_w2 + TP_block_ffn_w3

TP_layers = (TP_block_ln1 + TP_block_attn + TP_block_ln2 + TP_block_ffn) * num_layers

# ln_final
TP_ln_final = d_model

# lm_head
TP_lm_head = d_model * vocab_size

P = TP_token_embeddings + TP_layers + TP_ln_final + TP_lm_head

# Transformer block
## RMSNorm - ln1
### batch_size context_length d_model
AC_block_ln1 = batch_size * context_length * d_model + batch_size * context_length

## Multi-head self-attention sublayer
### Q proj - batch_size context_length d_model
### K proj - batch_size context_length d_model
### V proj - batch_size context_length d_model
AC_block_attn_q_proj = batch_size * context_length * d_model
AC_block_attn_k_proj = batch_size * context_length * d_model
AC_block_attn_v_proj = batch_size * context_length * d_model

### QK^T matrix multiply - batch_size num_heads sequence_length sequence_length
AC_block_attn_qk = batch_size * num_heads * context_length * context_length

### softmax - batch_size num_heads sequence_length sequence_length
AC_block_attn_softmax = batch_size * num_heads * context_length * context_length

### weighted sum of values - batch_size sequence_length d_model
AC_block_attn_weighted_sum = batch_size * context_length * d_model

### output proj - batch_size context_length d_model
AC_block_attn_output_proj = batch_size * context_length * d_model

AC_block_attn = (
    AC_block_attn_q_proj
    + AC_block_attn_k_proj
    + AC_block_attn_v_proj
    + AC_block_attn_qk
    + AC_block_attn_softmax
    + AC_block_attn_weighted_sum
    + AC_block_attn_output_proj
)

## RMSNorm - ln2
### batch_size context_length d_model
AC_block_ln2 = batch_size * context_length * d_model + batch_size * context_length

## SwiGLU
### W1 - batch_size context_length d_ff
AC_block_ffn_w1 = batch_size * context_length * d_ff

### W3 - batch_size context_length d_ff
AC_block_ffn_w3 = batch_size * context_length * d_ff

### SiLU - batch_size context_length d_ff
AC_block_ffn_SiLU = batch_size * context_length * d_ff

### element-wise product - batch_size context_length d_ff
AC_block_ffn_product = batch_size * context_length * d_ff

### W2 - batch_size context_length d_model
AC_block_ffn_w2 = batch_size * context_length * d_model

AC_block_ffn = (
    AC_block_ffn_w1
    + AC_block_ffn_w2
    + AC_block_ffn_w3
    + AC_block_ffn_SiLU
    + AC_block_ffn_product
)

AC_layers = (AC_block_ln1 + AC_block_attn + AC_block_ln2 + AC_block_ffn) * num_layers

# final RMSNorm - batch_size context_length d_model
AC_ln_final = batch_size * context_length * d_model + batch_size * context_length

# output embedding - batch_size context_length vocab_size
AC_output_embedding = batch_size * context_length * vocab_size

# cross-entropy on logits - batch_size context_length
AC_cross_entropy = batch_size * context_length

Ac = AC_layers + AC_ln_final + AC_output_embedding + AC_cross_entropy

# d_ff = 8 * d_model / 3

P_element = P.factor()
Ac_element = Ac.factor()
gradients_element = P.factor()
optimizer_state_element = (2 * P).factor()
total_element = (
    P_element + Ac_element + gradients_element + optimizer_state_element
).factor()

P_Bytes = 4 * P_element
Ac_Bytes = 4 * Ac_element
gradients_Bytes = 4 * gradients_element
optimizer_state_Bytes = 4 * optimizer_state_element
total_Bytes = 4 * total_element

print(f"""
Elements:
    Parameter: {P_element}
    Activation: {Ac_element}
    Gradients: {gradients_element}
    Optimizer state: {optimizer_state_element}
    Total: {(P_element+Ac_element+gradients_element+optimizer_state_element).factor()}
Bytes:
    Parameter: {P_Bytes}
    Activation: {Ac_Bytes}
    Gradients: {gradients_Bytes }
    Optimizer state: {optimizer_state_Bytes}
    Total: {total_Bytes}
      """)


GPT2_XL: dict[Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 48,
    d_model: 1600,
    num_heads: 25,
    d_ff: 4288,
}

Ac_gpt2_xl = Ac_element.replace_multiple(
    [
        Replacement(vocab_size, cast(int, GPT2_XL[vocab_size])),
        Replacement(context_length, cast(int, GPT2_XL[context_length])),
        Replacement(num_layers, cast(int, GPT2_XL[num_layers])),
        Replacement(d_model, cast(int, GPT2_XL[d_model])),
        Replacement(num_heads, cast(int, GPT2_XL[num_heads])),
        Replacement(d_ff, cast(int, GPT2_XL[d_ff])),
    ]
)
total_gpt2_xl = total_element.replace_multiple(
    [
        Replacement(vocab_size, cast(int, GPT2_XL[vocab_size])),
        Replacement(context_length, cast(int, GPT2_XL[context_length])),
        Replacement(num_layers, cast(int, GPT2_XL[num_layers])),
        Replacement(d_model, cast(int, GPT2_XL[d_model])),
        Replacement(num_heads, cast(int, GPT2_XL[num_heads])),
        Replacement(d_ff, cast(int, GPT2_XL[d_ff])),
    ]
)

print(f"""
GPT-2 XL has {(int(P_element.evaluate(GPT2_XL).real)):,} parameters, which is {(int(P_Bytes.evaluate(GPT2_XL).real)):,} bytes, about {(int(P_Bytes.evaluate(GPT2_XL).real)/1024/1024/1024):,} GiB.
GPT-2 XL has {Ac_gpt2_xl} activations, which is {4 * Ac_gpt2_xl} bytes, about {4*Ac_gpt2_xl/1024/1024/1024} GiB.
GPT-2 XL has {(int(gradients_element.evaluate(GPT2_XL).real)):,} gradients, which is {(int(gradients_Bytes.evaluate(GPT2_XL).real)):,} bytes, about {(int(gradients_Bytes.evaluate(GPT2_XL).real)/1024/1024/1024):,} GiB.
GPT-2 XL has {(int(optimizer_state_element.evaluate(GPT2_XL).real)):,} optimizer states, which is {(int(optimizer_state_Bytes.evaluate(GPT2_XL).real)):,} bytes, about {(int(optimizer_state_Bytes.evaluate(GPT2_XL).real)/1024/1024/1024):,} GiB
Total: {total_gpt2_xl} elements, {(total_gpt2_xl*4).expand()} bytes, which is  {(total_gpt2_xl*4/1024/1024/1024).expand()} GiB
""")
