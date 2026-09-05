from symbolica import S, Expression
from decimal import Decimal
from transformer_accounting_TP import TP

# model
# vocab_size, context_length, num_layers, d_model, num_heads, d_ff
vocab_size, context_length, num_layers, d_model, num_heads, d_ff = S(
    "V", "T", "L", "d", "H", "d_ff"
)

GPT2_XL: dict[Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 48,
    d_model: 1600,
    num_heads: 25,
    d_ff: 4288,
}
GPT2_small: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 12,
    d_model: 768,
    num_heads: 12,
    d_ff: round(768 * 8.0 / 3 / 64) * 64,
}
GPT2_medium: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 24,
    d_model: 1024,
    num_heads: 16,
    d_ff: round(1024 * 8.0 / 3 / 64) * 64,
}
GPT2_large: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 36,
    d_model: 1280,
    num_heads: 20,
    d_ff: round(1280 * 8.0 / 3 / 64) * 64,
}
GPT2_XL_Long: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    vocab_size: 50257,
    context_length: 16384,
    num_layers: 48,
    d_model: 1600,
    num_heads: 25,
    d_ff: 4288,
}

# FLOPS
## token_embeddings "... sequence_length -> ... sequence_length d_model "
## omitted, because the question asks for matrix-multiply FLOPs
FLOP_token_embeddings = 0

## layers
### ln1 " ... sequence_length d_model -> ... sequence_length d_model "
### omitted, because the question asks for matrix-multiply FLOPs
FLOP_block_ln1 = 0

### attn

#### q_proj [..., sequence_length, d_model] * [d_model, num_heads * d_k]
FLOP_block_attn_q_proj = 2 * context_length * d_model * d_model

#### k_proj
FLOP_block_attn_k_proj = 2 * context_length * d_model * d_model

#### v_proj
FLOP_block_attn_v_proj = 2 * context_length * d_model * d_model

#### scaled_dot_product_attention
##### Q K^T
##### [num_heads sequence_length d_k] * [num_heads sequence_length d_k] -> [num_heads sequence_length sequence_length]
##### num_heads * 2 * sequence_length * sequence_length * d_k = 2 * sequence_length * sequence_length * d_model
FLOP_block_attn_q_k = 2 * context_length * context_length * d_model
##### (Q K^T) V
##### [num_heads sequence_length sequence_length] * [num_heads sequence_length d_v] -> [num_heads sequence_length d_v]
##### num_heads * 2 * sequence_length * sequence_length * d_v = 2 * sequence_length * sequence_length * d_model
FLOP_block_attn_weights_v = 2 * context_length * context_length * d_model

#### output_proj
FLOP_block_attn_output_proj = 2 * context_length * d_model * d_model

FLOP_block_attn = (
    FLOP_block_attn_q_proj
    + FLOP_block_attn_k_proj
    + FLOP_block_attn_v_proj
    + FLOP_block_attn_output_proj
    + FLOP_block_attn_q_k
    + FLOP_block_attn_weights_v
)

### ln2
### omitted, because the question asks for matrix-multiply FLOPs
FLOP_block_ln2 = 0

### ffn
#### w1 [..., sequence_length, d_model] * [d_model, d_ff]
FLOP_block_ffn_w1 = 2 * context_length * d_model * d_ff
#### w2 [..., sequence_length, d_ff] * [d_ff, d_model]
FLOP_block_ffn_w2 = 2 * context_length * d_model * d_ff
#### w3 [..., sequence_length, d_model] * [d_model, d_ff]
FLOP_block_ffn_w3 = 2 * context_length * d_model * d_ff

FLOP_block_ffn = FLOP_block_ffn_w1 + FLOP_block_ffn_w2 + FLOP_block_ffn_w3

FLOP_layers = (
    FLOP_block_ln1 + FLOP_block_attn + FLOP_block_ln2 + FLOP_block_ffn
) * num_layers

## ln_final
FLOP_ln_final = 0

## lm_head [..., sequence_length, d_model] * [d_model, vocab_size]
FLOP_lm_head = 2 * context_length * d_model * vocab_size

FLOP = FLOP_token_embeddings + FLOP_layers + FLOP_ln_final + FLOP_lm_head

print(FLOP.factor())

print((FLOP / TP).factor())


def compute(
    FLOP,
    model: dict[Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]],
    name: str,
):
    # FLOPs = FLOP_layers + FLOP_lm_head
    total_FLOPs = int(FLOP.evaluate(model).real)
    num_FLOP_layers = int(FLOP_layers.evaluate(model).real)
    num_FLOP_lm_head = int(FLOP_lm_head.evaluate(model).real)

    # FLOP_layers = (FLOP_block_attn + FLOP_block_ffn) * num_layers
    num_FLOP_layers_attn = int((FLOP_block_attn * num_layers).evaluate(model).real)
    num_FLOP_layers_ffn = int((FLOP_block_ffn * num_layers).evaluate(model).real)

    # FLOP_layers_attn = (FLOP_block_attn_q_proj + FLOP_block_attn_k_proj
    #                   + FLOP_block_attn_v_proj + FLOP_block_attn_output_proj
    #                   + FLOP_block_attn_q_k    + FLOP_block_attn_weights_v) * num_layers
    num_FLOP_layers_attn_q_proj = int(
        (FLOP_block_attn_q_proj * num_layers).evaluate(model).real
    )
    num_FLOP_layers_attn_k_proj = int(
        (FLOP_block_attn_k_proj * num_layers).evaluate(model).real
    )
    num_FLOP_layers_attn_v_proj = int(
        (FLOP_block_attn_v_proj * num_layers).evaluate(model).real
    )
    num_FLOP_layers_attn_output_proj = int(
        (FLOP_block_attn_output_proj * num_layers).evaluate(model).real
    )
    num_FLOP_layers_attn_q_k = int(
        (FLOP_block_attn_q_k * num_layers).evaluate(model).real
    )
    num_FLOP_layers_attn_weights_v = int(
        (FLOP_block_attn_weights_v * num_layers).evaluate(model).real
    )

    # FLOP_layers_ffn = (FLOP_block_ffn_w1 + FLOP_block_ffn_w2 + FLOP_block_ffn_w3) * num_layers
    num_FLOP_layers_ffn_w1 = int((FLOP_block_ffn_w1 * num_layers).evaluate(model).real)
    num_FLOP_layers_ffn_w2 = int((FLOP_block_ffn_w2 * num_layers).evaluate(model).real)
    num_FLOP_layers_ffn_w3 = int((FLOP_block_ffn_w3 * num_layers).evaluate(model).real)

    lines = [
        f"{name} has {total_FLOPs:,} FLOPs",
        f"Total: {total_FLOPs:,} FLOPs",
        f"|- layers: {num_FLOP_layers:,} FLOPs ({(num_FLOP_layers/total_FLOPs):.2%})",
        f"  |- attn: {num_FLOP_layers_attn:,} FLOPs ({(num_FLOP_layers_attn/total_FLOPs):.2%})",
        f"    |- q_proj: {num_FLOP_layers_attn_q_proj:,} FLOPs ({(num_FLOP_layers_attn_q_proj/total_FLOPs):.2%})",
        f"    |- k_proj: {num_FLOP_layers_attn_k_proj:,} FLOPs ({(num_FLOP_layers_attn_k_proj/total_FLOPs):.2%})",
        f"    |- v_proj: {num_FLOP_layers_attn_v_proj:,} FLOPs ({(num_FLOP_layers_attn_v_proj/total_FLOPs):.2%})",
        f"    |- output_proj: {num_FLOP_layers_attn_output_proj:,} FLOPs ({(num_FLOP_layers_attn_output_proj/total_FLOPs):.2%})",
        f"    |- Q K^T: {num_FLOP_layers_attn_q_k:,} FLOPs ({(num_FLOP_layers_attn_q_k/total_FLOPs):.2%})",
        f"    |- softmax(QK^T / sqrt(d_k)) V: {num_FLOP_layers_attn_weights_v:,} FLOPs ({(num_FLOP_layers_attn_weights_v/total_FLOPs):.2%})",
        f"  |- FFN: {num_FLOP_layers_ffn:,} FLOPs ({(num_FLOP_layers_ffn/total_FLOPs):.2%})",
        f"    |- w1: {num_FLOP_layers_ffn_w1:,} FLOPs ({(num_FLOP_layers_ffn_w1/total_FLOPs):.2%})",
        f"    |- w2: {num_FLOP_layers_ffn_w2:,} FLOPs ({(num_FLOP_layers_ffn_w2/total_FLOPs):.2%})",
        f"    |- w3: {num_FLOP_layers_ffn_w3:,} FLOPs ({(num_FLOP_layers_ffn_w3/total_FLOPs):.2%})",
        f"|- LM head: {num_FLOP_lm_head:,} FLOPs ({(num_FLOP_lm_head/total_FLOPs):.2%})",
    ]

    print("\n".join(lines))


# compute(FLOP, GPT2_XL, "GPT-2 XL")
# compute(FLOP, GPT2_small, "GPT-2 small")
# compute(FLOP, GPT2_medium, "GPT-2 medium")
# compute(FLOP, GPT2_large, "GPT-2 large")
# compute(FLOP, GPT2_XL_Long, "GPT-2 XL with long context length")
