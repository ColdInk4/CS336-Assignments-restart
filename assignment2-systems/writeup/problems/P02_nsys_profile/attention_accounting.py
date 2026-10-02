from symbolica import S, Expression
from decimal import Decimal

# model
# vocab_size, context_length, num_layers, d_model, num_heads, d_ff
batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff = S(
    "B", "V", "T", "L", "d", "H", "d_ff"
)

small_256: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 256,
    num_layers: 12,
    d_model: 768,
    num_heads: 12,
    d_ff: 3072,
}
small_512: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 512,
    num_layers: 12,
    d_model: 768,
    num_heads: 12,
    d_ff: 3072,
}
small_2048: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 2048,
    num_layers: 12,
    d_model: 768,
    num_heads: 12,
    d_ff: 3072,
}
large_256: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 256,
    num_layers: 36,
    d_model: 1280,
    num_heads: 20,
    d_ff: 5120,
}
large_512: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 512,
    num_layers: 36,
    d_model: 1280,
    num_heads: 20,
    d_ff: 5120,
}
large_1024: dict[
    Expression, int | float | complex | Decimal | tuple[Decimal, Decimal]
] = {
    batch_size: 4,
    vocab_size: 10000,
    context_length: 1024,
    num_layers: 36,
    d_model: 1280,
    num_heads: 20,
    d_ff: 5120,
}


# FLOPS

## scaled_dot_product_attention
"""
d_k = d/H
 computing attention scores[matmul]
 Q K^T
 [batch_size num_heads sequence_length d_k] * [batch_size num_heads sequence_length d_k]
   -> [batch_size num_heads sequence_length sequence_length]
 batch_size * num_heads * 2 * sequence_length * sequence_length * d_k 
 = 2 * batch_size * sequence_length * sequence_length * d_model
"""
FLOP_block_attn_q_k = 2 * batch_size * context_length * context_length * d_model

"""
computing attention scores[element divide]
[batch_size num_heads sequence_length sequence_length]
"""
FLOP_block_attn_q_k_element_divide = (
    batch_size * num_heads * context_length * context_length
)

"""
mask fill
[batch_size num_heads sequence_length sequence_length]
"""
FLOP_block_attn_mask_fill = 0
"""
softmax
    max_values = in_features.amax(dim=dim, keepdim=True)
    shifted_inputs = in_features - max_values
    exp_inputs = shifted_inputs.exp()
    sum_exp_inputs = exp_inputs.sum(dim=dim, keepdim=True)
    return exp_inputs / sum_exp_inputs
amax: 0
exp: 0 (1)
[batch_size num_heads sequence_length sequence_length]
"""
FLOP_block_attn_softmax = 3 * batch_size * num_heads * context_length * context_length


"""
final matmul
(Q K^T) V
[num_heads sequence_length sequence_length] * [num_heads sequence_length d_v] 
 -> [num_heads sequence_length d_v]
num_heads * 2 * sequence_length * sequence_length * d_v 
 = 2 * sequence_length * sequence_length * d_model

"""
FLOP_block_attn_weights_v = 2 * batch_size * context_length * context_length * d_model

print(FLOP_block_attn_softmax)
print(FLOP_block_attn_q_k + FLOP_block_attn_weights_v)
print(FLOP_block_attn_softmax / (FLOP_block_attn_q_k + FLOP_block_attn_weights_v))
