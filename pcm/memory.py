"""Memory arithmetic from a model configuration (specification, section 4.2)."""
from __future__ import annotations

GIB = 1024**3
MIB = 1024**2
KIB = 1024


def kv_bytes_per_token(
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    bytes_per_element: float = 2.0,
) -> float:
    """Bytes of KV cache per token: keys and values, every layer, every KV head."""
    return 2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_element


def kv_bytes(tokens: int, per_token: float) -> float:
    return tokens * per_token


def qwen3_param_count(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_bias: bool = False,
    tie_word_embeddings: bool = True,
) -> dict:
    """Parameter count of a Qwen3 decoder rebuilt from its configuration.

    Layer layout: q/k/v/o projections, RMSNorm on queries and keys (head_dim
    each), a gated MLP (gate, up, down; no bias), two RMSNorms per layer, and a
    final RMSNorm.
    """
    q_out = num_attention_heads * head_dim
    kv_out = num_key_value_heads * head_dim
    attn = hidden_size * q_out + 2 * hidden_size * kv_out + q_out * hidden_size
    if attention_bias:
        attn += q_out + 2 * kv_out + hidden_size
    qk_norm = 2 * head_dim
    mlp = 3 * hidden_size * intermediate_size
    norms = 2 * hidden_size
    per_layer = attn + qk_norm + mlp + norms
    embedding = vocab_size * hidden_size
    non_embedding = num_hidden_layers * per_layer + hidden_size
    lm_head = 0 if tie_word_embeddings else vocab_size * hidden_size
    return {
        "embedding": embedding,
        "non_embedding": non_embedding,
        "lm_head_untied": lm_head,
        "total": embedding + non_embedding + lm_head,
    }
