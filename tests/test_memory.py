"""Memory arithmetic of the specification, sections 1.5 and 4.2, checked by hand-computed values."""
from pcm.memory import GIB, KIB, kv_bytes, kv_bytes_per_token, qwen3_param_count


def test_kv_bytes_per_token_bf16():
    assert kv_bytes_per_token(28, 8, 128) == 114_688
    assert kv_bytes_per_token(28, 8, 128) / KIB == 112


def test_kv_table_rows():
    per_token = kv_bytes_per_token(28, 8, 128)
    assert kv_bytes(32_768, per_token) / GIB == 3.5
    assert kv_bytes(131_072, per_token) / GIB == 14
    assert kv_bytes(262_144, per_token) / GIB == 28
    assert kv_bytes(1_048_576, per_token) / GIB == 112


def test_quantized_store_sizes():
    int4 = kv_bytes_per_token(28, 8, 128, bytes_per_element=0.5)
    assert kv_bytes(262_144, int4) / GIB == 7


def test_param_counts_match_model_cards():
    small = qwen3_param_count(151_936, 1024, 3072, 28, 16, 8, 128)
    large = qwen3_param_count(151_936, 2048, 6144, 28, 16, 8, 128)
    assert small["non_embedding"] == 440_467_456
    assert small["total"] == 596_049_920
    assert large["non_embedding"] == 1_409_410_048
    assert large["total"] == 1_720_574_976
    assert round(large["total"] * 2 / GIB, 1) == 3.2
    assert round(small["total"] * 2 / GIB, 1) == 1.1


def test_untied_head_adds_one_embedding_matrix():
    tied = qwen3_param_count(1000, 64, 128, 2, 4, 2, 16)
    untied = qwen3_param_count(1000, 64, 128, 2, 4, 2, 16, tie_word_embeddings=False)
    assert untied["total"] - tied["total"] == 1000 * 64
