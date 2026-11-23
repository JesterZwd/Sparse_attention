import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gqa_probe.chunked_prefill import PreallocatedAffineKV4Cache, prefill_chunked
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.quantization_round2 import AffineKV4Cache, prefill_for_quantization
from gqa_probe.ruler_inference import _copy_prefix, generate_from_prefill


def tiny_model():
    torch.manual_seed(13)
    config = Qwen3Config(vocab_size=24, hidden_size=128, intermediate_size=144,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=128,
                        sliding_window=None)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


def test_chunked_positions_and_causality_match_full_affine32():
    model = tiny_model()
    ids = list(range(1, 10))
    chunks = []
    def capture_inputs(_module, _args, kwargs):
        chunks.append((kwargs["cache_position"].tolist(), kwargs["position_ids"].tolist()))
    handle = model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        state = prefill_chunked(model, ids, chunk_size=4, max_new_tokens=4)
    finally:
        handle.remove()
    assert chunks == [(list(range(a, b)), [list(range(a, b))]) for a, b in ((0, 4), (4, 8), (8, 9))]

    full = prefill_for_quantization(model, ids, "affine32")
    torch.testing.assert_close(state.last_logits, full.last_logits, atol=1e-5, rtol=1e-4)
    assert state.last_logits.argmax().item() == full.last_logits.argmax().item()
    for actual, expected in zip(state.cache.key_cache + state.cache.value_cache,
                                full.cache.key_cache + full.cache.value_cache):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    # A token inside an offset chunk must see the previous prefix and itself.
    # Compare its deeper-layer K/V against a full run ending on that token.
    interior_prefix = prefill_for_quantization(model, ids[:6], "affine32")
    for buffered, ref in zip(state.cache.key_cache + state.cache.value_cache,
                             interior_prefix.cache.key_cache + interior_prefix.cache.value_cache):
        torch.testing.assert_close(buffered[..., :6, :], ref, atol=1e-5, rtol=1e-4)


def test_cache_appends_quantize_in_original_order_and_share_one_storage():
    torch.manual_seed(18)
    keys = torch.randn(1, 2, 9, 32, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    original = AffineKV4Cache()
    buffered = PreallocatedAffineKV4Cache(9)
    for first, last in ((0, 4), (4, 6)):
        args = keys[..., first:last, :], values[..., first:last, :], 0
        expected = original.update(*args)
        actual = buffered.update(*args)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    first_branch = _copy_prefix(buffered)
    ptrs = [x.untyped_storage().data_ptr() for x in actual]
    for tensor, ptr in zip(first_branch.update(keys[..., 6:8, :], values[..., 6:8, :], 0), ptrs):
        assert tensor.untyped_storage().data_ptr() == ptr
    assert first_branch.get_seq_length() == 8
    assert buffered.get_seq_length() == buffered._seen_tokens == 6
    second_branch = _copy_prefix(buffered)
    expected = original.update(keys[..., 8:9, :], values[..., 8:9, :], 0)
    for got, want in zip(second_branch.update(keys[..., 8:9, :], values[..., 8:9, :], 0), expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    with pytest.raises(ValueError, match="capacity"):
        second_branch.update(keys, values, 0)
    assert second_branch.get_seq_length() == second_branch._seen_tokens == 7


def test_dense_generation_reuses_the_prefill_without_mutating_visible_prefix():
    model = tiny_model()
    ids = list(range(1, 10))
    state = prefill_chunked(model, ids, chunk_size=4, max_new_tokens=4)
    prefix = [tensor.clone() for tensor in state.cache.key_cache + state.cache.value_cache]
    full = prefill_for_quantization(model, ids, "affine32")

    def generate(prefill):
        return generate_from_prefill(model, prefill, 4, [], OracleAttention("dense"))

    expected = generate(full)
    first, second = generate(state), generate(state)
    assert first.token_ids == second.token_ids == expected.token_ids
    assert first.attention_stats["decode_layer_calls"] == 6
    assert state.cache.get_seq_length() == state.cache._seen_tokens == len(ids)
    for current, saved in zip(state.cache.key_cache + state.cache.value_cache, prefix):
        torch.testing.assert_close(current, saved, atol=0, rtol=0)
