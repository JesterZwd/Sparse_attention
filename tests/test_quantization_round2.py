"""Prespecified quantization repairs and shared-prefix contracts."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gqa_probe.capture import KV4Cache, quantize_dequantize
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.quantization_round2 import (
    AffineKV4Cache, affine_quantize_dequantize, make_cache,
    prefill_for_quantization, quantization_spec,
)
from gqa_probe.ruler_inference import generate_from_prefill


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_affine_literals_round_before_integer_zero_point(device):
    x = torch.tensor([[-7., -2.5, 2.5, 8.]], device=device)
    expected = torch.tensor([[-7., -2., 2., 8.]], device=device)
    torch.testing.assert_close(affine_quantize_dequantize(x, 4), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_affine_encodes_fp16_scale_before_codes_and_dequantization(device):
    x = torch.tensor([[-2., -1., 0., 3.]], device=device)
    expected = torch.tensor([[-1.99951171875, -.999755859375, 0., 2.999267578125]], device=device)
    torch.testing.assert_close(affine_quantize_dequantize(x, 4), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_affine_zero_and_constant_groups_include_zero_and_keep_positive_scale(device):
    x = torch.tensor([[0., 0., 0., 0.], [15., 15., 15., 15.],
                      [-15., -15., -15., -15.], [1e-10, 0., -1e-10, 0.]], device=device)
    expected = torch.tensor([[0., 0., 0., 0.], [15., 15., 15., 15.],
                             [-15., -15., -15., -15.], [0., 0., 0., 0.]], device=device)
    torch.testing.assert_close(affine_quantize_dequantize(x, 4), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_affine_cache_quantizes_distinct_keys_and_values_on_every_append(device):
    cache = AffineKV4Cache(group_size=4)
    key = torch.tensor([[[[-7., -2.5, 2.5, 8.]]]], device=device)
    value = torch.tensor([[[[-2., -1., 0., 3.]]]], device=device)
    k1, v1 = cache.update(key, value, 0)
    torch.testing.assert_close(k1, torch.tensor([[[[-7., -2., 2., 8.]]]], device=device), atol=0, rtol=0)
    torch.testing.assert_close(v1, torch.tensor([[[[-1.99951171875, -.999755859375, 0., 2.999267578125]]]], device=device), atol=0, rtol=0)
    k2, v2 = cache.update(torch.zeros_like(key), torch.ones_like(value) * 15, 0)
    torch.testing.assert_close(k2[..., :1, :], k1, atol=0, rtol=0)
    torch.testing.assert_close(v2[..., :1, :], v1, atol=0, rtol=0)
    torch.testing.assert_close(k2[..., 1:, :], torch.zeros_like(key), atol=0, rtol=0)
    torch.testing.assert_close(v2[..., 1:, :], torch.ones_like(value) * 15, atol=0, rtol=0)


@pytest.mark.parametrize("name,cache_type,group_size,bits", [
    ("affine32", AffineKV4Cache, 32, 4.75), ("symmetric16", KV4Cache, 16, 5.),
])
def test_spec_builds_only_prespecified_cache_and_theoretical_layout(name, cache_type, group_size, bits):
    cache = make_cache(name)
    assert type(cache) is cache_type
    assert cache.group_size == group_size
    spec = quantization_spec(name)
    assert spec["theoretical_bits_per_element"] == bits
    assert spec["actual_cache_storage"] == "dequantized_model_dtype"
    assert spec["residual_unquantized_tokens"] == 0
    spec["group_size"] = 99
    assert quantization_spec(name)["group_size"] == group_size
    with pytest.raises(ValueError):
        make_cache(spec)


@pytest.mark.parametrize("device", DEVICES)
def test_symmetric16_factory_uses_existing_quantization_for_both_kv(device):
    x = torch.linspace(-10, 8, 32, device=device).reshape(1, 1, 1, 32)
    y = x.flip(-1) * .4
    cache = make_cache("symmetric16")
    k, v = cache.update(x, y, 0)
    torch.testing.assert_close(k, quantize_dequantize(x, 16), atol=0, rtol=0)
    torch.testing.assert_close(v, quantize_dequantize(y, 16), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", ["affine32", "symmetric16"])
def test_tiny_qwen_prefill_and_repeated_decode_preserve_original_prefix(device, spec):
    torch.manual_seed(203)
    config = Qwen3Config(vocab_size=24, hidden_size=128, intermediate_size=144,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(device=device,
                                       dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    state = prefill_for_quantization(model, [1, 2, 3], spec)
    key_list = state.cache.key_cache
    value_list = state.cache.value_cache
    keys = [t.clone() for t in key_list]
    values = [t.clone() for t in value_list]
    first = generate_from_prefill(model, state, 4, [], OracleAttention())
    second = generate_from_prefill(model, state, 4, [], OracleAttention())
    assert first.token_ids == second.token_ids
    assert first.attention_stats["decode_layer_calls"] == second.attention_stats["decode_layer_calls"] == 6
    assert state.input_tokens == state.cache.get_seq_length() == state.cache._seen_tokens == 3
    assert state.cache.key_cache is key_list
    assert state.cache.value_cache is value_list
    assert state.last_logits.shape == (1, 24)
    assert not state.last_logits.requires_grad
    for actual, expected in zip(key_list, keys):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(value_list, values):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("spec", ["symmetric32", "affine16", {}, {"name": "affine32", "bits": 8}])
def test_undeclared_quantization_specs_rejected(spec):
    with pytest.raises(ValueError):
        make_cache(spec)


def test_engineering_rule_requires_exact_32_and_prefers_lower_theoretical_bits():
    from scripts.run_quantization_engineering import choose_quantization

    def records(name, correct):
        return [dict(spec=name, sample_id=f"row{i}", complete_correct=i < correct) for i in range(32)]
    assert choose_quantization(records("affine32", 32) + records("symmetric16", 32))["selected_spec"] == "affine32"
    assert choose_quantization(records("affine32", 31) + records("symmetric16", 32))["selected_spec"] == "symmetric16"
    result = choose_quantization(records("affine32", 31) + records("symmetric16", 31))
    assert result["selected_spec"] is None
    assert result["decision"] == "stop_no_eligible_variant"
    assert choose_quantization(records("affine32", 32))["decision"] == "incomplete"
