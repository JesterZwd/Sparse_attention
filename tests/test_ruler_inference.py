"""Greedy token counting, EOS, and immutable shared-prefix integration tests."""

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from gqa_probe.capture import KV4Cache
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.ruler_inference import prefill_model, generate_from_prefill


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_model(device="cpu", zero_head=False):
    torch.manual_seed(137)
    config = Qwen3Config(vocab_size=24, hidden_size=128, intermediate_size=144,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=256)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(device=device,
                                       dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    if zero_head:
        with torch.no_grad():
            model.lm_head.weight.zero_()
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_bits", [4, 16])
def test_first_prefill_token_eos_never_decodes(device, kv_bits):
    model = tiny_model(device, zero_head=True)
    state = prefill_model(model, [1, 2, 3], kv_bits=kv_bits)
    result = generate_from_prefill(model, state, 128, [0], OracleAttention())
    assert result.token_ids == [0]
    assert result.attention_stats["decode_layer_calls"] == 0
    assert result.elapsed_seconds >= 0
    assert state.input_tokens == 3
    assert state.last_logits.shape == (1, 24)
    assert not state.last_logits.requires_grad
    assert state.elapsed_seconds >= 0


@pytest.mark.parametrize("device", DEVICES)
def test_one_new_token_limit_counts_prefill_and_never_decodes(device):
    model = tiny_model(device)
    state = prefill_model(model, torch.tensor([[1, 2, 3]], device=device), 16)
    expected = state.last_logits.argmax(-1).item()
    result = generate_from_prefill(model, state, 1, [], OracleAttention())
    assert result.token_ids == [expected]
    assert result.attention_stats["decode_layer_calls"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_new_token_limit_includes_prefill_and_sets_zero_based_decode_steps(device):
    model = tiny_model(device, zero_head=True)
    state = prefill_model(model, [1, 2, 3], 4)
    attention = OracleAttention()
    steps = []
    def observe_step(_module, _args, _kwargs):
        steps.append(attention.step)
    hook = model.register_forward_pre_hook(observe_step, with_kwargs=True)
    try:
        result = generate_from_prefill(model, state, 4, [], attention)
    finally:
        hook.remove()
    assert result.token_ids == [0, 0, 0, 0]
    assert steps == [0, 1, 2]
    assert result.attention_stats["decode_layer_calls"] == 6


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_bits", [4, 16])
def test_repeated_generations_share_immutable_prefix_and_own_cache_lists(device, kv_bits):
    model = tiny_model(device)
    state = prefill_model(model, [1, 2, 3, 4], kv_bits)
    original_keys, original_values = state.cache.key_cache, state.cache.value_cache
    key_ids = [id(tensor) for tensor in original_keys]
    value_ids = [id(tensor) for tensor in original_values]
    keys = [tensor.clone() for tensor in original_keys]
    values = [tensor.clone() for tensor in original_values]
    original_seen_tokens = state.cache._seen_tokens
    observed = []
    def observe_cache(_module, _args, kwargs):
        cache = kwargs["past_key_values"]
        observed.append((cache, cache.key_cache, cache.value_cache,
                         [id(tensor) for tensor in cache.key_cache],
                         [id(tensor) for tensor in cache.value_cache]))
    hook = model.register_forward_pre_hook(observe_cache, with_kwargs=True)
    try:
        first = generate_from_prefill(model, state, 4, [], OracleAttention())
        second = generate_from_prefill(model, state, 4, [], OracleAttention("SWAP", 1.))
    finally:
        hook.remove()
    assert first.token_ids == second.token_ids
    assert len(observed) == 6
    for index in [0, 3]:
        copied, key_list, value_list, shared_key_ids, shared_value_ids = observed[index]
        assert copied is not state.cache
        assert type(copied) is type(state.cache)
        assert key_list is not original_keys
        assert value_list is not original_values
        assert shared_key_ids == key_ids
        assert shared_value_ids == value_ids
    assert observed[0][0] is not observed[3][0]
    assert state.cache.key_cache is original_keys
    assert state.cache.value_cache is original_values
    assert state.cache.get_seq_length() == 4
    assert state.cache._seen_tokens == original_seen_tokens == 4
    for actual, original, tensor_id in zip(original_keys, keys, key_ids):
        assert id(actual) == tensor_id
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
    for actual, original, tensor_id in zip(original_values, values, value_ids):
        assert id(actual) == tensor_id
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_prefill_bit_paths_use_sdpa_and_expected_cache_precision(device):
    model = tiny_model(device)
    inputs = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        expected = model(inputs, past_key_values=DynamicCache(), use_cache=True,
                         logits_to_keep=1).logits[:, -1]
    baseline = prefill_model(model, inputs, 16)
    quantized = prefill_model(model, inputs, 4)
    assert type(baseline.cache) is DynamicCache
    assert type(quantized.cache) is KV4Cache
    torch.testing.assert_close(baseline.last_logits, expected, atol=0, rtol=0)
    assert not torch.equal(baseline.cache.key_cache[0], quantized.cache.key_cache[0])
    assert baseline.cache.get_seq_length() == quantized.cache.get_seq_length() == 4
    assert baseline.cache.key_cache[0].dtype == quantized.cache.key_cache[0].dtype


@pytest.mark.parametrize("kv_bits", [0, 8, 32, 4.5])
def test_invalid_cache_precision_rejected(kv_bits):
    with pytest.raises(ValueError):
        prefill_model(tiny_model(), [1, 2, 3], kv_bits)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_new_token_limit_rejected(limit):
    model = tiny_model()
    state = prefill_model(model, [1, 2, 3], 16)
    with pytest.raises(ValueError):
        generate_from_prefill(model, state, limit, [], OracleAttention())


def test_engineering_summary_pairs_by_prompt_and_preserves_task_counts():
    from scripts.run_ruler_engineering import summarize_engineering

    rows = [
        dict(sample_id="a", task="one", bitmode="BF16", official_score=1., complete_correct=True,
             prefill_seconds=2., decode_seconds=1.),
        dict(sample_id="a", task="one", bitmode="KV4", official_score=.5, complete_correct=False,
             prefill_seconds=2., decode_seconds=2.),
        dict(sample_id="b", task="one", bitmode="BF16", official_score=0., complete_correct=False,
             prefill_seconds=2., decode_seconds=1.),
        dict(sample_id="b", task="one", bitmode="KV4", official_score=1., complete_correct=True,
             prefill_seconds=2., decode_seconds=2.),
        dict(sample_id="c", task="two", bitmode="BF16", official_score=1., complete_correct=True,
             prefill_seconds=3., decode_seconds=1.),
    ]
    result = summarize_engineering(rows)
    assert result["completed_generations"] == 5
    assert result["paired_prompts"] == 2
    assert result["per_task"]["one"]["BF16"]["count"] == 2
    assert result["per_task"]["one"]["KV4"]["official_score"] == .75
    assert result["paired"]["official_score_delta_kv4_minus_bf16"] == .25
    assert result["paired"]["bf16_correct_kv4_wrong"] == 1
    assert result["paired"]["bf16_wrong_kv4_correct"] == 1
    assert result["paired"]["complete_correct_delta_kv4_minus_bf16"] == 0.
    assert result["per_task"]["two"]["BF16"]["count"] == 1


def test_engineering_summary_rejects_duplicate_prompt_method_records():
    from scripts.run_ruler_engineering import summarize_engineering

    row = dict(sample_id="a", task="one", bitmode="BF16", official_score=1., complete_correct=True,
               prefill_seconds=2., decode_seconds=1.)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_engineering([row, row])
