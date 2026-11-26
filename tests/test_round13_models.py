from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.round13_models import model_geometry, prefill_generic
from gqa_probe.ruler_inference import generate_from_prefill, prefill_model


def tiny_7to1():
    torch.manual_seed(13)
    config = Qwen2Config(vocab_size=24, hidden_size=224, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=7,
                        num_key_value_heads=1, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval()


def test_qwen2_geometry_uses_config_fallback_head_dimension():
    geometry = model_geometry(tiny_7to1().config)
    assert geometry["num_attention_heads"] == 7
    assert geometry["num_key_value_heads"] == 1
    assert geometry["query_heads_per_kv_head"] == 7
    assert geometry["head_dim"] == 32
    assert geometry["max_position_embeddings"] == 64


@pytest.mark.parametrize("bits", (4, 16))
def test_qwen2_chunked_prefill_offsets_cache_and_continuations(bits):
    torch.set_num_threads(2)
    model = tiny_7to1()
    ids = list(range(1, 10))
    offsets = []

    def record(_module, _args, kwargs):
        offsets.append((kwargs["cache_position"].tolist(), kwargs["position_ids"].tolist()))

    hook = model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        state = prefill_generic(model, ids, chunk_size=4, max_new_tokens=4, kv_bits=bits)
    finally:
        hook.remove()
    assert offsets == [(list(range(a, b)), [list(range(a, b))]) for a, b in ((0, 4), (4, 8), (8, 9))]
    full = prefill_model(model, ids, 16) if bits == 16 else prefill_for_quantization(model, ids, "affine32")
    torch.testing.assert_close(state.last_logits, full.last_logits, atol=1e-5, rtol=1e-4)
    cached = state.cache.key_cache + state.cache.value_cache
    expected = full.cache.key_cache + full.cache.value_cache
    for actual, reference in zip(cached, expected):
        assert actual.shape == (1, 1, 9, 32)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    first, second, reference = (generate_from_prefill(model, s, 4, [], OracleAttention("dense"))
                                for s in (state, state, full))
    assert first.token_ids == second.token_ids == reference.token_ids
    assert first.attention_stats["kv_group_calls"] == 6
    assert state.cache.get_seq_length() == len(ids)
    for actual, reference in zip(cached, expected):
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


def test_context_limit_includes_generation_reservation():
    with pytest.raises(ValueError, match="context limit"):
        prefill_generic(tiny_7to1(), [3] * 60, chunk_size=16, max_new_tokens=5)


def test_qwen2_7to1_sparse_workpoints_charge_each_query_head():
    from gqa_probe.round13_comrad import SPARSE
    from scripts.run_round10 import prefix_layers
    from scripts.run_round11 import WorkpointAttention

    torch.manual_seed(13)
    query = torch.randn(1, 28, 1, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 32593, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    state = SimpleNamespace(cache=SimpleNamespace(key_cache=[key[..., :-1, :]]))
    masks = {"mr96_score": {0: torch.arange(96).expand(4, -1)}}
    for policy in SPARSE:
        attention = WorkpointAttention(policy, prefix_layers(state, policy, masks), masks, [])
        attention.step = 0
        output, _ = attention.attention(SimpleNamespace(layer_idx=0), query, key, value, None)
        stats = attention.stats()
        assert output.shape == (1, 1, 28, 128) and torch.isfinite(output).all()
        assert stats["kv_group_calls"] == 4 and stats["query_head_calls"] == 28
        assert stats["qk_mac"] == stats["k_records"] * 7 * 128
        assert stats["read"] <= stats["cap_bytes"] and stats["over_budget_group_steps"] == 0
