"""Grouped shared-KV attention and callback audit contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from gqa_probe.attention import attention_output, page_mass
from gqa_probe.capture import KV4Cache, TraceRecorder
from gqa_probe.oracle_attention import OracleAttention, grouped_logits, grouped_output, torch_page_mass
from gqa_probe.selection import protected_pages, select_pages


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_model(device="cpu", dtype=torch.float32):
    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_attention_shares_values_and_matches_literal_selected_softmax(device):
    q = torch.tensor([[[[1., 0.]], [[0., 1.]], [[1., 1.]], [[-1., 1.]]]], device=device)
    k = torch.tensor([[[[1., 0.], [0., 1.], [1., 1.]],
                       [[2., 0.], [0., 2.], [1., 1.]]]], device=device)
    v = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]],
                       [[10., 20.], [30., 40.], [50., 60.]]]], device=device)
    logits = grouped_logits(q, k, .5)
    torch.testing.assert_close(logits.cpu(), torch.tensor([[[.5, 0., .5], [0., .5, .5]],
                                                           [[1., 1., 1.], [-1., 1., 0.]]]))
    masks = torch.tensor([[True, False, True], [False, True, True]], device=device)
    out = grouped_output(logits, v, masks, page_size=1)
    for group, pages in enumerate([[0, 2], [1, 2]]):
        expected = attention_output(logits[group].cpu().numpy(), v[0, group].cpu().numpy(), pages, 1)
        np.testing.assert_allclose(out[group].cpu(), expected, atol=3e-6, rtol=1e-6)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("device", DEVICES)
def test_full_history_float64_mass_matches_numpy_and_partial_page(device):
    logits = torch.linspace(-12, 12, 2 * 3 * 65, device=device).reshape(2, 3, 65)
    mass = torch_page_mass(logits)
    assert mass.dtype == torch.float64
    assert mass.shape == (2, 3, 3)
    for group in range(2):
        np.testing.assert_allclose(mass[group].cpu(), page_mass(logits[group].cpu().numpy()),
                                   atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quantized", [False, True])
def test_tiny_qwen_dense_matches_existing_decode_reference(tmp_path, device, quantized):
    model = tiny_model(device, torch.bfloat16 if device == "cuda" else torch.float32)
    inputs = torch.tensor([[1, 2, 3, 4]], device=device)
    cache_a = KV4Cache(4) if quantized else DynamicCache()
    cache_b = KV4Cache(4) if quantized else DynamicCache()
    recorder = TraceRecorder(tmp_path, layers=[0, 1], group_size=4)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with torch.inference_mode(), recorder:
        reference_prefill = model(inputs, past_key_values=cache_a, use_cache=True).logits
        recorder.step = 0
        reference_decode = model(torch.tensor([[5]], device=device), past_key_values=cache_a, use_cache=True).logits
    with torch.inference_mode(), OracleAttention() as oracle:
        prefill = model(inputs, past_key_values=cache_b, use_cache=True).logits
        oracle.step = 0
        decode = model(torch.tensor([[5]], device=device), past_key_values=cache_b, use_cache=True).logits
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    torch.testing.assert_close(prefill, reference_prefill, atol=0, rtol=0)
    torch.testing.assert_close(decode, reference_decode, atol=1e-6, rtol=1e-5)
    assert oracle.stats()["decode_layer_calls"] == 2


@pytest.mark.parametrize("algorithm", ["A1", "SWAP"])
def test_full_selected_pages_degenerate_to_dense_model(algorithm):
    model = tiny_model()
    outputs = []
    for policy in [OracleAttention(), OracleAttention(algorithm, budget_fraction=1.)]:
        with torch.inference_mode(), policy:
            cache = KV4Cache(4)
            model(torch.tensor([[1, 2, 3, 4]]), past_key_values=cache, use_cache=True)
            policy.step = 0
            outputs.append(model(torch.tensor([[5]]), past_key_values=cache, use_cache=True).logits)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    stats = policy.stats()
    assert stats["selected_pages"] == 4  # One page × two groups × two layers.
    assert stats["swaps"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_shadow_callback_uses_same_qkv_and_all_gold_spans(device):
    torch.manual_seed(301)
    q = torch.randn(1, 4, 1, 8, device=device)
    k = torch.randn(1, 2, 321, 8, device=device)
    v = torch.randn_like(k)
    records = []
    oracle = OracleAttention(shadow_specs=[dict(name="a1", algorithm="A1", budget_fraction=.6,
                                               target=.8, delta=.01, max_swaps=1)],
                             sample_layers=[7], sample_steps=[0], callback=records.append,
                             gold_spans=[[0, 2], [40, 43], [100, 101]])
    oracle.step = 0
    module = SimpleNamespace(layer_idx=7, training=False)
    with oracle:
        output, _ = oracle.attention(module, q, k, v, None, scaling=.3)
        # Unselected layer/step must not emit an audit record.
        oracle.attention(SimpleNamespace(layer_idx=0, training=False), q, k, v, None, scaling=.3)
        oracle.step = 1
        oracle.attention(module, q, k, v, None, scaling=.3)
    assert len(records) == 2
    scores = grouped_logits(q, k, .3)
    dense = grouped_output(scores, v)
    torch.testing.assert_close(output, dense.reshape(1, 4, 1, 8).transpose(1, 2), atol=0, rtol=0)
    for group, record in enumerate(records):
        mass = page_mass(scores[group].cpu().numpy())
        selection = select_pages(mass, protected_pages(321), 6, .8, "A1")
        expected = attention_output(scores[group].cpu().numpy(), v[0, group].cpu().numpy(), selection.order)
        error = np.linalg.norm(expected - dense[group].cpu().numpy(), axis=-1)
        assert record["spec"] == "a1"
        assert (record["layer"], record["step"], record["kv_group"]) == (7, 0, group)
        assert record["pages"] == sorted(selection.order)
        np.testing.assert_allclose(record["head_coverage"], selection.coverage, atol=2e-14)
        np.testing.assert_allclose(record["head_output_absolute_errors"], error, atol=1e-6, rtol=1e-5)
        gold_pages = {0, 1, 3}
        chosen = set(selection.order)
        assert record["gold_page_recall"] == len(gold_pages & chosen) / 3
        assert record["gold_span_recall"] == sum(p in chosen for p in [0, 1, 3]) / 3
        assert record["gold_token_recall"] == sum(n for p, n in [(0, 2), (1, 3), (3, 1)] if p in chosen) / 6
        assert record["work"]["executed_av_multiply_adds"] == 2 * 321 * 8
        assert record["work"]["logical_selected_av_multiply_adds"] <= 2 * 6 * 32 * 8
    assert oracle.stats()["decode_layer_calls"] == 3
    assert oracle.stats()["candidate_pairs"] == 0
    assert oracle.shadow_work["group_spec_evaluations"] == 2


def test_sparse_stats_accumulate_without_records_and_gold_cannot_change_decisions():
    torch.manual_seed(601)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 321, 8)
    v = torch.randn_like(k)
    outputs = []
    for spans in [[], [[64, 128]]]:
        records = []
        policy = OracleAttention("SWAP", .6, callback=records.append, gold_spans=spans)
        policy.step = 0
        with policy:
            outputs.append(policy.attention(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=.3)[0])
            policy.attention(SimpleNamespace(layer_idx=1), q, k, v, None, scaling=.3)
        assert records == []
        stats = policy.stats()
        assert stats["decode_layer_calls"] == 2
        assert stats["kv_group_calls"] == 4
        assert stats["selected_pages"] == 24
        assert stats["executed_candidate_pairs"] >= stats["candidate_pairs"]
        assert stats["executed_av_multiply_adds"] == 2 * 2 * 2 * 321 * 8
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_context_restores_sdpa_on_error_and_rejects_shadow_on_sparse_policy():
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with pytest.raises(RuntimeError, match="test failure"):
        with OracleAttention():
            raise RuntimeError("test failure")
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    with pytest.raises(ValueError):
        OracleAttention("SWAP", shadow_specs=[dict(name="a1", algorithm="A1")])


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_aggregate_work_and_gold_union_stats(device):
    # Uniform scores: protected pages {0,6,7,8,9,10} fill the entire B=6.
    # Gold spans overlap on page 1; unique tokens count, spans each count once.
    q = torch.zeros(1, 4, 1, 8, device=device)
    k = torch.zeros(1, 2, 321, 8, device=device)
    v = torch.ones_like(k)
    oracle = OracleAttention("A1", .6, gold_spans=[[0, 2], [31, 34], [32, 35]])
    oracle.step = 0
    with oracle:
        oracle.attention(SimpleNamespace(layer_idx=0), q, k, v, None)
    stats = oracle.stats()
    assert stats["logical_selected_softmax_elements"] == 2 * 2 * (32 * 5 + 1)
    assert stats["executed_softmax_elements"] == 2 * 2 * 321
    assert stats["gold_token_hits"] == 2 * 3  # Tokens 0,1,31; all others page 1.
    assert stats["gold_token_total"] == 2 * 6  # {0,1,31,32,33,34}.
    assert stats["gold_page_hits"] == 2
    assert stats["gold_page_total"] == 4
    assert stats["gold_span_hits"] == 2
    assert stats["gold_span_total"] == 6
    assert stats["gold_span_recall"] == 1 / 3
    assert stats["gold_token_recall"] == .5
    assert stats["gold_page_recall"] == .5
    assert stats["all_gold_spans_group_hits"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_optional_float64_audit_matches_numpy_selected_and_dense(device):
    torch.manual_seed(99)
    q = torch.randn(1, 4, 1, 8, device=device)
    k = torch.randn(1, 2, 321, 8, device=device)
    v = torch.randn_like(k)
    records = []
    oracle = OracleAttention(shadow_specs=[dict(name="float64", algorithm="SWAP", budget_fraction=.6,
                                               audit_float64=True)], callback=records.append)
    oracle.step = 0
    with oracle:
        oracle.attention(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=.2)
    logits = grouped_logits(q, k, .2).cpu().numpy()
    for group, record in enumerate(records):
        values = v[0, group].cpu().numpy()
        dense = attention_output(logits[group], values)
        selected = attention_output(logits[group], values, record["pages"])
        absolute = np.linalg.norm(selected - dense, axis=-1)
        relative = absolute / (np.linalg.norm(dense, axis=-1) + 1e-12)
        np.testing.assert_allclose(record["head_output_absolute_errors_float64_audit"], absolute,
                                   atol=2e-15, rtol=2e-14)
        np.testing.assert_allclose(record["head_output_relative_errors_float64_audit"], relative,
                                   atol=2e-15, rtol=2e-14)
