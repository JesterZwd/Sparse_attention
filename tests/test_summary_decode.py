import pytest
import torch

from gqa_probe.fixed_k_v import candidate_qk, ranked_pages, vector_burst
from gqa_probe.lpddr5_banked import read_banked
from gqa_probe.page_read_budget import page_history
from gqa_probe.pagepack_budget import plan_pages, select_gate_mask, value_ledger
from gqa_probe.shared_refill import State
from gqa_probe.summary_decode import plan, read_trace, screen


@pytest.mark.parametrize("length, tail", [(63, 63), (64, 64), (65, 1)])
def test_p64_tail_never_issues_padding(length, tail):
    history = page_history(length, 64)
    ranks = ranked_pages(torch.zeros(1, history.pages), history)
    decode = plan(ranks, history, summary_bytes=144, budget_percent=None,
                  max_pages=history.pages)
    q = torch.zeros(1, 2, 128)
    kv = torch.zeros(1, 1, length, 128)
    result = screen(decode, q, kv, kv, 1., gold_spans=[(length - 1, length)])
    assert int(decode.full_mask().sum()) == length
    assert decode.dense_bytes == 2 * ((history.pages - 1) * vector_burst(64) + vector_burst(tail))
    assert int(result.ledger["k_bytes"]) == decode.dense_bytes // 2
    assert int(result.ledger["v_bytes"]) == decode.dense_bytes // 2
    assert int(result.ledger["v_records"]) == length
    assert int(result.evidence["v_gold_token_hits"]) == 1
    _assert_trace(decode, result)


def _assert_trace(decode, result, group=0):
    phases = read_trace(decode, result.final_mask, group=group)
    ledger = result.ledger
    assert [sum(size for _, size in phase) for phase in phases] == [
        int(ledger[k][group]) for k in ("summary_bytes", "k_bytes", "v_bytes")]
    ranges = sorted((a, a + n) for phase in phases for a, n in phase)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    assert read_banked(phases)["bursts"] * 32 == int(ledger["total"][group])


def test_p32_round9_histogram_fallback_and_ledger_match():
    history = page_history(32769, 32)
    ranked = ranked_pages(torch.zeros(2, history.pages), history)
    counts = torch.full_like(ranked, history.page_size)
    counts[:, -1] = history.tail_size
    old = plan_pages(ranked, history, counts, counts, summary_bytes=144,
                     budget_percent=5, gate="hist64", v_reserve=.65)
    decode = plan(ranked, history, summary_bytes=144, budget_percent=5,
                  v_reserve=.65)
    for name in ("ids", "active", "count", "safe_count", "protected"):
        assert torch.equal(getattr(decode, name), getattr(old, name))
    assert decode.cap == old.cap
    q = torch.zeros(2, 4, 128)
    q[1, :, 0] = 1.
    k = torch.zeros(1, 2, history.tokens, 128)
    k[0, 1, 4, 0] = 40.
    logits, pos, valid, pins = candidate_qk(q, k, old.ids, history, 1., old.active)
    expected = select_gate_mask(old, State(old.ids, pos, logits, valid, pins, valid[:, None]))
    result = screen(decode, q, k, k, 1., gold_spans=[(32, 35)])
    assert result.fallback.tolist() == [True, False]
    assert torch.equal(result.final_mask, expected.mask)
    assert torch.equal(result.prefix, expected.prefix)
    torch.testing.assert_close(result.loss_read_k, expected.dropped_all)
    torch.testing.assert_close(result.loss_prefix, expected.dropped_prefix)
    for name, value in value_ledger(old, expected.mask).items():
        assert torch.equal(result.ledger[name], value), name
    assert result.evidence["k_page_hits"].tolist() == [1, 1]
    assert result.evidence["v_gold_token_hits"].tolist() == [3, 0]
    assert ((result.ledger["total"] <= decode.cap).all())
    _assert_trace(decode, result, group=0)
    _assert_trace(decode, result, group=1)


def test_p64_logical_prefix_respects_protection_and_paid_metadata():
    history = page_history(32769, 64)
    rank = ranked_pages(torch.zeros(1, history.pages), history)
    forced = len(history.protected)
    assert history.protected == (0, 510, 511, 512)
    assert history.expanded_tokens == 33
    decode = plan(rank, history, summary_bytes=144, budget_percent=5,
                  max_tokens=history.protected_kv.total // 152 + 65)
    assert decode.count.tolist() == [forced + 1]
    assert int(decode.full_mask().sum()) == 193 + 64
    assert decode.summary_bytes == 73888
    q = torch.zeros(1, 1, 128)
    kv = torch.zeros(1, 1, history.tokens, 128)
    result = screen(decode, q, kv, kv, 1.)
    assert int(result.ledger["over_bytes"]) == 0
    _assert_trace(decode, result)
    with pytest.raises((RuntimeError, ValueError), match="protect"):
        plan(rank, history, max_tokens=192)
