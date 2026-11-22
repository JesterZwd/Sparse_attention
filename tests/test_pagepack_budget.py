"""Physical allocation checks for the Round 8 fixed and ragged layouts."""
import pytest
import torch

from gqa_probe.fixed_k_v import ranked_pages, vector_burst
from gqa_probe.page_read_budget import page_history
from gqa_probe.pagepack_budget import plan_pages, read_trace, select_gate_mask, value_ledger
from gqa_probe.shared_refill import State


def setup_counts(groups=1):
    h = page_history(32751, 32)
    raw = torch.full((groups, h.pages), 32, dtype=torch.long)
    raw[:, -1] = h.tail_size
    ranks = ranked_pages(torch.zeros_like(raw, dtype=torch.float), h)
    return h, raw, ranks


def test_full_budget_uses_longest_prefix_and_savings_buy_more_pages():
    h, raw, ranks = setup_counts()
    reps = raw.clone()
    reps.fill_(16)
    reps[:, list(h.protected)] = raw[:, list(h.protected)]
    for counts in (raw, reps):
        plan = plan_pages(ranks, h, counts, counts, gate="full")
        used = value_ledger(plan, plan.full_mask())["total"].item()
        assert used <= plan.cap
        next_page = ranks[0, plan.count.item()]
        assert used + 2 * vector_burst(counts[0, next_page].item()) > plan.cap
        assert plan.safe_count.item() == plan.count.item()
    raw_plan = plan_pages(ranks, h, raw, raw)
    assert plan.count.item() > raw_plan.count.item()


def test_reserved_v_is_applied_to_the_same_rank_order_for_all_allocations():
    h, raw, ranks = setup_counts()
    plans = [plan_pages(ranks, h, raw, raw, gate="hist64", v_reserve=f) for f in (0, .5, 1)]
    assert plans[0].count.item() > plans[1].count.item() > plans[2].count.item()
    torch.testing.assert_close(plans[0].ids[:, :plans[2].ids.shape[1]], plans[2].ids)
    assert plans[0].safe_count.item() < plans[2].safe_count.item()


def test_gqa_v_union_coalesces_scale_and_zero_point_bursts():
    h, raw, ranks = setup_counts()
    plan = plan_pages(ranks, h, raw, raw, max_pages=len(h.protected) + 1)
    mask = torch.zeros((1, 4, plan.ids.shape[1], 32), dtype=torch.bool)
    mask[0, 0, -1, [0, 3, 4]] = True
    mask[0, 1, -1, [3, 8]] = True
    ledger = value_ledger(plan, mask)
    assert ledger["v_records"].item() == 4
    assert ledger["v_code_bytes"].item() == 4 * 64
    assert ledger["v_scale_bytes"].item() == 3 * 32
    assert ledger["v_zero_point_bytes"].item() == 2 * 32
    assert ledger["v_bytes"].item() == 4 * 64 + 5 * 32


def test_ragged_directory_labels_and_v_gather_match_physical_trace():
    h, raw, ranks = setup_counts(2)
    reps = raw.clone()
    reps[:, ::2] = 16
    reps[:, 1::2] = 8
    reps[:, list(h.protected)] = raw[:, list(h.protected)]
    reps[1, 10:20] = 24
    plan = plan_pages(ranks, h, reps, reps, gate="hist64", v_reserve=.5, ragged=True)
    mask = plan.full_mask(plan.safe_count)
    mask[:, len(h.protected):, 2::3] = False
    ledger = value_ledger(plan, mask)
    for i, expected in enumerate(ledger["total"].tolist()):
        phases = read_trace(plan, mask, group=i)
        meta, directory, key, value = (sum(size for _, size in phase) for phase in phases)
        assert meta == ledger["summary_bytes"][i].item()
        assert directory >= 16 * h.pages
        assert key == ledger["k_bytes"][i].item() + ledger["tag_bytes"][i].item()
        assert value == ledger["v_bytes"][i].item()
        assert sum((meta, directory, key, value)) == expected <= plan.cap
        runs = sorted((start, start + size) for phase in phases for start, size in phase)
        assert all(hi <= nxt for (_, hi), (nxt, _) in zip(runs, runs[1:]))


def test_hist64_fallback_preserves_protection_and_charges_all_discarded_k():
    h, raw, ranks = setup_counts()
    plan = plan_pages(ranks, h, raw, raw, gate="hist64", v_reserve=0.)
    valid = plan.full_mask()
    logits = torch.zeros((1, 4, plan.ids.shape[1], 32)).masked_fill(~valid[:, None], -torch.inf)
    state = State(plan.ids, torch.zeros_like(valid, dtype=torch.long), logits, valid, plan.protected, valid[:, None])
    outcome = select_gate_mask(plan, state, fallback="hist64")
    used = value_ledger(plan, outcome.mask)
    assert outcome.fallback.item()
    assert outcome.prefix.item() < plan.count.item()
    assert used["total"].item() <= plan.cap
    assert used["k_records"].item() == plan.k_counts.sum().item()
    assert (outcome.mask | ~(valid & plan.protected[..., None])[:, None]).all()
    assert outcome.dropped_prefix.max().item() <= .020003
    assert outcome.dropped_all.min().item() > .02


def test_protected_pages_must_be_first_and_remain_raw():
    h, raw, ranks = setup_counts()
    compressed = raw.clone()
    compressed[0, h.protected[-1]] = 1
    with pytest.raises(RuntimeError, match="protected"):
        plan_pages(ranks, h, raw, compressed)
    wrong = torch.arange(h.pages).view(1, -1)
    with pytest.raises(RuntimeError, match="protected"):
        plan_pages(wrong, h, raw, raw)


def test_variable_stride_requires_ragged_metadata_before_qk():
    h, raw, ranks = setup_counts()
    variable = raw.clone()
    variable[:, h.protected[0]+1] = 16
    with pytest.raises(RuntimeError, match="fixed stride"):
        plan_pages(ranks, h, variable, variable)


def test_duplicate_ranked_page_cannot_be_billed_twice():
    h, raw, ranks = setup_counts()
    ranks[:, len(h.protected)] = ranks[:, len(h.protected)+1]
    with pytest.raises(RuntimeError, match="unique"):
        plan_pages(ranks,h,raw,raw)


def test_integer_prefix_matches_uniform_group_prefix():
    h, raw, ranks = setup_counts(2)
    plan = plan_pages(ranks,h,raw,raw)
    n = len(h.protected)
    torch.testing.assert_close(plan.full_mask(n),plan.full_mask(torch.tensor([n,n])))
