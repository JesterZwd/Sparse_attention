import torch

from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.range_gate import choose_tiers, screen_four_tiers, tier_mask, tier_stats
from gqa_probe.round7_single import prepare_single
from gqa_probe.shared_refill import State


def test_exact_cutoffs_protection_and_tail():
    logits = torch.tensor([[[[0., -4., -6., -8.], [-10., -8., -4., 0.]]]])
    valid = torch.tensor([[[True, True, True, True], [True, True, True, False]]])
    protected = torch.tensor([[True, False]])
    state = State(torch.tensor([[0, 1]]), None, logits, valid, protected, valid[:, None])
    gaps, tails, total = tier_stats(state)
    want = logits.new_tensor([-4., -8., -10.]).exp()
    torch.testing.assert_close(tails[0, 0], torch.stack((want.sum(), want[1:].sum(),
                                                          want[1:].sum(), want[2])))
    assert total.item() > 1.
    tight = choose_tiers(tails, total, .01)
    loose = choose_tiers(tails, total, .02)
    assert tight.item() == 1
    assert loose.item() == 0
    tight_mask = tier_mask(state, gaps, tight)
    loose_mask = tier_mask(state, gaps, loose)
    assert (tight_mask | ~loose_mask).all()
    assert loose_mask[0, 0, 0].all()
    assert not loose_mask[0, 0, 1].any()


def sample_with_cold_pages(cold):
    tokens = 32737
    h = page_history(tokens, 32)
    q = torch.zeros((1, 4, 128))
    key = torch.zeros((1, 1, tokens, 128))
    ranks = ranked_pages(torch.zeros((1, h.pages)), h)
    state, order, h, cap, safe = prepare_single(q, key, ranks, tokens, 1., "q6", 1000, 68)
    if cold:
        optional = (~state.protected[0]).nonzero().flatten()
        state.logits[torch.isfinite(state.logits)] = -1.
        state.logits[:, :, optional[:25], :] = -4.
        state.logits[:, :, 0, 0] = 0.
    value = torch.zeros_like(key)
    dense = torch.zeros_like(q)
    return (state, order, h, cap, safe), ranks, value, dense


def test_relaxation_can_keep_all_k_and_counts_rescan():
    rows = screen_four_tiers(*sample_with_cold_pages(True))
    by_method = {r["policy"]: r for r in rows}
    strict = by_method["tiers2_full"]
    adapt = by_method["tiers234_full"]
    assert strict["fallback"]
    assert strict["retained_prefix"] < len(strict["initial_ids"])
    assert not adapt["fallback"]
    assert adapt["retained_prefix"] == len(strict["initial_ids"])
    assert strict["initial_ids"] == adapt["initial_ids"]
    assert adapt["accepted_limit"] == .03
    assert all(.02 < p <= .03 for p in adapt["lost_all_k"])
    assert adapt["mass_scans"] == 1
    assert adapt["mask_scans"] == adapt["budget_checks"] == 2
    assert adapt["retry_ck"] > 0
    for row in rows:
        assert row["read_bytes"] <= row["cap"]
        assert row["read_bytes"] == row["banked_bursts"] * 32


def test_identical_relaxation_cutoffs_skip_rescans_but_still_fall_back():
    rows = screen_four_tiers(*sample_with_cold_pages(False))
    by_method = {r["policy"]: r for r in rows}
    strict = by_method["tiers2_full"]
    adapt = by_method["tiers234_full"]
    assert strict["retained_prefix"] == adapt["retained_prefix"]
    assert strict["read_bytes"] == adapt["read_bytes"]
    assert adapt["fallback"]
    assert adapt["limits_tried"] == [.02, .03, .04]
    assert adapt["mask_scans"] == adapt["budget_checks"] == 1
    assert adapt["retry_ck"] > 0
    assert adapt["lost_all_k"][0] > .04
    assert adapt["lost_prefix"][0] == 0.


def test_report_pairs_states_and_exposes_adaptive_losses():
    from scripts.screen_range_gate import summarize_rows

    rows = screen_four_tiers(*sample_with_cold_pages(True))
    for r in rows:
        r.update(sample_id="calibr:0",step=0,layer=0,answer_start=True)
    report = summarize_rows(rows)
    by_method = {r["policy"]: r for r in report["rows"]}
    adaptive = by_method["tiers234_full"]
    assert adaptive["rescued_at_3pct"] == 1
    assert adaptive["full_k_drop_over_2pct_heads"] == 4
    assert adaptive["fallback_states"] == 0
    assert report["common_states"] == 1
    assert len(report["paired"]) == 4
