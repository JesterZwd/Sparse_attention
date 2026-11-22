import pytest
import torch

from gqa_probe.fixed_k_v import probabilities, ranked_pages
from gqa_probe.round6_attention import physical_ledger
from gqa_probe.round7_single import run_single, single_quota
from gqa_probe.shared_refill import RefillPolicy, run_shared


@pytest.mark.parametrize("tokens,method,budget", [(32736, "q6", 1000), (32737, "q6", 1000), (32737, "q4", 500)])
def test_flat_logits_use_prefix_and_charge_all_read_k(tokens, method, budget):
    history, cap, count, prefix = single_quota(tokens, method, budget, 68)
    q = torch.zeros((1, 4, 128))
    key = torch.zeros((1, 1, tokens, 128))
    ranks = ranked_pages(torch.zeros((1, history.pages)), history)
    out = run_single(q, key, ranks, tokens, 128**-.5, method=method, budget_bp=budget)
    k = out.final
    ledger = physical_ledger(k, history, method, torch.ones(1, dtype=torch.bool))
    assert out.ceiling == cap
    assert k.valid.any(-1).sum() == count
    assert len(set(k.ids[0].tolist())) == count
    assert out.fallback.item()
    assert out.prefix_pages.item() == prefix < count
    assert ledger["total"].item() <= cap
    assert not (k.mask & ~torch.isin(k.ids, ranks[:, :prefix])[:, None, :, None]).any()
    assert (k.mask | ~(k.valid & k.protected[..., None])[:, None]).all()
    assert out.initial_over_bytes.item() > 0
    assert out.lost_prefix.max() <= .020003
    assert out.lost_all.max() > .02
    torch.testing.assert_close(out.lost_all, (probabilities(k.logits) * ~k.mask).sum((2, 3)))


def test_no_fallback_keeps_full_k_probability_denominator():
    tokens = 32737
    history, _, count, _ = single_quota(tokens, "q6", 1000, 68)
    ranks = ranked_pages(torch.zeros((1, history.pages)), history)
    q = torch.ones((1, 4, 128))
    key = torch.zeros((1, 1, tokens, 128))
    key[:, :, 33] = 4
    out = run_single(q, key, ranks, tokens, 128**-.5)
    assert not out.fallback.item()
    assert out.prefix_pages.item() == count
    torch.testing.assert_close(out.lost_prefix, out.lost_all)


def test_dynamic_cap_keeps_budget_and_accounts_added_pages():
    tokens = 32737
    history, _, _, _ = single_quota(tokens, "q6", 1000, 68)
    ranks = ranked_pages(torch.zeros((1, history.pages)), history)
    q = torch.ones((1, 4, 128))
    key = torch.zeros((1, 1, tokens, 128))
    key[:, :, 33] = 4
    out = run_shared(RefillPolicy("sparta", .02), q, key, ranks, tokens, 128**-.5,
                     audit=True, max_pages=68)
    ledger = physical_ledger(out.final, history, "q6", torch.ones(1, dtype=torch.bool))
    assert ledger["k_pages"].item() == 68
    assert ledger["total"].item() <= out.ceiling


def test_all_initial_gates_use_exact_top_p_on_budget_fallback():
    t = 32737
    history, _, _, _ = single_quota(t, "q6", 1000, 68)
    ranks = ranked_pages(torch.zeros((1, history.pages)), history)
    q = torch.zeros((1, 4, 128))
    key = torch.zeros((1, 1, t, 128))
    ref = run_single(q,key,ranks,t,128**-.5,kind="exact")
    for gate,threshold in (("hist64",0.),("checked",8.),("blasst",.001)):
        student = run_single(q,key,ranks,t,128**-.5,kind=gate,threshold=threshold)
        assert student.fallback.item()
        assert torch.equal(student.final.mask, ref.final.mask)
