import torch

from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.refill_replay import (
    INDEX_BASE, build_phases, index_prefix_bytes, single_routes, dynamic_histogram,
)


def test_table_bursts_use_page_ids_and_are_issued_after_score():
    h = page_history(32737, 32)
    tokens = [0, 1, 10, h.pages-1]
    v = {p: torch.ones(32,dtype=torch.bool).numpy() for p in tokens}
    route = build_phases(h,[0,1,h.pages-1],[10],v,"table",
                         score_ck=22, initial_wait_ck=14, extra_wait_ck=18)
    phases,wait = route
    assert phases[1] == [(INDEX_BASE,32),(INDEX_BASE+32*((h.pages-1)//8),32)]
    assert phases[3] == [(INDEX_BASE+32,32)]
    assert wait == [0,22,0,14,0,18]
    assert index_prefix_bytes(torch.tensor([[0, 1, 10, 9, 0]]),"table").tolist() == [[0,32,32,64,64,64]]


def fixture():
    t = 32737
    h = page_history(t,32)
    q = torch.zeros((1,4,128))
    k = torch.zeros((1,1,t,128))
    rank = ranked_pages(torch.zeros((1,h.pages)),h)
    return h,q,k,rank


def test_full_v_fallback_keeps_all_tokens_in_exact_same_safe_prefix():
    h,q,k,rank = fixture()
    out = single_routes(q,k,rank,h,1.,"table")
    for gate in ("hist64","checked"):
        ep = out[gate+"_exact"]
        fv = out[gate+"_full"]
        assert ep.fallback.item() and fv.fallback.item()
        assert fv.prefix.item()==ep.prefix.item()
        assert fv.bytes.item()>=ep.bytes.item()
        assert fv.bytes.item()<=fv.cap
        assert fv.prefix_drop.max() == 0
        assert torch.equal(fv.state.ids,ep.state.ids)


def test_dynamic_hist_matches_guard_and_reservation_for_both_index_modes():
    h,q,k,rank = fixture()
    q.fill_(1)
    k[:,:,52].fill_(4)
    for mode in ("direct","table"):
        r=dynamic_histogram(q,k,rank,h,.125,mode)
        assert r.bytes.item() <= r.cap
        assert r.state.valid.any(-1).sum().item() <= 68
        assert r.added.item()>0
        assert r.prefix_drop.max() <= .020003
        assert (r.state.mask | ~(r.state.valid & r.state.protected[..., None])[:,None]).all()
