import json
import numpy as np
import pytest
import torch

from gqa_probe import shared_refill as ref
from gqa_probe.fixed_k_v import burst_round, probabilities, ranked_pages, vector_burst
from gqa_probe.shared_refill_audit import replay_group


def fixture(device):
    t=8195
    g,h,d=2,4,128
    key=torch.zeros((1,g,t,d),device=device)
    key[:,:,:32]=1.5
    q=torch.ones(g,h,d,device=device)
    history,_,_=ref.quota(t,1000)
    rank=ranked_pages(torch.zeros(g,history.pages,device=device),history)
    return key,q,history,rank


def test_reservation_counts_exact_index_bursts():
    f=2*vector_burst(32)
    for n in (8,13,57,71):
        for requested in (0,1,7,19,92):
            for slack in (-1,0,1,18,31,32,90):
                used=208*1024+2*n*2432+burst_round(4*n)
                extra=burst_round(4*(n+requested))-burst_round(4*n)
                cap=used+f*requested+extra+slack
                got=ref.reserve(torch.tensor([used]),n,cap,100)[0].item()
                expected=max(j for j in range(101) if j==0 or used+j*f+burst_round(4*(n+j))-burst_round(4*n)<=cap)
                assert got==expected


@pytest.mark.parametrize("mode,recompute",[("mass",False),("blasst",False),("sparta",False),("sparta",True)])
@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_shared_stages_protection_quota_and_bus(mode,recompute,device):
    key,q,h,ranks=fixture(device)
    out=ref.run_shared(ref.RefillPolicy(mode,.05,1000,recompute),q,key,ranks,h.tokens,128**-.5,audit=True)
    assert (out.additional>0).all()
    ids=out.final.ids
    n=out.initial_count
    assert torch.equal(ids[:,:n],ranks[:,:n].sort(dim=-1).values)
    assert torch.equal(ids[:,n:],ranks[:,n:n+ids.shape[1]-n])
    if not recompute:
        assert torch.equal(out.final.mask[:,:,:n],out.first.mask)
    if mode!="blasst":
        lost=(probabilities(out.final.logits)*~out.final.mask).sum((2,3))
        assert (lost<=.050002).all()
    ledger=ref.ledger(out.final,h)
    assert (ledger["total"]<=out.ceiling).all()
    valid=out.final.valid[0].any(-1).cpu().tolist()
    group=out.final.mask[0].any(0).cpu().numpy()
    pages=ids[0].cpu().tolist()
    result=replay_group(h,ranks[0,:n].cpu().tolist(),list(np.array(pages[n:n+out.additional[0]])),
                        {page:union for page, union, active in zip(pages,group,valid) if active},0)
    json.dumps(result)
    assert result["transactions"]*32==ledger["total"][0]
    assert result["v_pages"]==ledger["v_pages"][0]
    assert sum(ledger[k][0].item() if isinstance(ledger[k],torch.Tensor) else ledger[k] for k in (
        "k_codes","k_scale_payload","k_zp_payload","k_padding","v_codes","v_scale_payload","v_zp_payload",
        "v_padding","index_payload","index_padding","summary_payload","summary_padding"))==ledger["total"][0]


def test_no_extras_degenerates_and_global_sparta_falls_back(monkeypatch):
    key,q,h,ranks=fixture("cpu")
    b=ref.run_shared(ref.RefillPolicy("all"),q,key,ranks,h.tokens,128**-.5)
    for mode in ("mass","blasst","sparta"):
        x=ref.run_shared(ref.RefillPolicy(mode,0),q,key,ranks,h.tokens,128**-.5)
        assert x.final is x.first
        assert torch.equal(x.final.mask,b.final.mask)
    original=ref.keep_mask
    n=b.initial_count
    def allow_all_after_refill(logits,p,valid,protect,mode,e,order=None):
        if logits.shape[-2]>n:
            return valid[:,None].expand_as(logits)
        return original(logits,p,valid,protect,mode,e,order)
    monkeypatch.setattr(ref,"keep_mask",allow_all_after_refill)
    x=ref.run_shared(ref.RefillPolicy("sparta",.05,1000,True),q,key,ranks,h.tokens,128**-.5,audit=True)
    y=ref.run_shared(ref.RefillPolicy("sparta",.05,1000,False),q,key,ranks,h.tokens,128**-.5,audit=True)
    assert x.fallback.all() and torch.equal(x.final.mask,y.final.mask)
