import numpy as np
import pytest
import torch

from gqa_probe.fixed_k_v import probabilities, ranked_pages
from gqa_probe.light_a1_reference import light_probabilities
from gqa_probe.round6_attention import Policy, physical_ledger, run_policy, subpage_mask
from gqa_probe.round6_memory import read_phases, simulate, transaction_trace
from gqa_probe.shared_refill import quota


def test_light_a1_vector():
    s=np.array([[0,-np.log(2),-2*np.log(2)]])
    e,p=light_probabilities(s)
    assert e.tolist()==[[1048576,524288,262144]]
    assert p.tolist()==[[598512,299256,149628]]
    codes,p=light_probabilities([[0,-100,-1000]],valid=[[True,True,False]])
    assert codes.tolist()==[[1048576,0,0]]
    assert p[0,0]>1046000 and p[0,1]==0
    _,f=light_probabilities([[0,0,0]],exact=True)
    assert len(set(f[0]))==1
    # Tied scores preserve page-ID with the existing stable top-k rule.
    h,_,_=quota(16500,1000)
    rank=ranked_pages(torch.ones(1,h.pages),h)
    assert rank[0,:len(h.protected)].tolist()==list(h.protected)


def test_timing_waits_dependencies_and_expanded_code_burst():
    plans=[[(0,128)],[(2048,64)],[(4096,96)],[(8192,32)]]
    no=read_phases(plans)
    yes=read_phases(plans,(0,11,23,29))
    assert yes[3]>=no[3]+63
    assert tuple(yes[4:])==tuple(no[4:])
    h,_,_=quota(8305,1000)
    pick=[*h.protected,9]
    mask={p:(np.arange(32)<min(32,h.tokens-p*32)) for p in h.protected}
    mask[9]=np.zeros(32,dtype=bool)
    mask[9][10]=True
    base=simulate(h,pick,[],mask,h.pages*208,(0,0,0,0))
    alt=simulate(h,pick,[],mask,h.pages*208,(0,0,0,0),width=8)
    assert alt["expanded_tokens"]==7
    assert alt["reads_bytes"]-base["reads_bytes"]==alt["extra_v_bytes"]==7*64+32
    phases,_=transaction_trace(h,pick,[],mask,h.pages*208)
    for phase in phases:
        bursts=[b for a,s in phase for b in range(a,a+s,32)]
        assert len(set(bursts))==len(bursts)


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda",marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_window_positions_subpage_and_budget(device):
    t=16417
    h,cap,n=quota(t,1000)
    q=torch.randn((3,4,128),device=device)
    k=torch.randn((1,3,t,128),device=device)*.15
    rank=ranked_pages(torch.zeros(3,h.pages,device=device),h)
    remote=torch.tensor([[0,0,0,0],[0,1,0,1],[1,1,1,1]],dtype=torch.bool,device=device)
    for gate in ("token","subpage"):
        p=Policy("dynamic",gate=gate,window="head",map_id="cal")
        out=run_policy(p,q,k,rank,h,cap,n,128**-.5,remote)
        z=physical_ledger(out.final,h,"q6",remote.any(1))
        assert (z["total"]<=cap).all()
        assert z["k_pages"][0].item()==len(h.protected) and z["summary_payload"][0]==0
        s=out.final
        v=s.valid[:,None]&s.protected[:,None,:,None]
        assert (s.mask|~v).all()
        assert (out.final.mask|~v).all()
        assert not (s.mask & ((~remote)[:,:,None,None]&~s.protected[:,None,:,None])).any()
        assert ((probabilities(s.logits)*~s.mask).sum((2,3))<=.020003).all()
