import pytest
import torch

from gqa_probe.fixed_k_v import (candidate_qk, keep_mask, masked_output, page_keep, probabilities,
                               ranked_pages, traffic, gather_tokens)
from gqa_probe.oracle_attention import grouped_logits, grouped_output
from gqa_probe.page_read_budget import page_history
from gqa_probe.vskip_replay import replay
from gqa_probe.vskip_summary import encode, decoded, pack_codes, quantize_vector, unpack_codes


@pytest.mark.parametrize("bits",[4,6,8])
def test_packed_quest_outward_and_sign_equivalence(bits):
    generator=torch.Generator().manual_seed(37)
    x=torch.randn(8,23,128,generator=generator)
    x[0,0]=0
    lo=x-.5
    hi=x+.5
    code, step=quantize_vector(lo,bits,-1)
    packing=pack_codes(code,bits)
    assert packing.shape[-1]==128*bits//8
    assert torch.equal(code,unpack_codes(packing,bits))
    records=encode(x,lo,hi,torch.ones(8,23,1),f"q{bits}")
    l,u=decoded(records,f"q{bits}")
    assert (l<=lo).all() and (hi<=u).all()
    q=torch.randn(8,4,1,128,generator=generator)
    q[:,:,0,0]=0
    a=torch.where(q>=0,u[:,None],l[:,None])*q
    b=torch.maximum(q*l[:,None],q*u[:,None])
    assert torch.equal(a,b)
    history=page_history(23*32,32)
    w=a.sum(-1).softmax(-1).sum(1)
    assert torch.equal(ranked_pages(w,history),ranked_pages(b.sum(-1).softmax(-1).sum(1),history))
    tied=ranked_pages(torch.ones_like(w),history)[0].tolist()
    assert tied==list(history.protected)+[j for j in range(history.pages) if j not in history.protected]


@pytest.mark.parametrize("batch",[0,1,4,8])
def test_cumulative_cap_and_gpu_parity(batch):
    gen=torch.Generator().manual_seed(31)
    vals=torch.rand(8,4,29,generator=gen)**3
    vals=vals/vals.sum(-1,keepdim=True)
    protected=torch.zeros(8,29,dtype=torch.bool)
    protected[:,:6]=True
    order=vals.amax(1).argsort(dim=-1,stable=True)
    for eps in (0,.005,.01,.02,.05):
        keep=page_keep(vals,protected,eps,order=order,batch=batch)
        assert keep[:,:6].all()
        assert ((vals * ~keep[:,None]).sum(-1) <= eps+1e-7).all()
        if torch.cuda.is_available():
            gpu=page_keep(vals.cuda(),protected.cuda(),eps,order=order.cuda(),batch=batch)
            assert torch.equal(keep,gpu.cpu())


@pytest.mark.parametrize("size",[32,64])
def test_zero_epsilon_and_burst_tail(size):
    g=torch.Generator().manual_seed(32)
    h=page_history(size*8+3,size)
    k=torch.randn(1,2,h.tokens,128,generator=g)
    v=torch.randn(1,2,h.tokens,128,generator=g)
    q=torch.randn(1,8,1,128,generator=g)
    ids=ranked_pages(torch.randn(2,h.pages,generator=g),h)[:,:7].sort().values
    s=128**-.5
    logits,pos,valid,protected=candidate_qk(q.reshape(2,4,128),k,ids,h,s)
    p=probabilities(logits)
    all_=keep_mask(logits,p,valid,protected,"all",0)
    zero=keep_mask(logits,p,valid,protected,"mass",0)
    assert torch.equal(all_,zero)
    group_v=gather_tokens(v,pos)
    pm=torch.zeros(2,h.pages,dtype=torch.bool).scatter_(1,ids,True)
    expected=grouped_output(grouped_logits(q,k,s),v,pm,size)
    torch.testing.assert_close(masked_output(logits,group_v,zero),expected,atol=1e-6,rtol=1e-4)
    for method,e in (("mass",.05),("blasst",.05),("sparta",.05)):
        a=keep_mask(logits,p,valid,protected,method,e)
        assert (a | ~(protected[...,None] & valid)[:,None]).all()
        if method in ("mass","sparta"):
            assert ((p * ~a).sum((2,3)) <= e+1e-6).all()
    _,_,read,_,_=traffic(all_,valid,h,"q8")
    assert read[0].item()==replay(ids[0].tolist(),[True]*7,h.tokens,size,"q8")["total_bytes"]
