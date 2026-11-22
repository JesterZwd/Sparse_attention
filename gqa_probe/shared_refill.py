"""Shared INT6 candidate reservation and final V masks under a burst ceiling."""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
import triton
import triton.language as tl

from .fixed_k_v import burst_round, candidate_qk, keep_mask, probabilities, vector_burst
from .page_read_budget import page_history


PAGE = 32
SUMMARY_BYTES = 208


@lru_cache(maxsize=4096)
def quota(tokens, budget_bp):
    history = page_history(tokens, PAGE)
    dense = 2 * ((history.pages - 1) * vector_burst(PAGE) + vector_burst(history.tail_size))
    cap = dense * budget_bp // 10000
    summary = burst_round(history.pages * SUMMARY_BYTES)
    for n in range(min(history.pages, (cap-summary) // (2*vector_burst(PAGE)) + 2),
                   len(history.protected)-1, -1):
        kbytes = (n-1)*vector_burst(PAGE) + vector_burst(history.tail_size)
        if summary + burst_round(4*n) + 2*kbytes <= cap:
            return history, cap, n
    raise ValueError("budget cannot cover protection and INT6 metadata")


@dataclass(frozen=True)
class RefillPolicy:
    mode: str
    threshold: float = 0
    budget_bp: int = 1000
    recompute: bool = False

    def __post_init__(self):
        if self.mode not in ("all", "mass", "blasst", "sparta"):
            raise ValueError("unknown V mask rule")
        if not 0 <= self.threshold < 1 or self.mode == "all" and self.threshold:
            raise ValueError("invalid threshold")
        if self.recompute and self.mode != "sparta":
            raise ValueError("recompute applies to SPARTA")

    @property
    def name(self):
        return f"{self.mode}_{self.threshold:g}_{self.budget_bp}" + ("_recompute" if self.recompute else "")


@dataclass
class State:
    ids: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    valid: torch.Tensor
    protected: torch.Tensor
    mask: torch.Tensor


@dataclass
class Outcome:
    first: State
    final: State
    initial_count: int
    additional: torch.Tensor
    ceiling: int
    recomputed: torch.Tensor
    fallback: torch.Tensor


def ledger(state, h):
    """Break the common K/V allocation into codes, metadata and 32 B padding."""
    v = state.mask.any(1)
    vt = v.sum((1, 2))
    kt = state.valid.sum((1, 2))
    n = state.valid.any(-1).sum(1)
    codes_k, codes_v = 64*kt, 64*vt
    scale_v = 32*v.unflatten(-1, (-1,4)).any(-1).sum((1,2))
    zp_v = 32*v.unflatten(-1, (-1,8)).any(-1).sum((1,2))
    scale_k = (n-1)*burst_round(8*PAGE)+burst_round(8*h.tail_size)
    zp_k = (n-1)*burst_round(4*PAGE)+burst_round(4*h.tail_size)
    index = ((4*n+31)//32)*32
    summary = burst_round(h.pages*SUMMARY_BYTES)
    return dict(summary_payload=h.pages*SUMMARY_BYTES,summary_padding=summary-h.pages*SUMMARY_BYTES,
                index_payload=4*n,index_padding=index-4*n,k_codes=codes_k,k_scale_payload=8*kt,
                k_zp_payload=4*kt,k_padding=scale_k+zp_k-12*kt,
                v_codes=codes_v,v_scale_payload=8*vt,v_zp_payload=4*vt,
                v_padding=scale_v+zp_v-12*vt,k_bytes=codes_k+scale_k+zp_k,v_bytes=codes_v+scale_v+zp_v,
                total=summary+index+codes_k+scale_k+zp_k+codes_v+scale_v+zp_v,
                k_pages=n,v_pages=v.any(-1).sum(1),k_tokens=kt,v_tokens=vt)


def reserve(used, n, cap, max_add):
    """Largest prefix that can reserve a full K/V page and new index bursts."""
    page_kv = 2*vector_burst(PAGE)
    old_index = burst_round(4*n)
    nadd = ((cap-used+old_index-4*n)//(page_kv+4)).clamp(0, max_add)
    delta = ((4*(n+nadd)+31)//32)*32 - old_index
    return nadd - ((used+nadd*page_kv+delta>cap) & (nadd>0)).long()


@triton.jit
def _locked_mass(M, DROP, ACTIVE, OUT, H:tl.constexpr, E:tl.constexpr, HH:tl.constexpr, TH:tl.constexpr):
    g=tl.program_id(0)
    h=tl.arange(0,HH)
    spent=tl.load(DROP+g*H+h,h<H,0)
    for j in range(E):
        w=tl.load(M+(g*H+h)*E+j,h<H,0)
        active=tl.load(ACTIVE+g*E+j)!=0
        exceed=tl.sum(((h<H)&(spent+w>TH-1e-7)).to(tl.int32),0)>0
        drop=(exceed==0)&active
        spent+=tl.where(drop,w,0.)
        tl.store(OUT+g*E+j,tl.where(active&(drop==0),1,0))


def mass_extra(m, spent, active, threshold):
    g,h,n=m.shape
    if m.is_cuda:
        result=torch.empty_like(active, memory_format=torch.contiguous_format)
        _locked_mass[(g,)](m.contiguous(),spent.contiguous(),active.contiguous(),result,
                           h,n,triton.next_power_of_2(h),threshold,num_warps=1)
        return result
    keep=active.clone()
    for j in range(n):
        drop=((spent+m[:,:,j])<=threshold-1e-7).all(1)&active[:,j]
        spent=spent+m[:,:,j]*drop[:,None]
        keep[:,j]&=~drop
    return keep


def locked_sparta(p, old_keep, n, epsilon):
    """Maintain old retained tokens and reach the final Top-p using new tokens."""
    target=(1-epsilon+1e-7-(old_keep*p[...,:n,:]).sum((2,3),keepdim=False)).clamp_min(0)
    new=p[...,n:,:].flatten(2)
    order=new.argsort(dim=-1,descending=True,stable=True)
    sorted_p=new.gather(-1,order)
    keep=(sorted_p.cumsum(-1)-sorted_p<target[...,None])&(sorted_p>0)
    return torch.zeros_like(keep).scatter_(-1,order,keep).reshape_as(p[...,n:,:])


def run_shared(policy, q, key, ranks, tokens, scaling, *, audit=False, max_pages=None):
    h,cap,n=quota(tokens,policy.budget_bp)
    if max_pages is not None:
        if max_pages<len(h.protected):
            raise ValueError("K page limit excludes protection")
        n=min(n,max_pages)
    ids, perm=ranks[:,:n].sort(dim=-1)
    order=perm.argsort(-1)
    z,pos,valid,protect=candidate_qk(q,key,ids,h,scaling)
    p=probabilities(z) if policy.mode in ("mass","sparta") else None
    kept=keep_mask(z,p,valid,protect,policy.mode,policy.threshold,order)
    first=State(ids,pos,z,valid,protect,kept)
    zeros=torch.zeros(ids.shape[0],dtype=torch.int64,device=ids.device)
    if policy.mode=="all" or policy.threshold==0:
        return Outcome(first,first,n,zeros,cap,zeros.bool(),zeros.bool())
    used=ledger(first,h)["total"]
    # Initial full-read quota leaves less than one full page's reservation room.
    max_add=min(h.pages-n,(n-len(h.protected))//2+1)
    if max_pages is not None:
        max_add=min(max_add,max_pages-n)
    if max_add<=0:
        return Outcome(first,first,n,zeros,cap,zeros.bool(),zeros.bool())
    extra=reserve(used,n,cap,max_add)
    add_ids=ranks[:,n:n+max_add]
    active=torch.arange(max_add,device=ids.device)[None,:]<extra[:,None]
    l2,pos2,v2,p2=candidate_qk(q,key,add_ids,h,scaling,active)
    final=State(torch.cat((ids,add_ids),1),torch.cat((pos,pos2),1),torch.cat((z,l2),2),
                torch.cat((valid,v2),1),torch.cat((protect,p2),1),kept)
    if policy.mode=="blasst":
        local=l2.amax(-1)
        maximum=torch.maximum(local.cummax(-1).values,z.amax((2,3))[...,None])
        keep2=(local-maximum>=math.log(policy.threshold)).any(1)&active
        more=(keep2[...,None]&v2)[:,None].expand_as(l2)
    else:
        p=probabilities(final.logits)
        if policy.mode=="mass":
            spent=(~kept*p[...,:n,:]).sum((2,3))
            keep2=mass_extra(p[...,n:,:].sum(-1),spent,active,policy.threshold)
            more=(keep2[...,None]&v2)[:,None].expand_as(l2)
        else:
            more=locked_sparta(p,kept,n,policy.threshold)&v2[:,None]
    final.mask=torch.cat((kept,more),2)
    adopted=zeros.bool()
    rejected=zeros.bool()
    if policy.recompute:
        global_keep=keep_mask(final.logits,p,final.valid,final.protected,"sparta",policy.threshold)
        global_state=State(final.ids,final.positions,final.logits,final.valid,final.protected,global_keep)
        fits=ledger(global_state,h)["total"]<=cap
        adopted=fits&(extra>0)
        rejected=~fits&(extra>0)
        final.mask=torch.where(adopted[:,None,None,None],global_keep,final.mask)
    read=ledger(final,h)
    torch._assert_async((read["total"]<=cap).all(), "shared reservation exceeded ceiling")
    if policy.mode!="blasst":
        lost=(p*~final.mask).sum((2,3))
        torch._assert_async((lost<=policy.threshold+3e-6).all(),"head exceeds final mass limit")
    if audit:
        torch._assert_async((used<=cap).all(),"initial budget exceeded")
        mandatory=final.protected[...,None]&final.valid
        torch._assert_async((final.mask|~mandatory[:,None]).all(),"protected V omitted")
        ordered=final.ids.masked_fill(~final.valid.any(-1), h.pages).sort(dim=-1).values
        torch._assert_async(((ordered[:,1:]!=ordered[:,:-1])|(ordered[:,1:]==h.pages)).all(),"duplicated K read")
        torch._assert_async(((read["k_pages"]==n+extra)).all(),"additional K count mismatch")
        torch._assert_async(((final.mask[:,:,:n]==kept).all((1,2,3))|adopted).all(),"locked initial tokens changed")
    return Outcome(first,final,n,extra,cap,adopted,rejected)
