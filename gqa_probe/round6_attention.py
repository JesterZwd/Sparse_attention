"""INT6 single-pass, subpage PV and static head-window reference."""
from dataclasses import dataclass
from functools import lru_cache

import torch

from .fixed_k_v import (burst_round, byte_quota, candidate_qk, gather_tokens, keep_mask,
                        masked_output, page_keep, probabilities, ranked_pages, vector_burst)
from .oracle_attention import OracleAttention, grouped_logits, grouped_output
from .shared_refill import (State, Outcome, RefillPolicy, ledger, locked_sparta, mass_extra,
                            quota, reserve, run_shared)
from .vskip_summary import StaticOnline, decoded, METHOD_BYTES, weights


EPS=.02
SIZE=32
SAMPLE_LAYERS=(0,7,14,21,28,35)
SAMPLE_STEPS=(0,1,7,15,31,63)
HEAD_STEPS=(0,1,3,7,15,31,63)


@dataclass(frozen=True)
class Policy:
    kind: str
    pages: int=0
    gate: str="token"
    window: str="none"
    map_id: str=""
    method: str="q6"

    @property
    def name(self):
        return self.kind+(f"_{self.pages}" if self.pages else "")+(
            "_b8" if self.gate=="subpage" else "")+(
            f"_{self.window}_{self.map_id}" if self.window!="none" else "")+(
            f"_{self.method}" if self.kind=="compact" else "")


def int6_weights(q,record,scaling,remote):
    lo,hi=decoded(record,"q6")
    score=(torch.bmm(q.clamp_max(0),lo.transpose(1,2))
           +torch.bmm(q.clamp_min(0),hi.transpose(1,2)))*scaling
    return (score.softmax(-1)*remote[...,None]).sum(1)


def head_validity(state,remote):
    return state.valid[:,None] & (remote[:,:,None,None] | state.protected[:,None,:,None])


def subpage_mask(p,state,order,epsilon=EPS):
    """Low-score physical pages first, offsets 0,8,16,24 within each page."""
    chunks=p.unflatten(-1,(4,8)).sum(-1).flatten(-2)
    forced=state.protected.repeat_interleave(4,dim=1)
    slots=(order.flip(-1)[...,None]*4+torch.arange(4,device=p.device)).flatten(1)
    kept=page_keep(chunks,forced,epsilon,order=slots).unflatten(-1,(-1,4))
    return (kept[...,None].expand(*kept.shape,8).flatten(-2)&state.valid)[:,None].expand_as(p)


def build(q,key,ids,h,scale,remote,active=None):
    z,pos,val,pr=candidate_qk(q,key,ids,h,scale,active)
    out=State(ids,pos,z,val,pr,val[:,None].expand_as(z))
    masked=head_validity(out,remote)
    out.logits=z.masked_fill(~masked,-torch.inf)
    out.mask=masked
    return out


def physical_ledger(k,h,method,remote_groups):
    led=ledger(k,h)
    payload=h.pages*METHOD_BYTES[method]
    index=led["index_payload"]+led["index_padding"]
    led["summary_payload"]=remote_groups*payload
    led["summary_padding"]=remote_groups*(burst_round(payload)-payload)
    led["index_payload"]=led["index_payload"]*remote_groups
    led["index_padding"]=led["index_padding"]*remote_groups
    led["total"]=led["k_bytes"]+led["v_bytes"]+remote_groups*(burst_round(payload)+index)
    return led


def run_policy(policy,q,key,ranks,h,cap,quota_pages,scale,remote):
    """Numerical reference. Head masks also constrain the physical ledger."""
    xgroups=remote.any(1)
    if policy.kind=="dynamic" and policy.window=="none" and policy.gate=="token":
        return run_shared(RefillPolicy("sparta",EPS),q,key,ranks,h.tokens,scale)
    count=policy.pages or quota_pages
    if policy.kind=="compact":
        count=byte_quota(h.tokens,32,policy.method,5)
    ids,perm=ranks[:,:count].sort(-1)
    order=perm.argsort(-1)
    local_ids=torch.tensor(h.protected,device=ids.device)
    must=torch.isin(ids,local_ids)
    active=xgroups[:,None]|must
    first=build(q,key,ids,h,scale,remote,active)
    if policy.kind not in ("static","compact"):
        p=probabilities(first.logits)
        valid=first.mask
        if policy.gate=="subpage":
            first.mask=subpage_mask(p,first,order)&valid
        else:
            first.mask=keep_mask(first.logits,p,first.valid,first.protected,"sparta",EPS)&valid
    zeros=torch.zeros(ids.shape[0],dtype=torch.int64,device=ids.device)
    if policy.kind!="dynamic":
        return Outcome(first,first,count,zeros,cap,zeros.bool(),zeros.bool())
    max_add=min(h.pages-count,(count-len(h.protected))//2+1)
    if max_add==0:
        return Outcome(first,first,count,zeros,cap,zeros.bool(),zeros.bool())
    used=physical_ledger(first,h,policy.method,xgroups)["total"]
    extra=reserve(used,count,cap,max_add)*xgroups
    rest=ranks[:,count:count+max_add]
    active=torch.arange(max_add,device=rest.device)[None,:]<extra[:,None]
    rem=build(q,key,rest,h,scale,remote,active)
    final=State(torch.cat((ids,rest),1),torch.cat((first.positions,rem.positions),1),
                torch.cat((first.logits,rem.logits),2),torch.cat((first.valid,rem.valid),1),
                torch.cat((first.protected,rem.protected),1),first.mask)
    p=probabilities(final.logits)
    if policy.gate=="subpage":
        spent=(p[...,:count,:]*~first.mask).sum((2,3))
        add=p[...,count:,:].unflatten(-1,(4,8)).sum(-1)
        active_chunks=active.repeat_interleave(4,1).unflatten(-1,(-1,4))
        chosen=mass_extra(add.flip(2).flatten(-2),spent,
                          active_chunks.flip(1).flatten(1),EPS)
        extend=chosen.unflatten(-1,(-1,4)).flip(1).repeat_interleave(8,2)
        more=rem.mask&extend[:,None]
    else:
        more=locked_sparta(p,first.mask,count,EPS)&rem.mask
    final.mask=torch.cat((first.mask,more),2)
    torch._assert_async(((p*~final.mask).sum((2,3))<=EPS+3e-6).all(),"2% head limit exceeded")
    return Outcome(first,final,count,extra,cap,zeros.bool(),zeros.bool())


@lru_cache(maxsize=256)
def dense_bytes(tokens):
    h,_,_=quota(tokens,1000)
    return 2*((h.pages-1)*vector_burst(SIZE)+vector_burst(h.tail_size))


class Attention(OracleAttention):
    def __init__(self,policy,prefix,*,gold=(),sample=None,windows=None):
        super().__init__("dense",gold_spans=gold,page_size=SIZE)
        self.policy=policy
        self.prefix=prefix
        self.online={}
        self.sample=sample
        self.windows=windows
        self.peak=None
        self.peak_group_total=None
        self.over=[]

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step<0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        g,dim,t=key.shape[1],key.shape[-1],key.shape[-2]
        q=query.reshape(g,-1,dim).float()
        scale=kwargs.get("scaling") or dim**-.5
        h,cap,budget=quota(t,1000)
        layer=module.layer_idx
        policy=self.policy
        head_count=q.shape[1]
        self._add("decode_layer_calls",1)
        self._add("kv_group_calls",g)
        if policy.kind=="dense":
            logits=grouped_logits(query,key,scale)
            result=grouped_output(logits,value)
            kv=dense_bytes(t)
            self._add("read",kv*g)
            self._add("k_bytes",kv//2*g)
            self._add("v_bytes",kv//2*g)
            self._add("qk_mac",g*head_count*t*dim)
            self._add("pv_mac",g*head_count*t*dim)
            self._record_gold_aggregate(None,g,t,q.device)
            self._add("kv_groups_with_remote",g)
            self._add("remote_heads",g*head_count)
            if self.sample and self.step in HEAD_STEPS:
                self.sample.dense(self.step,layer,h,q,key,value,logits,result,scale)
            return result.reshape(1,query.shape[1],1,dim).to(query.dtype).transpose(1,2).contiguous(),None
        if layer not in self.online:
            self.online[layer]=StaticOnline(self.prefix[layer],(policy.method,))
        record=self.online[layer].append(key[0,:,-1],t)[policy.method]
        remote=(self.windows[layer] if self.windows is not None else
                torch.ones((g,head_count),dtype=torch.bool,device=q.device))
        if policy.window=="group":
            remote=remote.any(1,keepdim=True).expand_as(remote)
        elif policy.window=="none":
            remote=torch.ones_like(remote)
        rg=remote.any(1)
        if policy.window=="none":
            ranks=ranked_pages(weights(q,record,policy.method,scale),h)
        else:
            ranks=ranked_pages(int6_weights(q,record,scale,remote),h)
        out=run_policy(policy,q,key,ranks,h,cap,budget,scale,remote)
        k=out.final
        y=masked_output(k.logits,gather_tokens(value,k.positions),k.mask)
        led=physical_ledger(k,h,policy.method,rg)
        target_cap=out.ceiling//2 if policy.kind=="compact" else out.ceiling
        over=(led["total"]-target_cap).clamp_min(0)
        self._add("over_budget_group_steps",over>0)
        self._add("over_budget_bytes",over)
        if policy.kind=="single":
            self.over.append((self.step,layer,t,cap,over))
        if policy.kind!="single":
            torch._assert_async((over==0).all(),"unexpected read cap exception")
        for name,val in led.items():
            self._add("read" if name=="total" else name,val)
        qk_counts=head_validity(k,remote).sum((1,2,3))
        pv_counts=k.mask.sum((1,2,3))
        self._add("selected_k_pages",led["k_pages"])
        self._add("added_k_pages",out.additional)
        self._add("qk_mac",qk_counts*dim)
        self._add("pv_mac",pv_counts*dim)
        self._add("scorer_mac",remote.sum()*h.pages*dim)
        self._add("remote_heads",remote)
        self._add("kv_groups_with_remote",rg)
        mdl_peak=led["k_pages"].max()
        all_peak=led["k_pages"].sum()
        self.peak=mdl_peak if self.peak is None else torch.maximum(mdl_peak,self.peak)
        self.peak_group_total=all_peak if self.peak_group_total is None else torch.maximum(all_peak,self.peak_group_total)
        sel=torch.zeros((g,h.pages),device=q.device,dtype=torch.bool).scatter_(1,k.ids,k.mask.any((1,3)))
        self._record_gold_aggregate(sel,g,t,q.device)
        if self.sample and layer in SAMPLE_LAYERS and self.step in SAMPLE_STEPS:
            self.sample.closed(self.step,layer,h,policy,ranks,out,remote,led,qk_counts,pv_counts)
        return y.reshape(1,query.shape[1],1,dim).to(query.dtype).transpose(1,2).contiguous(),None

    def stats(self):
        result=super().stats()
        result["peak_k_pages"]=int(self.peak) if self.peak is not None else 0
        result["peak_total_k_pages"]=int(self.peak_group_total) if self.peak_group_total is not None else 0
        result["overshoot_events"]=[]
        if self.over:
            rows=torch.stack([x[-1] for x in self.over]).cpu().tolist()
            result["overshoot_events"]=[
                dict(step=step,layer=layer,tokens=t,ceiling=cap,group=g,extra_bytes=v)
                for (step,layer,t,cap,_),row in zip(self.over,rows) for g,v in enumerate(row) if v]
        return result
