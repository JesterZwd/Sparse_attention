"""Decode reference using a common physical budget for packed or original KV."""
from dataclasses import dataclass
from math import ceil

import torch

from .fixed_k_v import probabilities, ranked_pages
from .lpddr5_banked import read_banked
from .page_read_budget import page_history
from .pagepack import (SegmentPlan, pack_pages, packed_logits, packed_output,
                       physical_v_mask, plan_segments, segment_statistics, select_plan, summary_pages)
from .pagepack_budget import plan_pages, read_trace, value_ledger
from .pagepack_probe import PACKINGS, RAW, selected_pages, vector_bytes
from .round6_attention import SAMPLE_LAYERS, SAMPLE_STEPS
from .round6_audit import compute_cycles
from .round7_attention import Attention
from .round7_gates import gate_cost, gate_mask
from .shared_refill import State
from .vskip_summary import METHOD_BYTES, StaticOnline, StaticPrefix, weights


@dataclass(frozen=True)
class Policy:
    name: str
    summary: str
    packing: str = "raw"
    gate: str = "full"
    budget: int = 5
    v_reserve: float = .65
    max_pages: int | None = None

    @property
    def spec(self):
        return next(x for x in PACKINGS if x.name==self.packing)


@dataclass
class PrefixLayer:
    summary: StaticPrefix
    plans: dict
    counts: dict


def build_prefix(prefill,specs,methods):
    layers=[]
    for key,value in zip(prefill.cache.key_cache,prefill.cache.value_cache):
        tokens=key.shape[-2]
        whole=(tokens//32)*32
        k=key[0,:,:whole].unflatten(1,(-1,32))
        v=value[0,:,:whole].unflatten(1,(-1,32))
        lengths=torch.full(k.shape[:2],32,dtype=torch.long,device=k.device)
        shared=segment_statistics(k,lengths)
        rec={m:summary_pages(k,lengths,stats=shared,method=m) for m in methods}
        plans,counts={},{}
        for p in specs:
            if p==RAW:
                continue
            plan=plan_segments(k,v,lengths,p.mode,k_threshold=p.k_threshold,
                               v_threshold=p.v_threshold,key_stats=shared)
            plans[p.name]=plan
            counts[p.name]=((plan.key_lengths>0).sum(-1),(plan.value_lengths>0).sum(-1))
        tail=key[0,:,whole:].float()
        partial=tail.sum(1,keepdim=True)
        lo=tail.amin(1,keepdim=True) if tail.shape[1] else torch.full_like(partial,torch.inf)
        hi=tail.amax(1,keepdim=True) if tail.shape[1] else torch.full_like(partial,-torch.inf)
        layers.append(PrefixLayer(StaticPrefix(tokens,32,rec,partial,lo,hi),plans,counts))
    return layers


def layouts(layer,spec,h,ranks):
    raw=(h.tokens-torch.arange(h.pages,device=ranks.device)*32).clamp(1,32)
    raw=raw[None].expand_as(ranks)
    if spec==RAW:
        return raw,raw
    keep_raw=torch.isin(torch.arange(h.pages,device=ranks.device),ranks.new_tensor(h.protected))
    k,v=layer.counts[spec.name]
    # Every page appended since prefill remains pinned at <=128 answer tokens.
    new=h.pages-k.shape[1]
    protected_counts=raw.clone()
    def merge(x):
        y=torch.cat((x,protected_counts[:,-new:]),dim=-1) if new else x
        return torch.where(keep_raw,raw,y)
    return merge(k),merge(v)


def selected_pack(k,v,lengths,alloc,prefix,spec):
    lengths=lengths.masked_fill(~alloc.active,0)
    if spec==RAW:
        return pack_pages(k,v,lengths,"raw",with_summary=False)
    sealed=prefix.plans[spec.name]
    indices=alloc.ids.clamp_max(sealed.lengths.shape[1]-1)
    old=select_plan(sealed,indices,active=alloc.active)
    local=plan_segments(k,v,lengths,"raw")
    blend=lambda old,new: torch.where(alloc.protected[...,None],new,old)
    merged=SegmentPlan(lengths,*(blend(getattr(old,p),getattr(local,p))
            for p in ("key_lengths","value_lengths","segment_starts","value_map","q_to_v")),spec.mode)
    return pack_pages(k,v,lengths,spec.mode,quantized=True,with_summary=False,plan=merged)


def choose_mask(alloc,pack,logits,gate):
    valid=pack.key_lengths>0
    mask=valid[:,None].expand_as(logits)
    if gate=="full":
        return mask,torch.zeros_like(alloc.count,dtype=torch.bool),alloc.count
    source=State(alloc.ids,alloc.ids,logits,valid,alloc.protected,mask)
    first=gate_mask(source,"hist64")
    over=value_ledger(alloc,physical_v_mask(first,pack))["over_bytes"]>0
    prefix=torch.where(over,alloc.safe_count,alloc.count)
    in_prefix=torch.arange(alloc.ids.shape[1],device=logits.device)<prefix[:,None]
    if over.any().item():
        clipped=valid&in_prefix[...,None]
        z=logits.masked_fill(~clipped[:,None],-torch.inf)
        safe=State(alloc.ids,alloc.ids,z,clipped,alloc.protected,mask&clipped[:,None])
        replacement=gate_mask(safe,"hist64")
        first=torch.where(over[:,None,None,None],replacement,first)
    limited=logits.masked_fill(~in_prefix[:,None,:,None],-torch.inf)
    loss=(probabilities(limited)*~first).sum((2,3))
    torch._assert_async((loss<=.020003).all(),"histogram exceeded prefix dose")
    return first,over,prefix


def sampled_cycles(h,p,alloc,mask,ledger,pv,reductions,fallback,rank_prefix,heads):
    key_rows=ledger["k_records"].cpu().tolist()
    pv_counts=pv.cpu().tolist()
    reduction_counts=reductions.cpu().tolist()
    old_k=alloc.k_counts.cpu().tolist()
    backs=fallback.cpu().tolist()
    ps=rank_prefix.cpu().tolist()
    actual=ledger["total"].cpu().tolist()
    v_records=ledger["v_records"].cpu().tolist()
    totals=alloc.count.cpu().tolist()
    for group,n in enumerate(key_rows):
        waits=compute_cycles(h,heads,n,0,heads,"none")
        scorer=waits["score_dot_ck"]*(2 if p.summary=="mr25" else 1)+waits["normalize_ck"]+waits["topk_insert_ck"]
        forced=h.tail_size+(len(h.protected)-1)*32
        initial=gate_cost(p.gate,n,heads,optional_tokens=n-forced,pages=totals[group])
        redo=None
        if backs[group]:
            m=sum(old_k[group][:ps[group]])
            redo=gate_cost(p.gate,m,heads,optional_tokens=m-forced,pages=ps[group])
        guard=totals[group]+(ps[group] if redo else 0) if p.gate=="hist64" else 0
        qk=waits["initial_qk_ck"]
        planner=ceil((totals[group]+1)/4)
        delay=[0,scorer,planner,qk+initial["gate_ck"]+guard+(redo["gate_ck"] if redo else 0)]
        phases=read_trace(alloc,mask,group)
        flat=sorted((s,s+n) for phase in phases for s,n in phase)
        assert all(a[1]<=b[0] for a,b in zip(flat,flat[1:]))
        bank=read_banked(phases,delay)
        assert bank["bursts"]*32==actual[group]
        pv_reduce_ck=ceil(reduction_counts[group]/(heads*4))
        pv_clocks=8*v_records[group]+pv_reduce_ck
        group_cycle=bank["phase_end_ck"][-1]+pv_clocks
        descriptors=(20 if alloc.ragged else 4)*totals[group]
        control=max(initial["control_sram_bytes"],redo["control_sram_bytes"] if redo else 0)+8+descriptors
        workset=max(n*heads*4+max(control,heads*(128+2)*4),
                    h.pages*heads*4+8*totals[group], alloc.directory_bytes+descriptors)
        k_read_ck=bank["phase_end_ck"][2]-bank["phase_start_ck"][2]
        yield dict(group=group,reads_bytes=actual[group],bursts=bank["bursts"],qk_mac=n*heads*128,
                   pv_mac=pv_counts[group]*128,scorer_mac=h.pages*heads*128*(2 if p.summary=="mr25" else 1),
                   banked_completion_ck=group_cycle,first_v_ck=bank["phase_start_ck"][-1],
                   planner_ck=planner,v_pool_reduction_ck=pv_reduce_ck,
                   control_sram_bytes=control,logits_sram_bytes=n*heads*4,attention_sram_bytes=workset,
                   k_qk_double_buffer_lower_bound_ck=group_cycle-min(k_read_ck,qk),
                   extra_two_k_tile_bytes=2*2432,
                   exact_retry_candidate_id_bits=max(1,(n-1).bit_length()),
                   exact_retry_id_bytes=2*n*heads,fallback=backs[group])


class CompressionAttention(Attention):
    def __init__(self,policy,prefix,gold,*,audit=False):
        super().__init__(policy,prefix,gold,audit=audit)
        self.spec=policy.spec
        self.group_steps=[]

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step<0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        g,t,d=key.shape[1:]
        scale=kwargs.get("scaling") or d**-.5
        layer=module.layer_idx
        p=self.policy
        h=page_history(t,32)
        if layer not in self.online:
            self.online[layer]=StaticOnline(self.prefix[layer].summary,(p.summary,))
        record=self.online[layer].append(key[0,:,-1],t)[p.summary]
        q=query.reshape(g,-1,d).float()
        score=weights(q,record,p.summary,scale)
        rank=ranked_pages(score,h)
        kl,vl=layouts(self.prefix[layer],self.spec,h,rank)
        alloc=plan_pages(rank,h,kl,vl,summary_bytes=METHOD_BYTES[p.summary],budget_percent=p.budget,
                         gate=p.gate,v_reserve=p.v_reserve,max_pages=p.max_pages,ragged=self.spec.ragged)
        kk,lengths=selected_pages(key,alloc.ids,t)
        vv,_=selected_pages(value,alloc.ids,t)
        pack=selected_pack(kk,vv,lengths,alloc,self.prefix[layer],self.spec)
        torch._assert_async((((pack.key_lengths>0).sum(-1)==alloc.k_counts).all() &
                             ((pack.value_lengths>0).sum(-1)==alloc.v_counts).all()),
                            "physical plan and actual segment lengths disagree")
        z=packed_logits(q,pack,scale)
        mask,fallback,rank_prefix=choose_mask(alloc,pack,z,p.gate)
        vmask=physical_v_mask(mask,pack)
        booked=value_ledger(alloc,vmask)
        torch._assert_async((booked["over_bytes"]==0).all(),"physical cap exceeded")
        torch._assert_async((vmask|~(alloc.full_mask()&alloc.protected[...,None])[:,None]).all(),
                            "protected V omitted")
        y=packed_output(z,pack,mask)
        self._add("decode_layer_calls",1)
        self._add("kv_group_calls",g)
        self._add("fallback_group_steps",fallback)
        self._add("over_budget_group_steps",booked["over_bytes"]>0)
        self._add("fetched_pages",alloc.count)
        self._add("retained_prefix_pages",rank_prefix)
        self._add("qk_mac",booked["k_records"]*q.shape[1]*d)
        pv=vmask.sum((1,2,3))
        self._add("pv_mac",pv*d)
        reductions=mask.sum((1,2,3))-pv
        self._add("v_pool_reduction_adds",reductions)
        self._add("scorer_mac",g*q.shape[1]*h.pages*d*(2 if p.summary=="mr25" else 1))
        for name,value_ in booked.items():
            self._add("read" if name=="total" else name,value_)
        hit=torch.zeros((g,h.pages),device=q.device,dtype=torch.bool).scatter_(1,alloc.ids,alloc.active)
        unique_v=torch.zeros_like(hit).scatter_(1,alloc.ids,vmask.any((1,3)))
        self._record_gold_aggregate(unique_v,g,t,q.device)
        gold=sorted(self._gold_page_tokens)
        if gold:
            self._add("evidence_k_page_hits",hit[:,gold])
            self._add("evidence_k_page_total",g*len(gold))
        old=page_history(t-1,32)
        released=sorted(set(old.protected)-set(h.protected))
        self._add("summary_write_bytes",g*METHOD_BYTES[p.summary])
        self._add("shared_k_stat_channel_adds",g*d)
        self._add("shared_k_stat_channel_compare",2*g*d)
        if self.step==0:
            self._add("prefill_summary_write_bytes",g*METHOD_BYTES[p.summary]*old.pages)
            self._add("prefill_k_stat_channel_adds",g*d*(old.tokens-old.pages))
            if self.spec!=RAW:
                remote=~torch.isin(torch.arange(h.pages,device=q.device),rank.new_tensor(h.protected))
                self._add("prefill_representative_write_bytes",((vector_bytes(kl)+vector_bytes(vl))*remote).sum())
                self._add("prefill_v_stat_channel_adds",g*d*(old.tokens-old.pages))
        if released and self.spec!=RAW:
            self._add("representative_write_bytes",(vector_bytes(kl[:,released])+vector_bytes(vl[:,released])).sum())
            self._add("v_stat_released_page_channel_adds",len(released)*g*31*d)
        if self.audit:
            self.group_steps.append((self.step,layer,alloc.cap,torch.stack(
                (booked["total"],alloc.count,booked["k_records"],booked["v_records"],fallback.long(),
                 rank_prefix,booked["directory_bytes"],booked["tag_bytes"]),dim=-1)))
        if self.audit and layer in SAMPLE_LAYERS and self.step in SAMPLE_STEPS:
            self.timing.extend(dict(step=self.step,layer=layer,**r) for r in sampled_cycles(
                h,p,alloc,vmask,booked,pv,reductions,fallback,rank_prefix,q.shape[1]))
        return self.output(query,y)
