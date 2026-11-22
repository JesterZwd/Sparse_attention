"""Fixed-page FP32 and affine32 compression diagnostics."""
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class Packing:
    name: str
    mode: str
    k_threshold: float = .8
    v_threshold: float | None = None

    @property
    def ragged(self):
        return self.mode == "mixed"


RAW = Packing("raw","raw")
PACKINGS = (RAW, *[Packing(m,m) for m in ("v2","v4","kv2","kv4")],
            *[Packing(f"mixed_k_{k:g}","mixed",k) for k in (.5,.8,1.2)],
            *[Packing(f"mixed_kv_{k:g}","mixed",k,k) for k in (.5,.8,1.2)])


def vector_bytes(count):
    """32 B burst charged codes, scales, and zero points per page."""
    return 64*count+(8*count+31)//32*32+(4*count+31)//32*32


def selected_pages(cache, ids, tokens):
    offsets = torch.arange(32,device=ids.device)
    pos = ids[...,None]*32+offsets
    lengths = (tokens-ids*32).clamp(0,32)
    index=pos.clamp_max(tokens-1).flatten(1)[...,None].expand(-1,-1,cache.shape[-1])
    values=cache[0].gather(1,index).unflatten(1,ids.shape[1:]+(32,))
    return values, lengths


def segment_logit_span(logits,value_map,value_lengths,valid):
    destination = value_map.clamp_min(0)[:,None].expand_as(logits)
    shown = valid[:,None]
    high = torch.full_like(logits,-torch.inf).scatter_reduce_(
        -1,destination,logits.masked_fill(~shown,-torch.inf),reduce="amax")
    low = torch.full_like(logits,torch.inf).scatter_reduce_(
        -1,destination,logits.masked_fill(~shown,torch.inf),reduce="amin")
    merged = (value_lengths>1)[:,None].expand_as(logits)
    return (high-low).masked_fill(~merged,0),merged


def probe(q,key,value,h,ids,scale,dense,evidence):
    from .pagepack import pack_pages, packed_logits, packed_output, plan_segments, segment_statistics

    k,n = selected_pages(key,ids,h.tokens)
    v,_ = selected_pages(value,ids,h.tokens)
    pin = torch.isin(ids,torch.tensor(h.protected,device=ids.device))
    stats = segment_statistics(k,n)
    raw = pack_pages(k,v,n,"raw",forced=pin,key_stats=stats)
    logits = packed_logits(q,raw,scale)
    output = packed_output(logits,raw)
    dense_norm = dense.norm(dim=-1).clamp_min(1e-8)
    original_norm = output.norm(dim=-1).clamp_min(1e-8)
    ev = torch.isin(ids,torch.tensor(sorted(evidence),device=q.device,dtype=torch.int64))
    valid = raw.key_lengths>0
    event_hits = ev.sum(-1).cpu().tolist()
    rows=[]
    for model in PACKINGS:
        plan=plan_segments(k,v,n,model.mode,forced=pin,key_stats=stats,
                           k_threshold=model.k_threshold,v_threshold=model.v_threshold)
        k_records=(plan.key_lengths>0).sum(-1)
        v_records=(plan.value_lengths>0).sum(-1)
        byte_k=vector_bytes(k_records).sum(-1)
        byte_v=vector_bytes(v_records).sum(-1)
        nt=torch.stack((k_records.sum(-1),v_records.sum(-1),byte_k,byte_v),dim=-1).cpu().tolist()
        spans,merged=segment_logit_span(logits,plan.value_map,plan.value_lengths,valid)
        evidence_merged=merged & ev[:,None,:,None]
        max_span=spans.amax((2,3)).cpu().tolist()
        max_gold_span=spans.masked_fill(~evidence_merged,0).amax((2,3)).cpu().tolist()
        segment_hist=torch.stack([(plan.value_lengths==x).sum((1,2)) for x in (1,2,4)],dim=-1).cpu().tolist()
        for precision in ("fp32","kv4") if model!=RAW else ("kv4",):
            if model==RAW:
                y=output
            else:
                packed=pack_pages(k,v,n,model.mode,forced=pin,key_stats=stats,plan=plan,
                                  quantized=precision=="kv4")
                y=packed_output(packed_logits(q,packed,scale),packed)
            d=(y-dense).norm(dim=-1)
            c=(y-output).norm(dim=-1)
            accuracy=torch.stack((d,d/dense_norm,c,c/original_norm),dim=1).cpu().tolist()
            for group,(dabs,drel,cabs,crel) in enumerate(accuracy):
                count_k,count_v,k_bytes,v_bytes=nt[group]
                rows.append(dict(group=group,policy=model.name,precision=precision,head_dense_abs=dabs,
                                 head_dense_rel=drel,head_pack_abs=cabs,head_pack_rel=crel,
                                 worst_segment_logit_span=max_span[group],
                                 worst_evidence_logit_span=max_gold_span[group],
                                 evidence_pages=event_hits[group],evidence_total=len(evidence),
                                 segment_counts_124=segment_hist[group],k_records=count_k,v_records=count_v,
                                 k_burst_bytes=k_bytes,v_burst_bytes=v_bytes,
                                 qk_mac=count_k*q.shape[1]*128,pv_mac=count_v*q.shape[1]*128))
    return rows,logits
