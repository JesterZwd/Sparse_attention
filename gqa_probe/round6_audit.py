"""Common-state metrics, timing, and fixed-K head-window inference."""
from functools import lru_cache
from math import ceil
import numpy as np
import torch

from .fixed_k_v import gather_tokens, masked_output, probabilities, ranked_pages
from .oracle_attention import grouped_output
from .round6_attention import (Policy,SAMPLE_LAYERS,SAMPLE_STEPS,head_validity,
                              physical_ledger,run_policy,subpage_mask)
from .round6_memory import simulate
from .shared_refill import State,quota
from .vskip_summary import METHOD_BYTES, StaticPrefix, StaticOnline, weights


def compute_cycles(h,nh,ki,ka,heads,gate):
    active=nh>0
    # 4 head lanes * 16 channel MACs. A mask on a mixed group does
    # not shorten a row on the fixed-lane model.
    score_dot=ceil(128*h.pages/16) if active else 0
    normalize=ceil(heads*h.pages/16)*2+12*heads if active else 0
    sel=ceil(h.pages/4) if active else 0
    sort_gate=lambda tokens: ((2*tokens+128) if tokens else 0) if gate=="token" else (
        (ceil(tokens/4)+ceil(tokens/8)) if tokens and gate=="subpage" else 0)
    return dict(score_dot_ck=score_dot,normalize_ck=normalize,topk_insert_ck=sel,
                initial_qk_ck=8*ki,extra_qk_ck=8*ka,
                initial_gate_ck=sort_gate(ki),extra_gate_ck=sort_gate(ka))


def bit_patterns(mask):
    pos=torch.arange(32,device=mask.device,dtype=torch.int64)
    return (mask.to(torch.int64)<<pos).sum(-1)


def decode_pattern(value):
    return ((np.int64(value)>>np.arange(32,dtype=np.int64))&1).astype(bool)


def timing_and_counts(h,initial,added,reads,nh,active_heads,kt,pv,head_tokens,
                      gate,summary,controls=True,width=1):
    ni=sum(min(32,h.tokens-p*32) for p in initial)
    na=kt-ni
    # The fixed physical lane schedule remains gated per group.
    clocks=compute_cycles(h,nh,ni,na,active_heads,gate)
    if not controls:
        clocks["initial_gate_ck"]=clocks["extra_gate_ck"]=0
    score=clocks["score_dot_ck"]+clocks["normalize_ck"]+clocks["topk_insert_ck"]
    d1=clocks["initial_qk_ck"]+clocks["initial_gate_ck"]
    d2=clocks["extra_qk_ck"]+clocks["extra_gate_ck"]
    wait=(0,score,d1 if na else 0,d2 if na else d1)
    mem=simulate(h,initial,added,reads,summary,wait,width=width,index=bool(nh and summary))
    # Sequential PV model with no inter-group packing: one token per head
    # occupies a fixed lane even when it does not participate.
    physical_v_compute=sum(int(mask.sum()) for mask in reads.values())
    issued_q=kt*active_heads
    issued_pv=physical_v_compute*active_heads
    compact_pv=ceil(pv*128/(16*active_heads))
    return dict(**mem,**clocks,
                total_group_ck=mem["lpddr_ck"]+8*physical_v_compute,
                ideal_packed_array_ck=ceil((128*head_tokens+128*pv+nh*h.pages*128)/(16*active_heads)),
                fixed_array_ck=clocks["score_dot_ck"]+8*(kt+physical_v_compute),
                qk_mac=head_tokens*128,pv_mac=pv*128,
                scorer_mac=nh*h.pages*128 if summary else 0,
                scorer_pe_util=nh/active_heads if summary else None,
                qk_pe_util=head_tokens/max(1,issued_q),pv_pe_util=pv/max(1,issued_pv),
                peak_logits_bytes_fp32=head_tokens*4,
                peak_logits_bytes_fp16=head_tokens*2,
                token_mask_bytes=ceil(head_tokens/8),
                histogram64_bytes=nh*64*8 if gate=="token" else 0,
                histogram64_inserts=head_tokens if gate=="token" else 0,
                histogram64_bin_visits=128*nh*(1+bool(added)) if gate=="token" else 0,
                head_drop_state_bytes=4*nh if gate=="subpage" else 0,
                page_or_block_probability_bytes=4*nh*(len(initial)+len(added))*4 if gate=="subpage" else 0,
                subpage_reduce_adds=nh*(kt-ceil(kt/8)) if gate=="subpage" else 0,
                subpage_bound_checks=nh*ceil(kt/8) if gate=="subpage" else 0,
                pv_accumulator_bytes=active_heads*(128+2)*4,
                v_compute_tokens_physical=physical_v_compute,
                pv_ideal_packed_ck=compact_pv,
                initial_k=len(initial),added_k=len(added),group_barriers=(1+bool(added)+bool(summary)))


@lru_cache(maxsize=256)
def dense_timing(h):
    initial=list(range(h.pages))
    reads={p:(np.arange(32)<min(32,h.tokens-p*32)) for p in initial}
    return timing_and_counts(h,initial,[],reads,0,4,h.tokens,h.tokens*4,h.tokens*4,"none",0,False)


class Audit:
    def __init__(self,common_policy=None,*,dense_cost=False):
        self.common_policy=common_policy
        self.dense_cost=dense_cost
        self.rows=[]
        self.local=[]
        self.common=[]
        self.online={}

    def sparse_rows(self,h,policy,ranks,out,remote,led,head_counts,pv_counts):
        k=out.final
        ids=k.ids.cpu().tolist()
        active=k.valid.any(-1).cpu().tolist()
        payload=led["summary_payload"].cpu().tolist()
        union=k.mask.any(1).cpu().numpy()
        all_burst=led["total"].cpu().tolist()
        starters=ranks[:,:out.initial_count].cpu().tolist()
        nd=out.additional.cpu().tolist()
        nhs=remote.sum(1).cpu().tolist()
        kts=led["k_tokens"].cpu().tolist()
        qks=head_counts.cpu().tolist()
        pvs=pv_counts.cpu().tolist()
        for i,gids in enumerate(ids):
            selected={x for x,live in zip(gids[:out.initial_count],active[i][:out.initial_count]) if live}
            ini=[p for p in starters[i] if p in selected]
            add=gids[out.initial_count:out.initial_count+nd[i]]
            v={p:val for p,val,live in zip(gids,union[i],active[i]) if live}
            gate=policy.gate if policy.kind in ("dynamic","single") else "none"
            metrics=timing_and_counts(h,ini,add,v,nhs[i],remote.shape[1],kts[i],pvs[i],qks[i],gate,payload[i])
            if metrics["reads_bytes"]!=all_burst[i]:
                raise ValueError("address trace and packed bytes disagree")
            yield dict(group=i,over_budget=bool(all_burst[i]>out.ceiling),**metrics)

    def closed(self,step,layer,h,policy,ranks,out,remote,led,head_counts,pv_counts):
        self.rows.extend(dict(step=step,layer=layer,policy=policy.name,**x)
                         for x in self.sparse_rows(h,policy,ranks,out,remote,led,head_counts,pv_counts))

    def dense(self,step,layer,h,q,key,value,raw,dense,scale):
        valid=torch.zeros(h.pages,device=q.device,dtype=torch.bool)
        valid[list(h.protected)]=True
        local=grouped_output(raw,value,valid[None].expand(q.shape[0],-1))
        error=(dense-local).norm(dim=-1)/(dense.norm(dim=-1)+1e-8)
        if self.common_policy is not None:
            self.local.append(dict(layer=layer,step=step,relative=error.cpu().tolist()))
        if step not in SAMPLE_STEPS or layer not in SAMPLE_LAYERS:
            return
        if self.dense_cost:
            self.rows.append(dict(step=step,layer=layer,group=0,policy="dense",over_budget=False,
                                  **dense_timing(h)))
        pol=self.common_policy
        if pol is None:
            return
        # Rebuild only at observed common states; online-maintained summaries
        # are used by all actual decode policies.
        prefix=StaticPrefix.build(key[:,:,:-1],32,("q6",))
        data=StaticOnline(prefix,("q6",)).append(key[0,:,-1],h.tokens)["q6"]
        rank=ranked_pages(weights(q,data,"q6",scale),h)
        h,cap,quota_pages=quota(h.tokens,1000)
        remote=torch.ones_like(error,dtype=torch.bool)
        out=run_policy(pol,q,key,rank,h,cap,quota_pages,scale,remote)
        k=out.final
        v=gather_tokens(value,k.positions)
        reference=masked_output(k.logits,v,k.valid[:,None].expand_as(k.logits))
        token=masked_output(k.logits,v,k.mask)
        prob=probabilities(k.logits)
        k_order=rank.argsort(-1).gather(1,k.ids).argsort(-1)
        bmask=subpage_mask(prob,k,k_order)
        subpage=masked_output(k.logits,v,bmask)
        token_err=(token-reference).norm(dim=-1).cpu().tolist()
        block_err=(subpage-reference).norm(dim=-1).cpu().tolist()
        token_dense=(token-dense).norm(dim=-1).cpu().tolist()
        local_dense=(local-dense).norm(dim=-1).cpu().tolist()
        lost=(prob*~bmask).sum((2,3))
        torch._assert_async((lost<=.020003).all(),"B exceeds 2% common-state mass limit")
        ids=k.ids.cpu().tolist()
        live=k.valid.any(-1).cpu().tolist()
        old=bit_patterns(k.mask).cpu().tolist()
        block=bmask.any(1).cpu().numpy()
        tok=k.mask.any(1).cpu().numpy()
        block_pv=bmask.sum((1,2,3)).cpu().tolist()
        base_pv=k.mask.sum((1,2,3)).cpu().tolist()
        ini=rank[:,:out.initial_count].cpu().tolist()
        old_cost=physical_ledger(k,h,"q6",remote.any(1))["total"].cpu().tolist()
        for g,ids_g in enumerate(ids):
            real=[(i,p) for i,p in enumerate(ids_g) if live[g][i]]
            additional=[p for i,p in real if i>=out.initial_count]
            kt=sum(min(32,h.tokens-p*32) for _,p in real)
            base={p:tok[g,i] for i,p in real}
            blocks={p:block[g,i] for i,p in real}
            common=dict(h=h,initial=ini[g],added=additional,nh=4,active_heads=4,kt=kt,
                        head_tokens=4*kt,summary=h.pages*208)
            ref=timing_and_counts(reads=base,pv=base_pv[g],gate="token",**common)
            if ref["reads_bytes"]!=old_cost[g]:
                raise ValueError("same-state reference traffic mismatch")
            a4=timing_and_counts(reads=base,pv=base_pv[g],gate="token",width=4,**common)
            a8=timing_and_counts(reads=base,pv=base_pv[g],gate="token",width=8,**common)
            b=timing_and_counts(reads=blocks,pv=block_pv[g],gate="subpage",**common)
            if a4["pv_mac"]!=ref["pv_mac"] or a8["pv_mac"]!=ref["pv_mac"]:
                raise ValueError("expanded reads altered physical PV mask")
            self.common.append(dict(step=step,layer=layer,group=g,tokens=h.tokens,initial=ini[g],
                                    added=additional,active_pages=[p for _,p in real],
                                    per_head_pv_bits=[[old[g][r][i] for i,_ in real] for r in range(4)],
                                    token_abs=token_err[g],subpage_abs=block_err[g],
                                    baseline_dense_abs=token_dense[g],window_dense_abs=local_dense[g],
                                    cap=cap,base=ref,expand4=a4,expand8=a8,subpage=b))


def fixed_k_window(h,row,remote,kind):
    if kind=="group":
        remote=np.full(4,np.any(remote))
    nh=int(np.sum(remote))
    active=row["active_pages"]
    initial=row["initial"]
    add=row["added"]
    mask={}
    total_q=total_v=kt=0
    for j,page in enumerate(active):
        if not nh and page not in h.protected:
            continue
        heads=np.stack([decode_pattern(row["per_head_pv_bits"][r][j]) if remote[r] or page in h.protected else
                        np.zeros(32,dtype=bool) for r in range(4)])
        tok=min(32,h.tokens-page*32)
        kt+=tok
        total_v+=int(heads.sum())
        total_q+=tok*(4 if page in h.protected else nh)
        mask[page]=heads.any(0)
    if not nh:
        initial=[p for p in initial if p in h.protected]
        add=[]
    r=timing_and_counts(h,initial,add,mask,nh,4,kt,total_v,total_q,"token",h.pages*208 if nh else 0,bool(nh))
    return dict(remote_heads=nh,dense_abs=np.where(remote,row["baseline_dense_abs"],row["window_dense_abs"]).tolist(),**r)
