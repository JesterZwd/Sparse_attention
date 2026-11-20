"""Audit a common fullbatch traffic order and its logical storage/work."""
import math

import numpy as np
import torch

from .fixed_k_v import burst_round, vector_burst
from .shared_refill import SUMMARY_BYTES
from .vskip_replay import Bus


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).tolist()
    return zip(edges[::2], edges[1::2])


def replay_group(h, initial, added, union, group):
    """Common old addressing, one group, grouped contiguous 32 B reads."""
    bus = Bus()
    bus.read(0, h.pages * SUMMARY_BYTES)
    bus.read(1 << 20, 4 * (len(initial) + len(added)))
    stride = vector_burst(h.page_size)
    stage = []
    for batch in (initial, added):
        start = bus.bursts
        for p in batch:
            p = int(p)
            count = min(h.page_size,h.tokens-p*h.page_size)
            bus.read((1 << 22)+p*stride,vector_burst(count))
        stage.append(bus.bursts-start)
    v_commands = 0
    v_pages = 0
    for page, active in sorted(union.items()):
        if not active.any():
            continue
        v_pages += 1
        count = min(h.page_size,h.tokens-page*h.page_size)
        base = (1 << 25)+page*stride
        code_bursts = np.repeat(active,2)
        scales = active.reshape(-1,4).any(-1)
        zps = active.reshape(-1,8).any(-1)
        for start_addr, mask in ((base,code_bursts),
                                 (base+64*count,scales),
                                 (base+64*count+burst_round(8*count),zps)):
            for a,b in runs(mask):
                bus.read(start_addr+32*a,32*(b-a))
                v_commands += 1
    return dict(group=group,k_bursts_initial=stage[0],k_bursts_additional=stage[1],
                before_first_v_k_bursts=sum(stage),before_first_v_barriers=1+bool(added),
                v_coalesced_runs=v_commands,v_pages=v_pages,transactions=bus.bursts,
                row_hits=bus.row_hits,activations=bus.activations)


def snapshot_rows(out,h,ranks,led,first_error,final_error,loss_first,loss_final,
                  layer,step,policy,pv_counts):
    # The read ledger and address simulator are independently implemented.
    selected,valid,token_mask=[x.cpu().numpy() for x in
                               (out.final.ids,out.final.valid,out.final.mask.any(1))]
    g,heads,_,page=out.final.mask.shape
    primary=ranks[:,:out.initial_count].cpu().tolist()
    expected=led["total"].cpu().tolist()
    extras=out.additional.cpu().tolist()
    scalars=torch.stack((led["k_tokens"],led["v_tokens"],led["k_pages"],pv_counts),1).cpu().tolist()
    errors=torch.stack((first_error,final_error,loss_first,loss_final)).cpu().tolist()
    rows=[]
    for i in range(g):
        n=out.initial_count
        added=selected[i,n:n+extras[i]].tolist()
        union={int(page_id): keep for page_id,keep,live in zip(selected[i],token_mask[i],valid[i].any(-1)) if live}
        replay=replay_group(h,primary[i],added,union,i)
        if replay["transactions"]*32!=expected[i]:
            raise ValueError(f"address replay disagrees with burst accounting for group {i}")
        k_tokens,v_tokens,k_pages,pv=scalars[i]
        init_tokens=(n-1)*page+h.tail_size
        use_prob=policy.mode in ("sparta","mass") and bool(policy.threshold)
        fp32_logits=4*heads*k_pages*page
        page_table=4*heads*k_pages if policy.mode=="mass" else 0
        pv_acc=4*heads*(128+2)
        # Logical selector work and temporary tables (exclusive of KV/prefill).
        # Stable general sorting comparisons are estimates, not measured cycles.
        mass_sort=(math.ceil(n*math.log2(n)) if use_prob and policy.mode=="mass" else 0)
        tok_sort=0
        if use_prob and policy.mode=="sparta":
            widths=[init_tokens,(k_tokens if policy.recompute else k_tokens-init_tokens)]
            if policy.recompute and k_tokens>init_tokens:
                widths.append(k_tokens-init_tokens)
            tok_sort=heads*sum(math.ceil(w*math.log2(w)) for w in widths if w>1)
        rows.append(dict(layer=layer,step=step,initial_k=n,added_k=extras[i],k_pages=k_pages,
                         total_bytes=expected[i],qk_mac=heads*k_tokens*128,
                         v_tokens=v_tokens,first_stage_abs_error=errors[0][i],final_abs_error=errors[1][i],
                         initial_discarded_mass=errors[2][i],final_discarded_mass=errors[3][i],
                         exact_gate_exp=heads*(init_tokens+k_tokens) if use_prob else 0,
                         pv_renorm_exp=pv//128,pv_mac=pv,
                         q6_score_mac=128*heads*h.pages,
                         q6_normalize_exp=heads*h.pages,
                         q6_page_sort_comparisons_est=math.ceil(h.pages*math.log2(h.pages)),
                         fp32_logits=fp32_logits,fp16_logits=fp32_logits//2,
                         page_mass_table=page_table,accumulator_bytes=pv_acc,
                         token_mask_bytes=math.ceil(heads*k_pages*page/8),
                         mass_sort_comparisons_est=mass_sort,
                         exact_token_sort_comparisons_est=tok_sort,
                         mass_accumulation_check_ops=heads*k_pages if policy.mode=="mass" else 0,
                         mass_token_to_page_reduce_adds=heads*(init_tokens-n+k_tokens-k_pages) if policy.mode=="mass" else 0,
                         head_max_comparisons=(heads-1)*n if policy.mode=="mass" else 0,
                         blasst_peak_comparisons=heads*(k_tokens-k_pages) if policy.mode=="blasst" else 0,
                         blasst_running_max_comparisons=heads*k_pages if policy.mode=="blasst" else 0,
                         histogram64_counter_bytes=heads*64*8 if policy.mode=="sparta" else 0,
                         histogram64_token_inserts=heads*k_tokens if policy.mode=="sparta" else 0,
                         histogram64_two_pass_bin_visits=128*heads if policy.mode=="sparta" else 0,
                         **replay))
    return rows
