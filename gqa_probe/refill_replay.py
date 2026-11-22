"""Matched single-pass and locked histogram refill for a common Q/K/V state."""
from dataclasses import dataclass
from math import ceil

import torch

from .fixed_k_v import burst_round, candidate_qk, probabilities, vector_burst
from .round6_attention import physical_ledger
from .round6_audit import compute_cycles
from .round6_memory import merge, transaction_trace
from .round7_gates import HISTOGRAM_BINS, LOGIT_BIN_WIDTH, gate_cost, gate_mask
from .round7_single import prepare_single
from .shared_refill import State
from .vskip_summary import METHOD_BYTES


INDEX_BASE = 1 << 20
MAX_PAGES = 68
EPS = .02


def index_prefix_bytes(ranks, mode):
    """Unique physical LPDDR bursts covering a prefix of the ranked pages."""
    g,n = ranks.shape
    zero = ranks.new_zeros((g,1))
    if mode == "direct":
        return ranks.new_zeros((g,n+1))
    if mode != "table":
        raise ValueError("unknown index placement")
    pos = torch.arange(n,device=ranks.device)
    blocks = ranks//8
    same = blocks[:, :, None] == blocks[:, None, :]
    novel = ~(same & (pos[:, None] > pos[None, :])).any(-1)
    return torch.cat((zero,32*novel.long().cumsum(-1)),dim=-1)


def index_addresses(pages):
    return sorted({INDEX_BASE+(p//8)*32 for p in pages})


def build_phases(h, initial, added, v, mode, *, score_ck, initial_wait_ck, extra_wait_ck):
    """Summary, first index, first K, refill index, refill K, then V."""
    phases,_ = transaction_trace(h,initial,added,v,h.pages*METHOD_BYTES["q6"],index=False)
    first = index_addresses(initial) if mode=="table" else []
    extra = sorted(set(index_addresses(added))-set(first)) if mode=="table" else []
    lists = [phases[0],merge([(a,32) for a in first]),phases[1],
             merge([(a,32) for a in extra]),phases[2],phases[3]]
    return lists, [0,score_ck,0,initial_wait_ck,0,extra_wait_ck]


def full_prefix_bytes(h, count):
    return vector_burst(h.tail_size)+(count-1)*vector_burst(h.page_size)


def ledger(state,h,index):
    """Replace the historical contiguous order-list charges with the chosen layout."""
    group = torch.ones(state.ids.shape[0],dtype=torch.bool,device=state.ids.device)
    charge = physical_ledger(state,h,"q6",group)
    base = charge["total"]-charge["index_payload"]-charge["index_padding"]
    return base+index,charge["k_bytes"],charge["v_bytes"]


@dataclass
class Route:
    state: State
    initial: int
    added: torch.Tensor
    cap: int
    bytes: torch.Tensor
    v_bytes: torch.Tensor
    index: torch.Tensor
    fallback: torch.Tensor
    prefix: torch.Tensor
    all_drop: torch.Tensor
    prefix_drop: torch.Tensor
    gate: str
    fallback_mode: str


def with_mask(source,mask):
    return State(source.ids,source.positions,source.logits,source.valid,source.protected,mask)


def single_routes(q,key,ranks,h,scale,mode,*,histogram_fallback=False):
    source, slot_rank, _, cap, _ = prepare_single(q,key,ranks,h.tokens,scale,"q6",1000,MAX_PAGES)
    count = source.ids.shape[-1]
    indexes = index_prefix_bytes(ranks[:,:count],mode)[:,count]
    full = vector_burst(32)
    tail = vector_burst(h.tail_size)
    v_budget = cap-indexes-full_prefix_bytes(h,count)-burst_round(h.pages*METHOD_BYTES["q6"])
    safe = ((v_budget-tail)//full+1).clamp_max(count)
    torch._assert_async((safe>=len(h.protected)).all(),"cannot reserve protected V")
    whole = probabilities(source.logits)
    result = {}
    for gate,tau in (("hist64",0.),("checked",8.)):
        first = gate_mask(source,gate,tau)
        used,_,_ = ledger(with_mask(source,first),h,indexes)
        fallback = used>cap
        prefix = torch.where(fallback,safe,count)
        valid = source.valid & (slot_rank<prefix[:,None])[...,None]
        restricted = State(source.ids,source.positions,source.logits.masked_fill(~valid[:,None],-torch.inf),
                           valid,source.protected,source.mask & valid[:,None])
        prob = probabilities(restricted.logits)
        for option in ("exact","full","hist64") if histogram_fallback else ("exact","full"):
            keep = torch.where(fallback[:,None,None,None],gate_mask(restricted,option),first)
            state = with_mask(source,keep)
            spent,_, v = ledger(state,h,indexes)
            prefix_drop = (prob*~keep).sum((2,3))
            all_drop = (whole*~keep).sum((2,3))
            mandatory = (state.protected[...,None] & state.valid)[:,None]
            torch._assert_async(((spent<=cap).all() & (keep|~mandatory).all() &
                                (prefix_drop<=EPS+3e-6).all()),"single replay violated budget or mass")
            result[f"{gate}_{option}"] = Route(state,count,torch.zeros_like(safe),cap,spent,v,indexes,
                                              fallback,prefix,all_drop,prefix_drop,gate,option)
    return result


def locked_histogram(logits, old_mask, n):
    """Apply the same 64 bins to new tokens under the expanded denominator."""
    flat = logits.flatten(2)
    gap = flat.amax(-1,keepdim=True)-flat
    mass = (-gap).exp()
    old = old_mask.flatten(2)
    old_n = old.shape[-1]
    allowed = (EPS*mass.sum(-1,keepdim=True)-(mass[...,:old_n]*~old).sum(-1,keepdim=True)).clamp_min(0)
    new_gap = gap[...,old_n:]
    active = torch.isfinite(new_gap)
    bucket = (new_gap/LOGIT_BIN_WIDTH).clamp(0,HISTOGRAM_BINS-1).long()
    bins = mass.new_zeros((*mass.shape[:-1],HISTOGRAM_BINS))
    bins.scatter_add_(-1,bucket,mass[...,old_n:])
    keep_buckets = bins.flip(-1).cumsum(-1).flip(-1)>allowed
    return (keep_buckets.gather(-1,bucket)&active).reshape_as(logits[...,n:,:])


def dynamic_histogram(q,key,ranks,h,scale,mode):
    cap = 2*full_prefix_bytes(h,h.pages)//10
    cap_n = min(h.pages,MAX_PAGES)
    all_index = index_prefix_bytes(ranks[:,:cap_n],mode)
    summary = burst_round(h.pages*METHOD_BYTES["q6"])
    full = vector_burst(32)
    tail = vector_burst(h.tail_size)
    allowed = (cap-summary-2*tail)//(2*full)+1
    upper = min(cap_n,allowed)
    # Start with a shared depth that still allows all initial V in all groups.
    options = torch.arange(len(h.protected),upper+1,device=q.device)
    total_full = summary+2*(tail+(options-1)*full)+all_index[:,options]
    n = int(torch.where((total_full<=cap).all(0), options, 0).amax().item())
    if n<len(h.protected):
        raise ValueError("initial quota below protection")
    ids = ranks[:,:n].sort(-1).values
    z,pos,valid,must = candidate_qk(q,key,ids,h,scale)
    first = State(ids,pos,z,valid,must,valid[:,None].expand_as(z))
    kept = gate_mask(first,"hist64")
    first = with_mask(first,kept)
    used,_,_ = ledger(first,h,all_index[:,n])
    option = torch.arange(n,cap_n+1,device=q.device)
    needed = used[:,None]+2*full*(option-n)+all_index[:,n:]-all_index[:,n,None]
    count = torch.where(needed<=cap,option,n).amax(-1)
    extra = count-n
    if cap_n>n and extra.any().item():
        add_ids = ranks[:,n:cap_n]
        active = torch.arange(cap_n-n,device=q.device)[None,:] < extra[:,None]
        l2,p2,v2,m2 = candidate_qk(q,key,add_ids,h,scale,active)
        logits = torch.cat((z,l2),2)
        more = locked_histogram(logits,kept,n)
        final = State(torch.cat((ids,add_ids),1),torch.cat((pos,p2),1),logits,
                      torch.cat((valid,v2),1),torch.cat((must,m2),1),torch.cat((kept,more),2))
    else:
        final = first
    index = all_index.gather(1,count[:,None]).squeeze(-1)
    spent,_,v = ledger(final,h,index)
    dropped = (probabilities(final.logits)*~final.mask).sum((2,3))
    torch._assert_async(((spent<=cap).all() & (dropped<=EPS+3e-6).all()),
                        "locked histogram violated budget or head allowance")
    no_fallback = torch.zeros_like(extra,dtype=torch.bool)
    return Route(final,n,extra,cap,spent,v,index,no_fallback,count,dropped,dropped,"hist64","none")


def workset(h, initial, extra, prefix, heads, gate, fallback, *, index_mode, first_table, final_table):
    """Model concurrently live state, in bytes, and dependencies in LPDDR CK."""
    # First batch always includes protected tail; added pages are full.
    first_tokens = h.tail_size+(initial-1)*h.page_size
    add_tokens = extra*h.page_size
    protected = h.tail_size+(len(h.protected)-1)*h.page_size
    first = gate_cost(gate,first_tokens,heads,optional_tokens=first_tokens-protected,pages=initial)
    logits_first = first_tokens*heads*4
    logits_final = (first_tokens+add_tokens)*heads*4
    sort_ids = 4*(initial+extra)
    first_track = 4*heads*2  # max and partition/peak state, common to all.
    alloc = 8  # two int32 budget counters.
    table_held = first_table if index_mode=="table" else 0
    phase_one = first["control_sram_bytes"] + sort_ids + first_track + alloc + table_held
    fallback_tokens = h.tail_size+(prefix-1)*h.page_size
    fallback_ck = 0
    last_control = phase_one
    score_est = compute_cycles(h,heads,first_tokens,add_tokens,heads,"none")
    score_cycles = sum(score_est[k] for k in ("score_dot_ck","normalize_ck","topk_insert_ck"))
    # Scores for full-history normalization, plus bounded rank keys and IDs.
    scorer_state = h.pages*heads*4+8*(initial+extra)+first_track
    initial_wait = score_est["initial_qk_ck"]+first["gate_ck"]+initial
    final_wait = 0
    if extra:
        more = gate_cost(gate,add_tokens,heads,pages=extra)
        # Old logits and first mask are reread to carry the spent mass into
        # the expanded denominator. The peak may change after appended QK.
        final_wait = score_est["extra_qk_ck"]+more["gate_ck"]+2*ceil(first_tokens*heads/4)
        last_control = (sort_ids+first_track+alloc+final_table+
                        first["head_mask_bytes"]+first["union_mask_bytes"]+
                        more["control_sram_bytes"])
        initial_wait += extra+2  # reservation check over the appended prefix.
    elif fallback!="none":
        if fallback=="full":
            fallback_ck = ceil(fallback_tokens/4)+prefix+1
            last_control = sort_ids+first_track+alloc+table_held
        else:
            redo = gate_cost(fallback,fallback_tokens,heads,
                             pages=prefix,optional_tokens=fallback_tokens-protected)
            fallback_ck = redo["gate_ck"]+prefix+1
            last_control = redo["control_sram_bytes"]+sort_ids+first_track+alloc+table_held
        final_wait = fallback_ck
    # A table burst reused by the refill is retained through its second
    # lookup. It is released after K addressing, before PV accumulation.
    pv_state = heads*(128+2)*4 + sort_ids+ceil((first_tokens+add_tokens)*(heads+1)/8)
    memory = max(scorer_state, phase_one+logits_first, last_control+logits_final,
                 pv_state+logits_final)
    return dict(score_ck=score_cycles,initial_wait_ck=initial_wait,extra_wait_ck=final_wait,
                first_gate_ck=first["gate_ck"],fallback_ck=fallback_ck,
                control_sram_bytes=max(phase_one,last_control),
                gate_only_control_sram_bytes=first["control_sram_bytes"],
                scorer_sram_bytes=scorer_state, logits_sram_bytes=logits_final,
                pv_state_bytes=pv_state, attention_sram_bytes=memory,
                qk_mac=(first_tokens+add_tokens)*heads*128, scorer_mac=h.pages*heads*128)
