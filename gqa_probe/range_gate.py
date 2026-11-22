"""Fixed-K four-cutoff replay with an explicit read-budget fallback."""

from dataclasses import dataclass
from math import ceil

import torch

from .fixed_k_v import burst_round, gather_tokens, masked_output, probabilities, vector_burst
from .lpddr5_banked import read_banked
from .refill_replay import build_phases, full_prefix_bytes, ledger, with_mask, workset
from .round7_gates import gate_mask
from .shared_refill import State
from .vskip_summary import METHOD_BYTES


TAUS = (4., 6., 8., 10.)
LIMITS = (.02, .03, .04)
PHASES = ("summary", "index", "k", "refill_index", "refill_k", "v")


def tier_stats(state):
    """Accumulate Z and the four suffix masses with one exponential per score."""
    valid = state.valid[:, None].expand_as(state.logits)
    flat = state.logits.float().masked_fill(~valid, -torch.inf).flatten(2)
    peak = flat.amax(-1, keepdim=True)
    gaps = torch.where(torch.isfinite(peak), peak, 0.) - flat
    mass = (-gaps).exp()
    optional = valid & ~state.protected[:, None, :, None]
    buckets = torch.bucketize(gaps, gaps.new_tensor(TAUS), right=True)
    tail_mass = mass.new_zeros((*gaps.shape[:2], len(TAUS)))
    tail_mass.scatter_add_(-1, (buckets-1).clamp_min(0),
                           mass * (optional.flatten(2) & (buckets > 0)))
    return gaps.reshape_as(state.logits), tail_mass.flip(-1).cumsum(-1).flip(-1), mass.sum(-1)


def choose_tiers(tails, total, limit):
    """Index of the smallest admissible cutoff. Index four retains all tokens."""
    indices = torch.arange(len(TAUS), device=tails.device)
    valid = tails <= limit * total[..., None]
    return torch.where(valid, indices, len(TAUS)).amin(-1)


def tier_mask(state, gaps, tiers):
    cut = gaps.new_tensor((*TAUS, torch.inf))[tiers]
    optional = state.valid[:, None] & ~state.protected[:, None, :, None]
    drop = ((gaps >= cut[..., None, None]) & (tiers[..., None, None] < len(TAUS)) & optional)
    return state.valid[:, None] & ~drop


@dataclass
class Decision:
    mask: torch.Tensor
    bytes_at_2pct: torch.Tensor
    budget_ok: torch.Tensor
    limit_count: torch.Tensor
    mask_scans: torch.Tensor
    tiers: torch.Tensor | None = None


def _retry_tiers(state, h, cap, zero_index, initial, gaps, tails, mass):
    mask, fit, count, scans, tiers = (
        initial.mask, initial.budget_ok, initial.limit_count, initial.mask_scans, initial.tiers
    )
    for limit in LIMITS[1:]:
        pending = ~fit
        if not pending.any().item():
            break
        count = count + pending
        next_tier = choose_tiers(tails, mass, limit)
        rebuild = pending & (tiers != next_tier).any(-1)
        tiers = torch.where(pending[:, None], next_tier, tiers)
        if rebuild.any().item():
            next_mask = tier_mask(state, gaps, tiers)
            mask = torch.where(rebuild[:, None, None, None], next_mask, mask)
            used, _, _ = ledger(with_mask(state, mask), h, zero_index)
            fit = fit | (rebuild & (used <= cap))
            scans = scans + rebuild
    return Decision(mask, initial.bytes_at_2pct, fit, count, scans, tiers)


def _cost(h, initial, prefix, heads, gate, fallback, limits_tried, mask_scans):
    # Reuse the Round 7 stage/lifetime model for scoring, QK, PV and fallback.
    common = workset(h, initial, 0, prefix, heads, "hist64", fallback,
                     index_mode="direct", first_table=0, final_table=0)
    n = h.tail_size+(initial-1)*h.page_size
    n_optional = (initial-len(h.protected))*h.page_size
    n_prefix = h.tail_size+(prefix-1)*h.page_size
    if gate == "hist64":
        return dict(**common, retry_ck=0, mass_scans=1+int(fallback == "hist64"),
                    gate_exponentials=heads*(n+(n_prefix if fallback == "hist64" else 0)))

    # Same four-score/CK model as hist64. One lane per head at the 4:1
    # workpoint combines emitted bits into the GQA union as each token arrives.
    # Physical burst feasibility is processed at one selected page per CK.
    scan_ck = ceil(heads*n/4)
    tail_ck = ceil(heads*len(TAUS)/4)
    pick_ck = tail_ck+ceil(heads/4)
    gate_ck = scan_ck + 2*tail_ck + pick_ck + scan_ck
    retry_ck = (limits_tried-1)*pick_ck + (mask_scans-1)*(scan_ck+initial)
    shared_mask = ceil(heads*n/8)+ceil(n/8)
    # Four FP32 buckets, FP32 Z, and one chosen cutoff byte per head.
    gate_control = shared_mask + heads*(4*(len(TAUS)+1)+1)
    control = gate_control + 4*initial + 8*heads + 8
    logits = common["logits_sram_bytes"]
    attention = max(common["scorer_sram_bytes"], control+logits, common["pv_state_bytes"]+logits)
    common["initial_wait_ck"] += gate_ck-common["first_gate_ck"]+retry_ck
    common.update(first_gate_ck=gate_ck, control_sram_bytes=control,
                  gate_only_control_sram_bytes=gate_control, attention_sram_bytes=attention)
    return dict(**common, retry_ck=retry_ck, mass_scans=1, gate_exponentials=heads*n,
                gate_cutoff_comparisons=heads*(len(TAUS)*(n_optional+limits_tried)+mask_scans*n))


@torch.inference_mode()
def screen_four_tiers(prepared, ranks, value, dense, *, evidence=()):
    """Replay each gate on identical original logits, using direct page addressing.

    A successful adaptive retry emits no additional mass pass. Changed choices
    incur a new mask/union scan and burst-feasibility pass. An unchanged cutoff
    tuple retains the previous over-budget verdict. A fallback never relaxes
    the selected prefix after the full-read-safe byte bound is applied.
    """
    source, slot_rank, h, cap, _ = prepared
    group_count, heads, page_count, _ = source.logits.shape
    bank_index_bytes = source.ids.new_zeros(group_count)
    v_allowance = cap-burst_round(h.pages*METHOD_BYTES["q6"])-full_prefix_bytes(h,page_count)
    full, tail = vector_burst(h.page_size), vector_burst(h.tail_size)
    safe_prefix = min(page_count, (v_allowance-tail)//full+1)
    if safe_prefix < len(h.protected):
        raise ValueError("fixed-K trace cannot afford all protected V")
    initial = ranks[:, :page_count].cpu().tolist()
    ids = source.ids.cpu().tolist()
    evidence = set(evidence)
    selected_value = gather_tokens(value, source.positions)
    original_probs = probabilities(source.logits)
    ones = torch.ones_like(bank_index_bytes)

    initial_hist = gate_mask(source, "hist64")
    hist_read, _, _ = ledger(with_mask(source,initial_hist), h, bank_index_bytes)
    hist = Decision(initial_hist,hist_read,hist_read<=cap,ones,ones)
    gaps, tails, mass = tier_stats(source)
    tier = choose_tiers(tails,mass,LIMITS[0])
    initial_tier = tier_mask(source,gaps,tier)
    tier_read, _, _ = ledger(with_mask(source,initial_tier), h, bank_index_bytes)
    strict = Decision(initial_tier,tier_read,tier_read<=cap,ones,ones,tier)
    adaptive = _retry_tiers(source,h,cap,bank_index_bytes,strict,gaps,tails,mass)

    rows = []
    for name, kind, fallback, decision in (
            ("hist64_full","hist64","full",hist), ("hist64_hist64","hist64","hist64",hist),
            ("tiers2_full","tiers","full",strict), ("tiers234_full","tiers","full",adaptive)):
        prefix = torch.where(decision.budget_ok,page_count,safe_prefix)
        restricted = source.valid & (slot_rank < prefix[:,None])[...,None]
        prefix_logits = source.logits.masked_fill(~restricted[:,None],-torch.inf)
        if fallback == "hist64":
            subset = State(source.ids,source.positions,prefix_logits,restricted,source.protected,
                           restricted[:,None].expand_as(prefix_logits))
            replacement = gate_mask(subset,"hist64")
        else:
            replacement = restricted[:, None]
        keep = torch.where(decision.budget_ok[:,None,None,None],decision.mask,replacement)
        used, _, v_bytes = ledger(with_mask(source,keep), h, bank_index_bytes)
        union = keep.any(1)
        whole_loss = (original_probs*~keep).sum((2,3))
        prefix_loss = (probabilities(prefix_logits)*~keep).sum((2,3))
        requested = source.logits.new_tensor(LIMITS)[decision.limit_count-1]
        required = (source.valid & source.protected[...,None])[:,None]
        guards = torch.where(decision.budget_ok,requested,LIMITS[0])
        torch._assert_async(((used<=cap).all() & (keep|~required).all() &
                             (prefix_loss<=guards[:,None]+3e-6).all()),"range gate budget or mass violation")
        out = masked_output(source.logits, selected_value, keep)
        error = (out-dense).norm(dim=-1)
        relative = error/(dense.norm(dim=-1)+1e-8)
        tuples = zip(used.tolist(), v_bytes.tolist(), prefix.tolist(), decision.budget_ok.tolist(),
                     decision.bytes_at_2pct.tolist(), decision.limit_count.tolist(),
                     decision.mask_scans.tolist(), keep.sum((1,2,3)).tolist(),
                     whole_loss.tolist(), prefix_loss.tolist(), error.tolist(), relative.tolist(),
                     union.cpu().numpy())
        cutoffs = decision.tiers.tolist() if decision.tiers is not None else [None]*group_count
        for group, (spent, v_used, retained, fit, at_two, limits, scans, retained_head_tokens,
                    all_drop, prefix_drop, abs_error, rel_error, mask) in enumerate(tuples):
            first = initial[group]
            v = {p: bits for p,bits in zip(ids[group],mask) if bits.any()}
            cost = _cost(h,page_count,retained,heads,kind,fallback if not fit else "none",limits,scans)
            phases, waits = build_phases(h,first,[],v,"direct", score_ck=cost["score_ck"],
                                         initial_wait_ck=cost["initial_wait_ck"],
                                         extra_wait_ck=cost["extra_wait_ck"])
            ranges = sorted((addr,addr+size) for phase in phases for addr,size in phase)
            if (any(end>begin for (_,end),(begin,_) in zip(ranges,ranges[1:])) or
                    sum(end-begin for begin,end in ranges)!=spent or
                    sum(size for _,size in phases[-1])!=v_used):
                raise ValueError("range gate read ranges do not match the physical ledger")
            timing = read_banked(phases,waits,queue_depth=32)
            tokens_in_v = sum(int(bits.sum()) for bits in v.values())
            pv_cycles = 8*tokens_in_v
            rows.append(dict(policy=name,group=group,cap=cap,read_bytes=spent,v_bytes=v_used,
                             initial_ids=first,retained_prefix=retained,fallback=not fit,
                             initial_excess_bytes=max(at_two-cap,0),accepted_limit=LIMITS[limits-1] if fit else None,
                             limits_tried=list(LIMITS[:limits]),mask_scans=scans,budget_checks=scans,
                             head_cutoffs=[TAUS[i] if i<len(TAUS) else None for i in cutoffs[group]]
                             if cutoffs[group] is not None else None,
                             v_union_tokens=tokens_in_v,evidence=len(evidence),
                             k_evidence=len(evidence.intersection(first)),
                             v_evidence=len(evidence.intersection(v)),
                             lost_all_k=all_drop,lost_prefix=prefix_drop,
                             abs_error=abs_error,relative_error=rel_error,
                             pv_mac=retained_head_tokens*128,pv_cycles=pv_cycles,
                             banked_first_v_ready=timing["phase_start_ck"][-1],
                             banked_first_v_command=timing["phase_first_command_ck"][-1],
                             banked_bursts=timing["bursts"],banked_cycle=timing["phase_end_ck"][-1]+pv_cycles,
                             phase_ranges=phases,stage_wait_ck=waits,**cost))
    return rows
