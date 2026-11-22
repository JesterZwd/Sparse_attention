"""Single-batch K reservation and checked physical V issuance."""
from dataclasses import dataclass
from functools import lru_cache

import torch

from .fixed_k_v import burst_round, candidate_qk, probabilities, vector_burst
from .page_read_budget import page_history
from .round6_attention import physical_ledger
from .round7_gates import gate_mask
from .shared_refill import State
from .vskip_summary import METHOD_BYTES


@lru_cache(maxsize=4096)
def single_quota(tokens, method, budget_bp, max_pages=68):
    """Return the read count and a safe full-V prefix, charging all K first."""
    h = page_history(tokens, 32)
    full, tail = vector_burst(32), vector_burst(h.tail_size)
    cap = 2 * ((h.pages-1)*full + tail) * budget_bp // 10000
    summary = burst_round(h.pages * METHOD_BYTES[method])
    mandatory = (len(h.protected)-1)*full + tail
    for count in range(min(h.pages, max_pages), len(h.protected)-1, -1):
        k_bytes = (count-1)*full + tail
        v_budget = cap - summary - burst_round(4*count) - k_bytes
        if v_budget >= mandatory:
            return h, cap, count, min(count, (v_budget-tail)//full+1)
    raise ValueError("hard cap cannot reserve protected K and V")


def prepare_single(q, key, ranks, tokens, scaling, method, budget_bp, max_pages):
    h, cap, count, safe_prefix = single_quota(tokens, method, budget_bp, max_pages)
    ids, perm = ranks[:, :count].sort(-1)
    logits, positions, valid, protected = candidate_qk(q, key, ids, h, scaling)
    return (State(ids, positions, logits, valid, protected, valid[:, None].expand_as(logits)),
            perm, h, cap, safe_prefix)


@dataclass
class SingleOutcome:
    first: State
    final: State
    initial_count: int
    additional: torch.Tensor
    ceiling: int
    fallback: torch.Tensor
    prefix_pages: torch.Tensor
    lost_all: torch.Tensor
    lost_prefix: torch.Tensor
    initial_over_bytes: torch.Tensor


def decide_v(prepared, method, kind, threshold):
    source, slot_rank, h, cap, safe_prefix = prepared
    order = slot_rank.argsort(-1)
    mask = gate_mask(source, kind, threshold, order)
    first = State(source.ids, source.positions, source.logits, source.valid, source.protected, mask)
    groups = torch.ones_like(source.ids[:, 0], dtype=torch.bool)
    over = (physical_ledger(first, h, method, groups)["total"] - cap).clamp_min(0)
    fallback = over > 0
    prefix = torch.where(fallback, safe_prefix, source.ids.shape[-1])
    prefix_valid = source.valid & (slot_rank < prefix[:, None])[..., None]
    prefix_logits = source.logits.masked_fill(~prefix_valid[:, None], -torch.inf)
    if fallback.any().item():
        restricted = State(source.ids, source.positions, prefix_logits, prefix_valid,
                           source.protected, source.mask & prefix_valid[:, None])
        replacement = gate_mask(restricted, "exact")
        mask = torch.where(fallback[:, None, None, None], replacement, mask)
    final = State(source.ids, source.positions, source.logits, source.valid, source.protected, mask)
    physical = physical_ledger(final, h, method, groups)
    mandatory = (source.valid & source.protected[..., None])[:, None]
    torch._assert_async(((physical["total"] <= cap).all() & (mask | ~mandatory).all()),
                        "budget or protected V violation before V issue")
    all_mass = (probabilities(source.logits) * ~mask).sum((2, 3))
    prefix_mass = (probabilities(prefix_logits) * ~mask).sum((2, 3))
    if kind in ("exact", "hist64", "checked"):
        torch._assert_async((prefix_mass <= .020003).all(), "V prefix guard violated")
    else:
        torch._assert_async(((prefix_mass <= .020003) | ~fallback[:, None]).all(),
                            "exact fallback prefix guard violated")
    return SingleOutcome(first, final, source.ids.shape[-1], torch.zeros_like(over), cap,
                         fallback, prefix, all_mass, prefix_mass, over)


def run_single(q, key, ranks, tokens, scaling, *, method="q6", budget_bp=1000,
               max_pages=68, kind="exact", threshold=0.):
    batch = prepare_single(q, key, ranks, tokens, scaling, method, budget_bp, max_pages)
    return decide_v(batch, method, kind, threshold)
