"""Raw affine32 decode reference with common physical accounting at P32 and P64."""
from dataclasses import dataclass
from fractions import Fraction

import torch

from .fixed_k_v import burst_round, candidate_qk, gather_tokens, masked_output, probabilities, vector_burst
from .page_read_budget import PageHistory
from .pagepack_budget import K_BASE, V_BASE
from .round6_memory import merge, value_runs
from .round7_gates import gate_mask
from .shared_refill import State


@dataclass(frozen=True)
class SummaryPlan:
    history: PageHistory
    ids: torch.Tensor
    active: torch.Tensor
    protected: torch.Tensor
    count: torch.Tensor
    safe_count: torch.Tensor
    k_counts: torch.Tensor
    cap: int | None
    summary_bytes: int
    summary_payload_bytes: int
    dense_bytes: int
    gate: str

    def full_mask(self, prefix=None):
        valid = torch.arange(self.history.page_size, device=self.ids.device) < self.k_counts[..., None]
        if prefix is None:
            return valid
        slots = torch.arange(self.ids.shape[1], device=self.ids.device) < prefix[..., None]
        return valid & slots[..., None]


def plan(ranks, history, *, summary_bytes=144, budget_percent=5, v_reserve=.65,
         max_pages=None, max_tokens=None, gate="hist64"):
    """Allocate the longest rank prefix after prepaying summary and protected V.

    The complete ranking begins with the physical protection pages. The optional
    page and logical token limits include protection. Use ``budget_percent=None``
    for a fixed-size diagnostic outside the closed-loop budget comparison.
    """
    h = history
    if (h.page_size not in (32, 64) or ranks.ndim != 2 or not ranks.shape[0] or
            ranks.shape[1] != h.pages or ranks.dtype != torch.long):
        raise ValueError("expected a complete [group, page] P32 or P64 rank tensor")
    if gate not in ("full", "hist64") or not isinstance(summary_bytes, int) or summary_bytes < 0:
        raise ValueError("invalid V gate or per-page summary bytes")
    if budget_percent is None and max_pages is None and max_tokens is None:
        raise ValueError("an unbudgeted diagnostic requires a page or token limit")
    if budget_percent is not None and not 0 < budget_percent <= 100:
        raise ValueError("invalid budget percentage")
    reserve = Fraction(str(v_reserve)) if gate == "hist64" else Fraction(1)
    if not 0 <= reserve <= 1:
        raise ValueError("V reservation fraction must belong to [0, 1]")

    pin_ids = ranks.new_tensor(h.protected)
    mandatory = len(h.protected)
    pinned_tokens = (mandatory - 1) * h.page_size + h.tail_size
    limit = h.pages if max_pages is None else min(h.pages, max_pages)
    if max_tokens is not None:
        limit = min(limit, mandatory + (max_tokens - pinned_tokens) // h.page_size)
    if limit < mandatory:
        raise ValueError("candidate limit cannot preserve protected pages")
    occupied = torch.zeros_like(ranks, dtype=torch.bool).scatter_(1, ranks.clamp(0, h.pages - 1), True)
    torch._assert_async((occupied.all() & ((ranks >= 0) & (ranks < h.pages)).all()),
                        "ranked page IDs must be unique and cover the history")
    torch._assert_async((ranks[:, :mandatory].sort(-1).values == pin_ids).all(),
                        "protected pages must lead the ranking")

    full = vector_burst(h.page_size)
    tail = vector_burst(h.tail_size)
    dense_bytes = 2 * ((h.pages - 1) * full + tail)
    payload = h.pages * summary_bytes
    paid_summary = burst_round(payload)
    if budget_percent is None:
        cap = None
        width = safe = limit
    else:
        fraction = Fraction(str(budget_percent)) / 100
        cap = dense_bytes * fraction.numerator // fraction.denominator
        pinned_v = (mandatory - 1) * full + tail
        spare = cap - paid_summary - 2 * pinned_v
        if spare < 0:
            raise ValueError("budget cannot preserve protected pages")
        # Every optional page is complete because Local protects the tail.
        # Floor here enforces the same upper bound as ceiling the V reserve.
        width = min(limit, mandatory + spare * reserve.denominator //
                    (full * (reserve.denominator + reserve.numerator)))
        paid_k = pinned_v + (width - mandatory) * full
        safe = min(width, mandatory + (cap - paid_summary - paid_k - pinned_v) // full)
    ids = ranks[:, :width]
    active = torch.ones_like(ids, dtype=torch.bool)
    protected = (torch.arange(width, device=ranks.device) < mandatory).expand_as(active)
    k_counts = ids.new_full(ids.shape, h.page_size).masked_fill(ids == h.pages - 1, h.tail_size)
    return SummaryPlan(h, ids, active, protected, ranks.new_full((ranks.shape[0],), width),
                       ranks.new_full((ranks.shape[0],), safe), k_counts, cap,
                       paid_summary, payload, dense_bytes, gate)


def physical_union(plan, mask):
    """Map per-head PV masks to potentially shared V reads."""
    union = mask.any(1) if mask.ndim == 4 else mask
    valid = plan.full_mask()
    if union.dtype != torch.bool or union.shape != valid.shape:
        raise ValueError("V mask does not match the selected pages and physical slot width")
    torch._assert_async((~union | valid).all(), "V mask includes an invalid record")
    return union


def value_ledger(plan, union):
    """Account for code, scale and zero-point bursts and the shared summary scan."""
    k_records = plan.k_counts.sum(1)
    v_records = union.sum((1, 2))
    k_code, v_code = 64 * k_records, 64 * v_records
    k_scale = burst_round(plan.k_counts * 8).sum(1)
    k_zero = burst_round(plan.k_counts * 4).sum(1)
    v_scale = 32 * union.unflatten(-1, (-1, 4)).any(-1).sum((1, 2))
    v_zero = 32 * union.unflatten(-1, (-1, 8)).any(-1).sum((1, 2))
    k_bytes = k_code + k_scale + k_zero
    v_bytes = v_code + v_scale + v_zero
    summary = torch.full_like(k_bytes, plan.summary_bytes)
    zero = torch.zeros_like(k_bytes)
    total = summary + k_bytes + v_bytes
    return dict(total=total, over_bytes=zero if plan.cap is None else (total - plan.cap).clamp_min(0),
                directory_bytes=zero, tag_bytes=zero, index_bytes=zero,
                summary_bytes=summary, summary_payload_bytes=torch.full_like(summary, plan.summary_payload_bytes),
                summary_alignment_bytes=summary - plan.summary_payload_bytes,
                k_bytes=k_bytes, v_bytes=v_bytes, k_records=k_records, v_records=v_records,
                k_code_bytes=k_code, v_code_bytes=v_code, k_scale_bytes=k_scale, v_scale_bytes=v_scale,
                k_zero_point_bytes=k_zero, v_zero_point_bytes=v_zero,
                k_alignment_bytes=k_scale + k_zero - 12 * k_records,
                v_alignment_bytes=v_scale + v_zero - 12 * v_records)


def read_trace(plan, mask, *, group=0):
    """Replay phases in dependency order as [summary, K, V] burst ranges."""
    h = plan.history
    stride = vector_burst(h.page_size)
    ids = plan.ids[group].detach().tolist()
    counts = plan.k_counts[group].detach().tolist()
    union = physical_union(plan, mask)[group].detach().cpu().numpy()
    k = [(K_BASE + page * stride, vector_burst(count)) for page, count in zip(ids, counts)]
    v = []
    for slot in sorted(range(len(ids)), key=ids.__getitem__):
        v.extend(value_runs(union[slot], counts[slot], V_BASE + ids[slot] * stride))
    return [merge([(0, plan.summary_bytes)]), merge(k), merge(v)]


def _evidence(plan, positions, valid, union, initial_mask, prefix, spans):
    gold_tokens = torch.zeros_like(valid)
    gold_page_ids = set()
    gold_token_ids = set()
    for start, end in spans:
        start, end = max(0, int(start)), min(plan.history.tokens, int(end))
        if start >= end:
            continue
        gold_tokens |= (positions >= start) & (positions < end)
        gold_page_ids.update(range(start // plan.history.page_size, (end - 1) // plan.history.page_size + 1))
        gold_token_ids.update(range(start, end))
    gold_tokens &= valid
    gold_pages = gold_tokens.any(-1)
    live_pages = plan.full_mask(prefix).any(-1)
    return dict(gold_pages=len(gold_page_ids), gold_tokens=len(gold_token_ids),
                k_page_hits=gold_pages.sum(-1), k_gold_token_hits=gold_tokens.sum((1, 2)),
                final_k_page_hits=(gold_pages & live_pages).sum(-1),
                initial_v_gold_token_hits=(initial_mask.any(1) & gold_tokens).sum((1, 2)),
                v_page_hits=(gold_pages & union.any(-1)).sum(-1),
                v_evidence_page_hits=(union & gold_tokens).any(-1).sum(-1),
                v_gold_token_hits=(union & gold_tokens).sum((1, 2)))


@dataclass(frozen=True)
class SummaryScreen:
    output: torch.Tensor
    logits: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    initial_mask: torch.Tensor
    final_mask: torch.Tensor
    union: torch.Tensor
    fallback: torch.Tensor
    prefix: torch.Tensor
    ledger: dict
    evidence: dict
    loss_read_k: torch.Tensor
    loss_prefix: torch.Tensor


def screen(plan, query, keys, values, scaling, *, gold_spans=(), epsilon=.02):
    """Run same-state attention and rerun hist64 on the safe prefix if over cap."""
    query = query.float()
    logits, positions, valid, pins = candidate_qk(query, keys, plan.ids, plan.history, scaling, plan.active)
    state = State(plan.ids, positions, logits, valid, pins, valid[:, None])
    initial = gate_mask(state, plan.gate, epsilon=epsilon)
    mask = initial
    union = physical_union(plan, mask)
    ledger = value_ledger(plan, union)
    fallback = ledger["over_bytes"] > 0
    prefix = plan.count
    prefix_logits = logits
    prefix_valid = valid
    if plan.cap is not None and plan.gate == "hist64" and fallback.any().item():
        prefix = torch.where(fallback, plan.safe_count, plan.count)
        prefix_valid = plan.full_mask(prefix)
        prefix_logits = logits.masked_fill(~prefix_valid[:, None], -torch.inf)
        replacement = gate_mask(State(plan.ids, positions, prefix_logits, prefix_valid, pins,
                                      prefix_valid[:, None]), plan.gate, epsilon=epsilon)
        mask = torch.where(fallback[:, None, None, None], replacement, initial)
        union = physical_union(plan, mask)
        ledger = value_ledger(plan, union)
    lost_all = (probabilities(logits) * ~mask).sum((2, 3))
    lost_prefix = lost_all if prefix_logits is logits else (probabilities(prefix_logits) * ~mask).sum((2, 3))
    forced = (pins[..., None] & valid)[:, None]
    torch._assert_async(((ledger["over_bytes"] == 0).all() & (mask | ~forced).all() &
                         (lost_prefix <= epsilon + 3e-6).all()),
                        "decode violated the budget, protection or prefix mass constraint")
    output = masked_output(logits, gather_tokens(values, positions), mask)
    evidence = _evidence(plan, positions, valid, union, initial, prefix, gold_spans)
    return SummaryScreen(output, logits, positions, valid, initial, mask, union, fallback, prefix,
                         ledger, evidence, lost_all, lost_prefix)
