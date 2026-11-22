"""Prefix allocation and LPDDR read accounting for packed affine32 pages.

Fixed-size representations use direct stride addressing and a separate region
for the pinned raw pages when their record stride differs. The mixed-length
layout scans a 16-byte directory per logical page before QK. It stores byte
addresses for K, V and segment labels, plus the two record counts. Mixed
segment lengths use two bits per representative, in a separately aligned
plane. Charge the directory scan in every candidate, including fallbacks.
"""
from dataclasses import dataclass
from fractions import Fraction

import torch

from .fixed_k_v import burst_round, probabilities, vector_burst
from .page_read_budget import PageHistory
from .round6_memory import merge, value_runs
from .round7_gates import gate_mask
from .shared_refill import State


DIRECTORY_BASE = 1 << 20
K_BASE = 1 << 22
RAW_K_BASE = 1 << 24
V_BASE = 1 << 25
TAG_BASE = 1 << 26
RAW_V_BASE = 1 << 27
DIRECTORY_RECORD_BYTES = 16


@dataclass(frozen=True)
class BytePlan:
    history: PageHistory
    ids: torch.Tensor
    active: torch.Tensor
    protected: torch.Tensor
    count: torch.Tensor
    safe_count: torch.Tensor
    k_counts: torch.Tensor
    v_counts: torch.Tensor
    tag_bytes: torch.Tensor
    all_k_counts: torch.Tensor
    all_v_counts: torch.Tensor
    all_tag_bytes: torch.Tensor
    summary_bytes: int
    directory_bytes: int
    cap: int
    ragged: bool

    def full_mask(self, prefix=None):
        """Physical V slots for the fetched pages or a shorter rank prefix."""
        valid = torch.arange(self.history.page_size, device=self.ids.device) < self.v_counts[..., None]
        if prefix is None:
            return valid
        lengths = torch.as_tensor(prefix, device=self.ids.device)
        selected = torch.arange(self.ids.shape[1], device=self.ids.device) < lengths[..., None]
        return valid & selected[..., None]


def plan_pages(ranks, history, k_counts, v_counts, *, summary_bytes=144,
               budget_percent=5, gate="full", v_reserve=.65, max_pages=None,
               ragged=False, tags=None):
    """Largest ranked prefix under the common byte allocation rule.

    Counts are [group, physical_page] record counts after preserving protected
    pages. `v_reserve` is the fraction of optional full V reads provisioned
    before QK when hist64 is enabled. Full V and all protected V are charged
    exactly. The safe-prefix bound charges every K and label already read.
    """
    h = history
    if h.page_size != 32 or ranks.ndim != 2 or not ranks.shape[0] or ranks.shape[1] != h.pages:
        raise ValueError("expected a complete [group, P32 page] ranking")
    if ranks.dtype != torch.long or ranks.shape != k_counts.shape or ranks.shape != v_counts.shape:
        raise ValueError("ranking and K/V count tensors must have the same shape")
    if gate not in ("full", "hist64") or not 0 < budget_percent <= 100 or summary_bytes < 0:
        raise ValueError("invalid gate, summary size or budget")
    reserve = Fraction(str(v_reserve)) if gate == "hist64" else Fraction(1)
    if not 0 <= reserve <= 1:
        raise ValueError("V reservation fraction must belong to [0, 1]")
    protected_ids = ranks.new_tensor(h.protected)
    mandatory = protected_ids.numel()
    limit = h.pages if max_pages is None else min(h.pages, max_pages)
    if limit < mandatory:
        raise ValueError("page cap cannot preserve protected pages")
    occupied = torch.zeros_like(ranks, dtype=torch.bool).scatter_(1, ranks.clamp(0, h.pages - 1), True)
    torch._assert_async((occupied.all() & ((ranks >= 0) & (ranks < h.pages)).all()),
                        "ranked page IDs must be unique and cover the history")
    torch._assert_async((ranks[:, :mandatory].sort(-1).values == protected_ids).all(),
                        "protected pages must lead the ranking")
    k_counts, v_counts = k_counts.long(), v_counts.long()
    raw_counts = ranks.new_full((h.pages,), h.page_size)
    raw_counts[-1] = h.tail_size
    compressed = (k_counts != raw_counts) | (v_counts != raw_counts)
    torch._assert_async(((k_counts >= 1) & (v_counts >= 1) &
                         (k_counts <= raw_counts) & (v_counts <= raw_counts)).all(),
                        "invalid physical K/V record counts")
    torch._assert_async((~compressed[:, protected_ids]).all(), "protected pages must remain raw")
    if not ragged and mandatory < h.pages:
        optional = torch.ones_like(raw_counts, dtype=torch.bool)
        optional[protected_ids] = False
        first_optional = ranks[:, mandatory:mandatory+1]
        torch._assert_async((~optional | ((k_counts == k_counts.gather(1, first_optional)) &
                                         (v_counts == v_counts.gather(1, first_optional)))).all(),
                            "fixed stride requires a uniform historical record count")

    if tags is not None and (not ragged or tags.shape != ranks.shape):
        raise ValueError("segment label counts require the ragged layout")
    tag_payload = ((v_counts + 3) // 4).masked_fill(~compressed, 0) if tags is None else tags.long()
    torch._assert_async(((tag_payload >= 0).all() & (tag_payload[:,protected_ids] == 0).all()),
                        "invalid segment label sizes")
    all_tag_bytes = burst_round(tag_payload) if ragged else torch.zeros_like(k_counts)
    summary = burst_round(h.pages * summary_bytes)
    directory = burst_round(DIRECTORY_RECORD_BYTES * h.pages) if ragged else 0
    metadata = summary + directory
    dense_bytes = 2 * ((h.pages - 1) * vector_burst(h.page_size) + vector_burst(h.tail_size))
    fraction = Fraction(str(budget_percent)) / 100
    cap = dense_bytes * fraction.numerator // fraction.denominator

    ids = ranks[:, :limit]
    k = k_counts.gather(1, ids)
    v = v_counts.gather(1, ids)
    lengths = all_tag_bytes.gather(1, ids)
    k_total = (vector_burst(k) + lengths).cumsum(1)
    v_total = vector_burst(v).cumsum(1)
    protect_v = v_total[:, mandatory - 1:mandatory]
    optional_v = (v_total - protect_v).clamp_min(0)
    # Integer ceiling keeps a fractional reservation conservative by <1 B;
    # the post-QK ledger always uses whole physical bursts.
    reserved = (optional_v * reserve.numerator + reserve.denominator - 1) // reserve.denominator
    count = (metadata + k_total + protect_v + reserved <= cap).sum(1)
    torch._assert_async((count >= mandatory).all(), "budget cannot preserve protected pages")
    fetched_k = k_total.gather(1, (count.clamp_min(1) - 1)[:, None])
    active = torch.arange(limit, device=ranks.device) < count[:, None]
    safe_count = (active & (metadata + fetched_k + v_total <= cap)).sum(1)
    torch._assert_async((safe_count >= mandatory).all(), "no safe V prefix after K read")

    width = int(count.max().item())
    active = active[:, :width]
    protected = (torch.arange(width, device=ranks.device) < mandatory).expand_as(active)
    return BytePlan(h, ids[:, :width], active, protected, count, safe_count,
                    k[:, :width].masked_fill(~active, 0), v[:, :width].masked_fill(~active, 0),
                    lengths[:, :width].masked_fill(~active, 0), k_counts, v_counts,
                    all_tag_bytes, summary, directory, cap, ragged)


def physical_union(plan, mask):
    """Collapse Q heads onto the shared packed V records."""
    union = mask.any(1) if mask.ndim == 4 else mask
    full = plan.full_mask()
    if union.dtype != torch.bool or union.shape != full.shape:
        raise ValueError("V mask must use [group, optional head, page slot, 32 record slots]")
    torch._assert_async((~union | full).all(), "V mask includes an invalid record")
    return union


def value_ledger(plan, mask):
    """Physical read lengths, including shared packed scale/zero-point bursts."""
    union = physical_union(plan, mask)
    k_records = plan.k_counts.sum(1)
    v_records = union.sum((1, 2))
    k_codes, v_codes = 64 * k_records, 64 * v_records
    k_scale = burst_round(plan.k_counts * 8).sum(1)
    k_zero = burst_round(plan.k_counts * 4).sum(1)
    v_scale = 32 * union.unflatten(-1, (8, 4)).any(-1).sum((1, 2))
    v_zero = 32 * union.unflatten(-1, (4, 8)).any(-1).sum((1, 2))
    k_bytes = k_codes + k_scale + k_zero
    v_bytes = v_codes + v_scale + v_zero
    labels = plan.tag_bytes.sum(1)
    summary = torch.full_like(k_bytes, plan.summary_bytes)
    directory = torch.full_like(k_bytes, plan.directory_bytes)
    total = summary + directory + k_bytes + labels + v_bytes
    return dict(total=total, over_bytes=(total - plan.cap).clamp_min(0), summary_bytes=summary,
                directory_bytes=directory, tag_bytes=labels, k_bytes=k_bytes, v_bytes=v_bytes,
                k_records=k_records, v_records=v_records, k_code_bytes=k_codes,
                v_code_bytes=v_codes, k_scale_bytes=k_scale, v_scale_bytes=v_scale,
                k_zero_point_bytes=k_zero, v_zero_point_bytes=v_zero)


def _byte_addresses(counts, h, base, pinned, ragged):
    if ragged:
        pos = base
        addresses = []
        for count in counts:
            addresses.append(pos)
            pos += vector_burst(count)
        return addresses
    protected = {page: pos for pos, page in enumerate(h.protected)}
    optional = {n for p, n in enumerate(counts) if p not in protected}
    if len(optional) > 1:
        raise ValueError("fixed stride requires a uniform historical record count")
    stride_count = next(iter(optional), h.page_size)
    stride = vector_burst(stride_count)
    if stride_count == h.page_size:
        return [base + p * stride for p in range(h.pages)]
    return [pinned + protected[p] * vector_burst(h.page_size) if p in protected else base + p * stride
            for p in range(h.pages)]


def read_trace(plan, mask, group=0):
    """Address phases as [summary, directory, fetched K and labels, retained V]."""
    h = plan.history
    group_mask = physical_union(plan, mask)[group].detach().cpu().numpy()
    k_counts = plan.all_k_counts[group].detach().cpu().tolist()
    v_counts = plan.all_v_counts[group].detach().cpu().tolist()
    ids = plan.ids[group, :int(plan.count[group].item())].detach().cpu().tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fetched page")
    k_addr = _byte_addresses(k_counts, h, K_BASE, RAW_K_BASE, plan.ragged)
    v_addr = _byte_addresses(v_counts, h, V_BASE, RAW_V_BASE, plan.ragged)
    labels = plan.all_tag_bytes[group].detach().cpu().tolist()
    label_addr = []
    pos = TAG_BASE
    for n in labels:
        label_addr.append(pos)
        pos += n
    k_runs = [(k_addr[p], vector_burst(k_counts[p])) for p in ids]
    k_runs += [(label_addr[p], labels[p]) for p in ids if labels[p]]
    v_runs = []
    for slot, page in sorted(enumerate(ids), key=lambda pair: pair[1]):
        v_runs.extend(value_runs(group_mask[slot], v_counts[page], v_addr[page]))
    return [merge([(0, plan.summary_bytes)]), merge([(DIRECTORY_BASE, plan.directory_bytes)]),
            merge(k_runs), merge(v_runs)]


@dataclass(frozen=True)
class GateBudgetResult:
    mask: torch.Tensor
    fallback: torch.Tensor
    prefix: torch.Tensor
    dropped_all: torch.Tensor
    dropped_prefix: torch.Tensor


def select_gate_mask(plan, state, *, fallback="hist64", epsilon=.02):
    """Apply hist64 and roll back to the safe prefix for aligned K/V slots."""
    if fallback not in ("full", "hist64"):
        raise ValueError("fallback must be full or hist64")
    if state.ids.shape != plan.ids.shape or state.valid.shape != plan.full_mask().shape:
        raise ValueError("unaligned route state")
    torch._assert_async(((plan.k_counts == plan.v_counts).all() &
                         (state.ids == plan.ids).all() &
                         (state.valid == plan.full_mask()).all() &
                         (state.protected == plan.protected).all()),
                        "hist64 requires aligned K and V records")
    mask = gate_mask(state, "hist64", epsilon=epsilon)
    over = value_ledger(plan, mask)["over_bytes"] > 0
    prefix = torch.where(over, plan.safe_count, plan.count)
    valid = plan.full_mask(prefix)
    logits = state.logits.masked_fill(~valid[:, None], -torch.inf)
    if over.any().item():
        safe = State(state.ids, state.positions, logits, valid, state.protected, valid[:, None])
        replacement = gate_mask(safe, fallback, epsilon=epsilon)
        mask = torch.where(over[:, None, None, None], replacement, mask)
    dropped_all = (probabilities(state.logits) * ~mask).sum((2, 3))
    dropped_prefix = (probabilities(logits) * ~mask).sum((2, 3))
    spent = value_ledger(plan, mask)["total"]
    forced = (valid & plan.protected[..., None])[:, None]
    torch._assert_async(((spent <= plan.cap).all() & (mask | ~forced).all() &
                         (dropped_prefix <= epsilon + 3e-6).all()),
                        "hist64 violated budget, protection or the prefix mass constraint")
    return GateBudgetResult(mask, over, prefix, dropped_all, dropped_prefix)
