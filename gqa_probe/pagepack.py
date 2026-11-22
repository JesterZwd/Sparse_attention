"""Consecutive page-local K/V representatives and shared Mean+Radius stats."""
from dataclasses import dataclass
from math import isfinite

import torch

from .quantization_round2 import affine_quantize_dequantize
from .vskip_summary import encode


MODES = ("raw", "v2", "v4", "kv2", "kv4", "mixed")


def _lengths(x, lengths):
    if x.ndim != 4 or not x.shape[-2] or x.shape[-2] % 4:
        raise ValueError("expected [G, pages, P, D] with positive P divisible by 4")
    return torch.as_tensor(lengths, dtype=torch.long, device=x.device).broadcast_to(x.shape[:2])


def _valid(lengths, page_size):
    return torch.arange(page_size, device=lengths.device) < lengths[..., None]


@dataclass(frozen=True)
class SegmentStatistics:
    """Pair statistics. Quartets and page records combine these same pairs."""

    sums: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    count: torch.Tensor


def _pair_sums(x, valid):
    pair = x.float().unflatten(-2, (-1, 2))
    pair_valid = valid.unflatten(-1, (-1, 2))
    return pair.masked_fill(~pair_valid[..., None], 0).sum(-2), pair_valid.sum(-1)


def segment_statistics(k, lengths):
    """Use the original affine32 dequantizations, including partial pairs."""
    valid = _valid(_lengths(k, lengths), k.shape[-2])
    k = k.float()
    sums, count = _pair_sums(k, valid)
    pair = k.unflatten(-2, (-1, 2))
    missing = ~valid.unflatten(-1, (-1, 2))[..., None]
    return SegmentStatistics(sums, pair.masked_fill(missing, torch.inf).amin(-2),
                             pair.masked_fill(missing, -torch.inf).amax(-2), count)


def select_statistics(stats, ids):
    """Gather cached sealed statistics using per-group physical page IDs."""
    return SegmentStatistics(*(_select(t, ids) for t in
                               (stats.sums, stats.minimum, stats.maximum, stats.count)))


def summary_pages(k, lengths, *, stats=None, method="mr25"):
    """Encode full-page statistics without taking extrema over means."""
    stats = segment_statistics(k, lengths) if stats is None else stats
    counts = stats.count.sum(-1, keepdim=True)
    sums = stats.sums.sum(-2)
    empty = counts == 0
    lo = stats.minimum.amin(-2).masked_fill(empty, 0)
    hi = stats.maximum.amax(-2).masked_fill(empty, 0)
    record = encode(sums, lo, hi, counts.clamp_min(1), method)
    return (*record[:3], counts.to(torch.uint8)) if method.startswith("mr") else record


def _quartets(s):
    return SegmentStatistics(s.sums.unflatten(-2, (-1, 2)).sum(-2),
                             s.minimum.unflatten(-2, (-1, 2)).amin(-2),
                             s.maximum.unflatten(-2, (-1, 2)).amax(-2),
                             s.count.unflatten(-1, (-1, 2)).sum(-1))


def _acceptable(stats, threshold, width):
    span = stats.maximum - stats.minimum
    mean = stats.sums / stats.count.clamp_min(1)[..., None]
    return (stats.count == width) & (
        span.norm(dim=-1) <= threshold * mean.norm(dim=-1).clamp_min(1e-12))


def _mixed_starts(k_stats, v_stats, k_threshold, v_threshold):
    pair = _acceptable(k_stats, k_threshold, 2)
    quartet = _acceptable(_quartets(k_stats), k_threshold, 4)
    if v_stats is not None:
        pair &= _acceptable(v_stats, v_threshold, 2)
        quartet &= _acceptable(_quartets(v_stats), v_threshold, 4)
    # Within a 4-token block, accepted quartets dominate the pair decision.
    pair_starts = torch.stack((1 + pair.long(), (~pair).long()), dim=-1).flatten(-2)
    return torch.where(quartet[..., None],
                       pair.new_tensor([4, 0, 0, 0], dtype=torch.long),
                       pair_starts.unflatten(-1, (-1, 4))).flatten(-2)


@dataclass(frozen=True)
class SegmentPlan:
    lengths: torch.Tensor
    key_lengths: torch.Tensor
    value_lengths: torch.Tensor
    segment_starts: torch.Tensor
    value_map: torch.Tensor
    q_to_v: torch.Tensor
    mode: str


def plan_segments(k, v, lengths, mode, *, forced=None, k_threshold=.5, v_threshold=None,
                  key_stats=None, value_stats=None):
    """Resolve a physical layout, leaving the protected and partial pages raw."""
    if mode not in MODES or k.shape != v.shape:
        raise ValueError("unknown packing mode or mismatched K/V shape")
    lengths = _lengths(k, lengths)
    page_size = k.shape[-2]
    valid = _valid(lengths, page_size)
    pos = torch.arange(page_size, device=k.device)
    must = lengths < page_size
    if forced is not None:
        must = must | torch.as_tensor(forced, device=k.device, dtype=torch.bool)
    if mode == "mixed":
        thresholds = [k_threshold] if v_threshold is None else [k_threshold, v_threshold]
        if any(not isfinite(x) or x < 0 for x in thresholds):
            raise ValueError("mixed thresholds must be finite and nonnegative")
        if key_stats is None:
            key_stats = segment_statistics(k, lengths)
        if v_threshold is not None and value_stats is None:
            value_stats = segment_statistics(v, lengths)
        start_lengths = _mixed_starts(key_stats, value_stats if v_threshold is not None else None,
                                      k_threshold, v_threshold)
    else:
        width = 1 if mode == "raw" else int(mode[-1])
        start_lengths = (pos % width == 0).long() * width
    start_lengths = torch.where(must[..., None], 1, start_lengths)
    start_lengths = start_lengths.masked_fill(~valid, 0)
    real = start_lengths > 0
    value_map = (real.long().cumsum(-1)-1).clamp_min(0)
    slot = torch.zeros_like(value_map)
    value_lengths = slot.scatter_reduce(-1, value_map, start_lengths, reduce="amax")
    starts = slot.scatter_reduce(-1, value_map, torch.where(real, pos, 0), reduce="amax")
    value_map = value_map.masked_fill(~valid, 0)
    v_only = mode in ("v2", "v4")
    return SegmentPlan(lengths, valid.long() if v_only else value_lengths, value_lengths, starts,
                       value_map, value_map if v_only else pos.expand_as(value_map), mode)


def _select(x, ids):
    view = ids.view(*ids.shape, *([1] * (x.ndim - 2)))
    return x.gather(1, view.expand(*ids.shape, *x.shape[2:]))


def select_plan(plan, ids, *, active=None, forced=None):
    """Gather the selected plan and optionally restore current protection."""
    lengths = _select(plan.lengths, ids)
    key_lengths = _select(plan.key_lengths, ids)
    value_lengths = _select(plan.value_lengths, ids)
    starts = _select(plan.segment_starts, ids)
    value_map = _select(plan.value_map, ids)
    q_to_v = _select(plan.q_to_v, ids)
    if forced is not None:
        pos = torch.arange(key_lengths.shape[-1], device=ids.device)
        raw = forced[..., None]
        unit = _valid(lengths, key_lengths.shape[-1]).long()
        key_lengths = torch.where(raw, unit, key_lengths)
        value_lengths = torch.where(raw, unit, value_lengths)
        starts = torch.where(raw, pos * unit, starts)
        value_map = torch.where(raw, pos * unit, value_map)
        q_to_v = torch.where(raw, pos * unit, q_to_v)
    if active is not None:
        lengths = lengths.masked_fill(~active, 0)
        mask = ~active[..., None]
        key_lengths, value_lengths, starts, value_map, q_to_v = (
            x.masked_fill(mask, 0) for x in (key_lengths, value_lengths, starts, value_map, q_to_v))
    return SegmentPlan(lengths, key_lengths, value_lengths, starts, value_map, q_to_v, plan.mode)


@dataclass(frozen=True)
class Affine32:
    code: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

    def decode(self, dtype=torch.float32):
        lo, hi = self.code & 15, self.code >> 4
        data = torch.stack((lo, hi), dim=-1).flatten(-2).unflatten(-1, (-1, 32)).float()
        return ((data - self.zero_point[..., None].float()) *
                self.scale[..., None].float()).flatten(-2).to(dtype)


def _encode_affine32(x):
    groups = x.float().unflatten(-1, (-1, 32))
    lo = groups.amin(-1).clamp_max(0)
    hi = groups.amax(-1).clamp_min(0)
    scale = ((hi-lo)/15).clamp_min(2**-24).half()
    steps = scale.float()[..., None]
    zero = (-lo / scale.float()).round().clamp(0,15).to(torch.uint8)
    code = ((groups/steps).round() + zero[..., None].float()).clamp(0,15).to(torch.uint8)
    code = code.flatten(-2)
    return Affine32(code[..., 0::2] | (code[..., 1::2] << 4), scale, zero)


def _gather_vectors(x, starts):
    return x.gather(-2,starts[..., None].expand(*starts.shape, x.shape[-1]))


def _packed_vectors(x, plan, slots, *, compressed, stats, quantized, with_codes):
    if not compressed:
        return x.float().masked_fill(~(slots > 0)[..., None], 0), None
    valid = _valid(plan.lengths, x.shape[-2])
    sums, counts = (stats.sums, stats.count) if stats is not None else _pair_sums(x, valid)
    pair = sums / counts.clamp_min(1)[..., None]
    quad = sums.unflatten(-2, (-1, 2)).sum(-2)
    quad /= counts.unflatten(-1, (-1, 2)).sum(-1).clamp_min(1)[..., None]
    starts = plan.segment_starts
    rows = _gather_vectors(x.float(), starts)
    rows = torch.where((slots == 2)[..., None], _gather_vectors(pair, starts//2), rows)
    rows = torch.where((slots == 4)[..., None], _gather_vectors(quad, starts//4), rows)
    rows = rows.masked_fill(~(slots > 0)[..., None], 0)
    if not quantized:
        return rows, None
    merge = slots > 1
    representatives = rows[merge]
    if not with_codes:
        if representatives.numel():
            rows[merge] = affine_quantize_dequantize(representatives).to(x.dtype).float()
        return rows, None
    packed = _encode_affine32(representatives)
    rows[merge] = packed.decode(x.dtype).float()
    planes = []
    for source in (packed.code, packed.scale, packed.zero_point):
        out = source.new_zeros((*slots.shape, source.shape[-1]))
        out[merge] = source
        planes.append(out)
    return rows, Affine32(*planes)


@dataclass(frozen=True)
class PackedPages(SegmentPlan):
    key: torch.Tensor
    value: torch.Tensor
    summary: tuple | None
    key_codes: Affine32 | None
    value_codes: Affine32 | None


def pack_pages(k, v, lengths, mode, *, forced=None, quantized=False, k_threshold=.5, v_threshold=None,
               plan=None, with_summary=True, key_stats=None, value_stats=None, with_codes=False):
    """Pack selected pages; padded rows never become physical read records."""
    if mode not in MODES or k.shape != v.shape:
        raise ValueError("unknown packing mode or mismatched K/V shape")
    lengths = _lengths(k, lengths)
    if key_stats is None and (with_summary or (plan is None and mode == "mixed")):
        key_stats = segment_statistics(k, lengths)
    if value_stats is None and plan is None and mode == "mixed" and v_threshold is not None:
        value_stats = segment_statistics(v, lengths)
    if plan is None:
        plan = plan_segments(k, v, lengths, mode, forced=forced, k_threshold=k_threshold,
                             v_threshold=v_threshold, key_stats=key_stats, value_stats=value_stats)
    elif plan.mode != mode:
        raise ValueError("packing mode differs from supplied plan")
    opts = dict(quantized=quantized, with_codes=with_codes)
    key, kc = _packed_vectors(k, plan, plan.key_lengths, compressed=mode in ("mixed", "kv2", "kv4"),
                              stats=key_stats, **opts)
    value, vc = _packed_vectors(v, plan, plan.value_lengths, compressed=mode != "raw",
                                stats=value_stats, **opts)
    summary = summary_pages(k, lengths, stats=key_stats) if with_summary else None
    return PackedPages(**vars(plan),key=key,value=value,summary=summary,key_codes=kc,value_codes=vc)


def packed_logits(query, packed, scale):
    """Per-head record logits; log n corrects each joint segment's mass."""
    logits = (torch.bmm(query.float(),packed.key.flatten(1,2).transpose(1,2)) *
              scale).unflatten(-1, packed.key_lengths.shape[1:])
    lengths = packed.key_lengths[:, None]
    return (logits + lengths.clamp_min(1).float().log()).masked_fill(lengths == 0, -torch.inf)


def _scatter_values(weight, packed):
    if packed.mode not in ("v2","v4"):
        return weight
    index = packed.q_to_v[:, None].expand_as(weight)
    return torch.zeros_like(weight).scatter_add_(-1,index,weight)


def physical_v_mask(token_mask, packed):
    """Per-head representative reads, preserving the GQA union externally."""
    valid = token_mask & (packed.key_lengths[:, None] > 0)
    return _scatter_values(valid.to(torch.int32), packed) > 0


def packed_output(logits, packed, mask=None):
    """PV after grouping any retained token probabilities into V rows."""
    scores = logits if mask is None else logits.masked_fill(~mask, -torch.inf)
    probs = scores.flatten(2).softmax(-1).reshape_as(scores)
    weights = _scatter_values(probs, packed).flatten(2)
    return torch.bmm(weights, packed.value.flatten(1,2))
