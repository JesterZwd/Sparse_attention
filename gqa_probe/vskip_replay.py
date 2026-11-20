"""Address replay and conservative progressive INT8 ranking certificates."""

from dataclasses import dataclass
import math

import numpy as np

from .fixed_k_v import burst_round, vector_burst
from .vskip_summary import METHOD_BYTES


@dataclass
class Bus:
    # One request stream per layer/KV group, with 32 B bursts striped by row.
    banks: int = 8
    row_bytes: int = 2048

    def __post_init__(self):
        self.open = [-1] * self.banks
        self.bursts = self.row_hits = self.activations = 0

    def read(self, offset, size):
        if offset % 32:
            raise ValueError("unaligned layout segment")
        self.bursts += burst_round(size) // 32
        end = offset + burst_round(size)
        while offset < end:
            row = offset // self.row_bytes
            bank = row % self.banks
            upto = min(end, (row+1)*self.row_bytes)
            n = (upto-offset)//32
            if self.open[bank] == row:
                self.row_hits += n
            else:
                self.activations += 1
                self.row_hits += n-1
                self.open[bank] = row
            offset = upto


def replay(ids, keep, tokens, page_size, method, *, batch=0, bus_geometry=(8, 2048)):
    """Logical read order, row-hit count, and scratch storage for one group."""
    h = (tokens + page_size - 1) // page_size
    bus = Bus(*bus_geometry)
    stride = vector_burst(page_size)
    k_base = 1 << 22
    v_base = 1 << 25
    bus.read(0, h * METHOD_BYTES[method])
    bus.read(1 << 20, 4 * len(ids))
    first_wait = None
    barriers = 0
    consumed = 0
    chosen_batch = batch or len(ids)
    for i in range(0, len(ids), chosen_batch):
        chunk = list(zip(ids[i:i+chosen_batch], keep[i:i+chosen_batch]))
        batch_k = [(p, vector_burst(min(page_size, tokens - p*page_size))) for p, _ in chunk]
        for p, size in batch_k:
            bus.read(k_base + p * stride, size)
            consumed += size//32
        barriers += 1
        for (p, retained), (_, size) in zip(chunk, batch_k):
            if retained:
                if first_wait is None:
                    first_wait = consumed
                bus.read(v_base + p * stride, size)
    return dict(total_bytes=bus.bursts*32, transactions=bus.bursts, row_hits=bus.row_hits,
                activations=bus.activations, k_bursts_before_first_v=first_wait, barriers=barriers)


def _envelopes(lo, hi):
    # Subtraction of a common per-head maximum and a formula that excludes the
    # candidate's own page avoid optimistic probability bounds.
    shift = hi.max(axis=-1, keepdims=True)
    a, b = np.exp(lo-shift), np.exp(hi-shift)
    # Use nonnegative exclusive prefix/suffix sums to avoid cancellation.
    def other(x):
        left = np.cumsum(x, axis=-1)
        right = np.cumsum(x[..., ::-1], axis=-1)[..., ::-1]
        return np.pad(left[..., :-1], ((0,0),(1,0))) + np.pad(right[..., 1:], ((0,0),(0,1)))
    return (a / (a + other(b))).sum(0), (b / (b + other(a))).sum(0)


def progressive(query, low_codes, low_scale, high_codes, high_scale, protect, count, scaling):
    """One exact-residual pass, then full fallback unless the selection certifies."""
    q = np.asarray(query, dtype=np.float64)
    low = np.asarray(low_codes, dtype=np.int16)
    high = np.asarray(high_codes, dtype=np.int16)
    ls, hs = low_scale.astype(np.float64), high_scale.astype(np.float64)
    low_coarse = ((low + 128) // 4) * 4 - 128
    high_coarse = ((high + 128) // 4) * 4 - 128
    def dot(l, h):
        return (q.clip(max=0) @ (l*ls).T + q.clip(min=0) @ (h*hs).T) * scaling
    exact = dot(low, high)
    lower = dot(low_coarse + 3, high_coarse)
    upper = dot(low_coarse, high_coarse + 3)
    eligible = np.ones(low.shape[0], bool)
    eligible[list(protect)] = False
    take = count-len(protect)
    def rank(est):
        return np.argsort(-np.where(eligible, est, -np.inf), kind="stable")[:take]
    def cert(sel, a, b):
        outside = eligible.copy()
        outside[sel] = False
        return (not sel.size) or (not outside.any()) or a[sel].min() > b[outside].max() + 1e-12
    a, b = _envelopes(lower, upper)
    s = rank((a+b)*.5)
    if take == 0 or take == eligible.sum():
        return 0, False, True
    if cert(s,a,b):
        return 0, False, set(s)==set(rank(_envelopes(exact,exact)[0]))
    outside = eligible.copy()
    outside[s] = False
    overlap = outside & (b >= a[s].min())
    overlap[s] = a[s] <= b[outside].max()
    attempted = int(overlap.sum())
    lower[:,overlap] = exact[:,overlap]
    upper[:,overlap] = exact[:,overlap]
    a,b = _envelopes(lower,upper)
    s = rank((a+b)*.5)
    if not cert(s,a,b):
        return low.shape[0], True, True
    return attempted, False, set(s)==set(rank(_envelopes(exact,exact)[0]))
