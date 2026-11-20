"""Fixed-candidate attention with group-shared V masks and per-head PV masks."""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .page_read_budget import page_history
from .vskip_summary import METHOD_BYTES


def burst_round(size):
    return (size + 31) // 32 * 32


def vector_burst(tokens):
    return 64 * tokens + burst_round(8 * tokens) + burst_round(4 * tokens)


def byte_quota(tokens, page_size, method, percent, *, burst=True):
    h = page_history(tokens, page_size)
    tile = vector_burst if burst else lambda n: (h.full.total if n == page_size else h.tail.total) // 2
    full, tail = tile(page_size), tile(h.tail_size)
    dense = 2 * ((h.pages - 1) * full + tail)
    index = burst_round if burst else lambda x: x
    summary = index(h.pages * METHOD_BYTES[method])
    cap = dense * percent // 100
    pages = min(h.pages, (cap - summary - 2 * tail) // (2 * full) + 1)
    while pages >= len(h.protected):
        if summary + index(4 * pages) + 2 * ((pages - 1) * full + tail) <= cap:
            return pages
        pages -= 1
    raise ValueError(f"protection and metadata exceed the {percent}% budget at {tokens} tokens")


def ranked_pages(weights, h):
    score = weights.clone()
    score[:, list(h.protected)] = torch.inf
    return score.argsort(dim=-1, descending=True, stable=True)


def candidate_qk(query, keys, ids, h, scaling, active=None):
    protected = torch.zeros(h.pages, device=ids.device, dtype=torch.bool)
    protected[list(h.protected)] = True
    positions = ids[..., None] * h.page_size + torch.arange(h.page_size, device=ids.device)
    valid = positions < h.tokens
    if active is not None:
        valid = valid & active[..., None]
    safe = positions.clamp_max(h.tokens - 1)
    k = gather_tokens(keys, safe)
    logits = (torch.bmm(query, k.transpose(1, 2)) * scaling).unflatten(-1, positions.shape[1:])
    return logits.masked_fill(~valid[:, None], -torch.inf), safe, valid, protected[ids]


def gather_tokens(tensor, positions):
    index = positions.flatten(1)[..., None].expand(-1, -1, tensor.shape[-1])
    return torch.gather(tensor[0], 1, index).float()


def probabilities(logits):
    return logits.flatten(2).softmax(-1).reshape_as(logits)


@triton.jit
def _greedy(M, ORDER, PROTECT, OUT, H: tl.constexpr, J: tl.constexpr,
            HD: tl.constexpr, EPS: tl.constexpr, BATCH: tl.constexpr):
    g = tl.program_id(0)
    head = tl.arange(0, HD)
    drop = tl.full((HD,), 0, tl.float32)
    denom = tl.full((HD,), 1, tl.float32)
    if BATCH > 0:
        denom = tl.full((HD,), 0, tl.float32)
    for i in range(J):
        j = tl.load(ORDER + g * J + i)
        if BATCH > 0:
            if i % BATCH == 0:
                for r in range(BATCH):
                    u = tl.load(ORDER + g * J + i + r, i + r < J, 0)
                    bump = tl.load(M + (g * H + head) * J + u, (head < H) & (i + r < J), 0)
                    denom += bump
        w = tl.load(M + (g * H + head) * J + j, head < H, 0)
        protected = tl.load(PROTECT + g * J + j) != 0
        legal = tl.sum(((head < H) & (drop + w > EPS * denom - 1e-7)).to(tl.int32), 0) == 0
        skip = legal & (protected == 0)
        drop += tl.where(skip, w, 0)
        tl.store(OUT + g * J + j, tl.where(skip, 0, 1))


def page_keep(mass, protected, epsilon, *, order=None, batch=0):
    """Return the physical V mask. Microbatches process the specified order."""
    if not epsilon:
        return torch.ones_like(protected)
    if order is None:
        # Caller uses ascending Page ID for this stable mass tie break.
        order = mass.amax(1).argsort(dim=-1, stable=True)
    g, h, j = mass.shape
    if mass.is_cuda:
        out = torch.empty_like(protected, memory_format=torch.contiguous_format)
        _greedy[(g,)](mass.contiguous(), order.contiguous(), protected.contiguous(), out,
                      h, j, triton.next_power_of_2(h), float(epsilon), batch, num_warps=1)
        return out
    dropped = torch.zeros_like(mass[:, :, 0])
    denom = torch.zeros_like(dropped) if batch else torch.ones_like(dropped)
    order = order.long()
    keep = torch.ones_like(protected)
    for i in range(j):
        if batch and i % batch == 0:
            inds = order[:, None, i:i+batch].expand(-1, h, -1)
            denom += mass.gather(-1, inds).sum(-1)
        ids = order[:, i:i+1]
        p = mass.gather(-1, ids[:, None, :].expand(-1, h, -1)).squeeze(-1)
        skip = ((dropped + p <= epsilon * denom - 1e-7).all(1, keepdim=True)
                & ~protected.gather(1, ids))
        dropped += skip * p
        keep.scatter_(1, ids, ~skip)
    return keep


def keep_mask(logits, p, valid, protected, mode, threshold, traversal=None):
    """[group,head,page,token] final mask, with all protected tokens retained."""
    valid = valid[:, None]
    if mode == "all" or threshold == 0:
        return valid.expand_as(logits)
    if mode == "mass":
        keep = page_keep(p.sum(-1), protected, threshold)
    elif mode == "blasst":
        peak = logits.amax(-1)
        if traversal is None:
            ordered = peak
        else:
            ordered = peak.gather(-1, traversal[:, None].expand_as(peak))
        skipped = (ordered - ordered.cummax(-1).values < math.log(threshold)).all(dim=1)
        if traversal is not None:
            skipped = torch.empty_like(skipped).scatter_(1, traversal, skipped)
        keep = ~skipped | protected
    elif mode == "sparta":
        token = p.flatten(2)
        forced = (protected[..., None] & valid[:, 0]).flatten(1)[:, None]
        target = (1 - threshold - (token * forced).sum(-1, keepdim=True)).clamp_min(0)
        optional = token.masked_fill(forced, 0)
        order = optional.argsort(dim=-1, descending=True, stable=True)
        ranked = optional.gather(-1, order)
        include = (ranked.cumsum(-1) - ranked < target) & (ranked > 0)
        chosen = torch.zeros_like(include).scatter_(-1, order, include)
        return (chosen | forced).reshape_as(logits) & valid
    else:
        raise ValueError(mode)
    return (keep[..., None] & valid[:, 0])[:, None].expand_as(logits)


def masked_output(logits, values, mask):
    weights = logits.masked_fill(~mask, -torch.inf).flatten(2).softmax(-1)
    return torch.bmm(weights, values)


def traffic(mask, valid, h, method):
    """Group-shared 32 B transactions for three token-contiguous V planes."""
    union = mask.any(1)
    tokens = union.sum((1, 2))
    page_count = union.any(-1).sum(-1)
    scales = union.unflatten(-1, (-1, 4)).any(-1).sum((1, 2))
    zps = union.unflatten(-1, (-1, 8)).any(-1).sum((1, 2))
    v_bytes = 64 * tokens + 32 * (scales + zps)
    if mask.shape[-2] == 0:
        raise ValueError("empty K set")
    active = valid.any(-1).sum(-1)
    k_bytes = (active - 1) * vector_burst(h.page_size) + vector_burst(h.tail_size)
    read_bytes = k_bytes + v_bytes + burst_round(h.pages * METHOD_BYTES[method]) + ((4 * active + 31) // 32 * 32)
    pv_mac = mask.sum((1, 2, 3)) * 128
    return v_bytes, k_bytes, read_bytes, page_count, pv_mac


def micro_order(ranks, ids, batch, address):
    if address:
        j = ids.shape[-1]
        parts = []
        for start in range(0, j, batch):
            slots = ranks[:, start:start+batch]
            parts.append(slots.gather(-1, ids.gather(-1, slots).argsort(dim=-1, stable=True)))
        return torch.cat(parts, -1)
    return ranks
