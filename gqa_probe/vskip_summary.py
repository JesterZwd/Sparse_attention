"""Packed Quest extrema and a single compact mean/radius ablation."""
from dataclasses import dataclass

import torch


QUEST_BYTES = {8: 272, 6: 208, 4: 144}
METHOD_BYTES = {**{f"q{b}": n for b, n in QUEST_BYTES.items()}, "mr25": 144, "mr50": 144}


def quantize_vector(x, bits, direction):
    limit = (1 << (bits - 1)) - 1
    target = x.abs().amax(-1, keepdim=True) / limit
    step = target.clamp_min(2 ** -24).half()
    if direction:
        step = (step.contiguous().view(torch.int16) + (step.float() < target).short()).view(torch.float16)
    scale = step.float()
    u = x / scale
    if direction > 0:
        q = u.ceil()
        q += (q * scale < x).float()
    elif direction < 0:
        q = u.floor()
        q -= (q * scale > x).float()
    else:
        q = u.round()
    return q.clamp(-limit, limit).to(torch.int8), step


def pack_codes(codes, bits):
    """Little-endian signed codes in 4- or 6-bit fields, returned as bytes."""
    if bits == 8:
        return codes.view(torch.uint8)
    lanes, nbytes = (2, 1) if bits == 4 else (4, 3)
    x = codes.to(torch.int32).unflatten(-1, (-1, lanes)) & ((1 << bits) - 1)
    shifts = torch.arange(lanes, device=x.device, dtype=torch.int32) * bits
    word = (x << shifts).sum(-1, dtype=torch.int32)
    parts = [(word >> (8 * index)).to(torch.uint8) for index in range(nbytes)]
    return torch.stack(parts, -1).flatten(-2)


def unpack_codes(packed, bits):
    if bits == 8:
        return packed.view(torch.int8)
    lanes, nbytes = (2, 1) if bits == 4 else (4, 3)
    x = packed.to(torch.int32).unflatten(-1, (-1, nbytes))
    word = sum(x[..., i] << (8 * i) for i in range(nbytes))
    shifts = torch.arange(lanes, device=x.device, dtype=torch.int32) * bits
    v = (word[..., None] >> shifts) & ((1 << bits) - 1)
    center = 1 << (bits - 1)
    return ((v ^ center) - center).flatten(-2).to(torch.int8)


def encode(sums, lo, hi, counts, method):
    if method.startswith("q"):
        bits = int(method[1:])
        low, ls = quantize_vector(lo, bits, -1)
        high, hs = quantize_vector(hi, bits, 1)
        return pack_codes(low, bits), ls, pack_codes(high, bits), hs
    mean = sums / counts
    code, scale = quantize_vector(mean, 6, 0)
    radius = torch.maximum(hi - mean, mean - lo)
    rs = (radius.amax(-1, keepdim=True) / 3).clamp_min(2 ** -24).half()
    rad = (radius / rs.float()).round().clamp(0, 3).to(torch.uint8)
    return ((code.view(torch.uint8) & 63) | (rad << 6)), scale, rs, counts.to(torch.uint8)


def decoded(records, method):
    if method.startswith("q"):
        bits = int(method[1:])
        a, sa, b, sb = records
        return unpack_codes(a, bits).float() * sa, unpack_codes(b, bits).float() * sb
    record, scale, rs, counts = records
    mean = (((record & 63).short() ^ 32) - 32).float() * scale
    return mean, (record >> 6).float() * rs, counts


def weights(query, records, method, scaling):
    if method.startswith("q"):
        lo, hi = decoded(records, method)
        # Factored sign-select formula. Each signed term uses one extreme.
        scores = (torch.bmm(query.clamp_max(0), lo.transpose(1, 2))
                  + torch.bmm(query.clamp_min(0), hi.transpose(1, 2))) * scaling
    else:
        center, radius, counts = decoded(records, method)
        alpha = int(method[2:]) / 100
        scores = (torch.bmm(query, center.transpose(1, 2))
                  + alpha * torch.bmm(query.abs(), radius.transpose(1, 2))) * scaling
        scores = scores + counts.float().log().transpose(1, 2)
    return scores.softmax(dim=-1).sum(1)


@dataclass(frozen=True)
class StaticPrefix:
    tokens: int
    page: int
    records: dict
    tail_sum: torch.Tensor
    tail_lo: torch.Tensor
    tail_hi: torch.Tensor

    @classmethod
    def build(cls, key, page, methods):
        _, groups, tokens, _ = key.shape
        full, tail = divmod(tokens, page)
        body = key[0, :, :full * page].unflatten(1, (full, page)).float()
        sums, lo, hi = body.sum(2), body.amin(2), body.amax(2)
        lengths = torch.full((groups, full, 1), page, device=key.device, dtype=torch.int32)
        records = {m: encode(sums, lo, hi, lengths, m) for m in methods}
        partial = key[0, :, full * page:].float()
        tail_sum = partial.sum(1, keepdim=True)
        low = partial.amin(1, keepdim=True) if tail else torch.full_like(tail_sum, torch.inf)
        high = partial.amax(1, keepdim=True) if tail else torch.full_like(tail_sum, -torch.inf)
        return cls(tokens, page, records, tail_sum, low, high)


class StaticOnline:
    def __init__(self, prefix, methods):
        self.page = prefix.page
        self.length = prefix.tokens
        self.data = {m: prefix.records[m] for m in methods}
        self.sums = prefix.tail_sum
        self.low = prefix.tail_lo
        self.high = prefix.tail_hi

    def append(self, key, tokens):
        if tokens != self.length + 1:
            raise ValueError("nonconsecutive KV append")
        self.length = tokens
        key = key.float().unsqueeze(1)
        self.sums = self.sums + key
        self.low, self.high = torch.minimum(self.low, key), torch.maximum(self.high, key)
        count = (tokens - 1) % self.page + 1
        lengths = torch.full((*key.shape[:2], 1), count, device=key.device, dtype=torch.int32)
        result = {}
        for method, sealed in self.data.items():
            tail = encode(self.sums, self.low, self.high, lengths, method)
            result[method] = tuple(torch.cat((x, y), dim=1) for x, y in zip(sealed, tail))
        if count == self.page:
            self.data = result
            self.sums = torch.zeros_like(self.sums)
            self.low = torch.full_like(self.low, torch.inf)
            self.high = torch.full_like(self.high, -torch.inf)
        return result
