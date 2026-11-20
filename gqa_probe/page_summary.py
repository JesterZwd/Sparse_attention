"""Online INT8 page summaries for post-RoPE, affine32-decoded keys."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def encode_vector(vector, direction=0):
    """Nearest INT8 centroid, or directed rounding that encloses the keys."""
    target = vector.abs().amax(-1, keepdim=True) / 127
    scale = target.clamp_min(2 ** -24).half()
    if direction:
        # For positive IEEE-754 binary16, one increment of the stored bits
        # gives the next value. CUDA nextafter does not accept Half in 2.5.
        bits = scale.contiguous().view(torch.int16)
        scale = (bits + (scale.float() < target).to(torch.int16)).view(torch.float16)
    scale32 = scale.float()
    normalized = vector / scale32
    if direction < 0:
        q = normalized.floor()
        q -= (q * scale32 > vector).float()
    elif direction > 0:
        q = normalized.ceil()
        q += (q * scale32 < vector).float()
    else:
        q = normalized.round()
    return q.clamp(-127, 127).to(torch.int8), scale


@dataclass(frozen=True)
class EncodedPages:
    counts: torch.Tensor
    centroid: tuple[torch.Tensor, torch.Tensor]
    min_key: tuple[torch.Tensor, torch.Tensor]
    max_key: tuple[torch.Tensor, torch.Tensor]
    fp_centroid: torch.Tensor

    @classmethod
    def from_stats(cls, sums, minimum, maximum, counts):
        mean = sums / counts[None, :, None]
        return cls(counts.to(torch.uint8), encode_vector(mean), encode_vector(minimum, -1),
                   encode_vector(maximum, 1), mean)

    def append(self, other):
        def join(a, b):
            return tuple(torch.cat((x, y), dim=1) for x, y in zip(a, b))
        return EncodedPages(torch.cat((self.counts, other.counts)),
                            join(self.centroid, other.centroid), join(self.min_key, other.min_key),
                            join(self.max_key, other.max_key),
                            torch.cat((self.fp_centroid, other.fp_centroid), dim=1))


@dataclass(frozen=True)
class PrefixSummary:
    tokens: int
    page_size: int
    sealed: EncodedPages
    tail_sum: torch.Tensor
    tail_min: torch.Tensor
    tail_max: torch.Tensor
    tail_count: int

    @classmethod
    def build(cls, key, pagesize):
        """Compute once during prefill for sharing across decode policies."""
        _, _, tokens, _ = key.shape
        sealed_pages, tail_count = divmod(tokens, pagesize)
        body = key[0, :, :sealed_pages * pagesize].unflatten(1, (sealed_pages, pagesize)).float()
        counts = torch.full((sealed_pages,), pagesize, dtype=torch.int32, device=key.device)
        sealed = EncodedPages.from_stats(body.sum(2), body.amin(2), body.amax(2), counts)
        tail = key[0, :, sealed_pages * pagesize:].float()
        sums = tail.sum(1)
        if tail_count:
            minimum, maximum = tail.amin(1), tail.amax(1)
        else:
            minimum, maximum = torch.full_like(sums, torch.inf), torch.full_like(sums, -torch.inf)
        return cls(tokens, pagesize, sealed, sums, minimum, maximum, tail_count)


class OnlineSummary:
    def __init__(self, prefix):
        self.tokens = prefix.tokens
        self.page_size = prefix.page_size
        self.sealed = prefix.sealed
        self.sums = prefix.tail_sum
        self.minimum = prefix.tail_min
        self.maximum = prefix.tail_max
        self.count = prefix.tail_count

    def append(self, newest_key, tokens):
        if tokens != self.tokens + 1:
            raise ValueError("expected consecutive one-token summary updates")
        key = newest_key.float()
        self.tokens = tokens
        self.sums = self.sums + key
        self.minimum = torch.minimum(self.minimum, key)
        self.maximum = torch.maximum(self.maximum, key)
        self.count += 1
        counts = torch.full((1,), self.count, dtype=torch.int32, device=key.device)
        tail = EncodedPages.from_stats(self.sums[:, None], self.minimum[:, None], self.maximum[:, None], counts)
        pages = self.sealed.append(tail)
        if self.count == self.page_size:
            self.sealed = pages
            self.sums = torch.zeros_like(self.sums)
            self.minimum = torch.full_like(self.minimum, torch.inf)
            self.maximum = torch.full_like(self.maximum, -torch.inf)
            self.count = 0
        return pages


def decode_vector(encoded):
    code, scale = encoded
    return code.float() * scale.float()


def summary_weights(query, pages, method, scaling):
    """Return [KV group, physical page] ranks from Query and summaries only."""
    groups = pages.centroid[0].shape[0]
    q = query.reshape(groups, -1, query.shape[-1]).float()
    if method.startswith("quest_"):
        positive, negative = q.clamp_min(0), q.clamp_max(0)
        upper = (torch.bmm(positive, decode_vector(pages.max_key).transpose(1, 2)) +
                 torch.bmm(negative, decode_vector(pages.min_key).transpose(1, 2))) * scaling
        if method == "quest_max":
            return (upper - upper.amax(-1, keepdim=True)).amax(1)
        if method != "quest_softmax":
            raise ValueError(f"unknown Quest aggregation {method}")
        per_head = upper
    elif method in ("centroid8", "centroid_fp32"):
        mean = decode_vector(pages.centroid) if method == "centroid8" else pages.fp_centroid
        per_head = torch.bmm(q, mean.transpose(1, 2)) * scaling + pages.counts.float().log()
    else:
        raise ValueError(f"unknown summary method {method}")
    return torch.softmax(per_head, dim=-1).sum(1)


def select_ranked(weights, protected, count):
    """Stable descending rank with Page ID as tie break."""
    if not len(protected) <= count <= weights.shape[-1]:
        raise ValueError("selection budget cannot include required pages")
    score = weights.clone()
    score[:, list(protected)] = torch.inf
    ids = score.argsort(dim=-1, descending=True, stable=True)[:, :count]
    return torch.zeros_like(weights, dtype=torch.bool).scatter_(1, ids, True)


def gathered_output(query, key, value, mask, history, count, scaling):
    """Read only selected tokens for QK and PV. Trim partial protected tail."""
    groups, _, dim = mask.shape[0], key.shape[-2], key.shape[-1]
    ids = mask.to(torch.int32).argsort(descending=True, stable=True)[:, :count]
    offsets = torch.arange(history.page_size, device=key.device)
    valid_tokens = (count - 1) * history.page_size + history.tail_size
    indices = (ids[..., None] * history.page_size + offsets).flatten(1)[:, :valid_tokens]
    indices = indices[..., None].expand(-1, -1, dim)
    k = torch.gather(key[0], 1, indices).float()
    v = torch.gather(value[0], 1, indices).float()
    q = query.reshape(groups, -1, dim).float()
    logits = torch.bmm(q, k.transpose(1, 2)) * scaling
    return torch.bmm(F.softmax(logits, dim=-1), v)
