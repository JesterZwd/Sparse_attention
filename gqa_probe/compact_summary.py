"""Round 10 packed mean/radius summaries and calibration-only channel selection."""

from dataclasses import dataclass

import torch

from . import scorer_int
from .vskip_summary import decoded, encode, pack_codes, quantize_vector, unpack_codes


SUMMARY_BYTES = {"q4": 144, "mr25": 144, "mr4": 112,
                 "mr96": 112, "mr96_k": 112, "mr96_score": 112}
_CHANNEL_COUNTS = {"mr96": 96, "mr96_k": 96, "mr96_score": 96}
_HEAD_DIM = 128


def _take_channels(values, channels, method):
    width = _CHANNEL_COUNTS[method]
    if channels is None or channels.shape != (values.shape[0], width):
        raise ValueError(f"{method} requires {width} frozen channel IDs per KV group")
    if channels.dtype != torch.long or channels.device != values.device:
        raise ValueError("channel IDs must be int64 on the input device")
    return values.gather(-1, channels[:, None, :].expand(-1, values.shape[1], -1))


def encode_summary(sums, low, high, counts, method, *, channels=None):
    """Pack signed means and unsigned radii, with two FP16 scales and a length."""
    if method not in SUMMARY_BYTES:
        raise ValueError(f"unknown summary method: {method}")
    if method in _CHANNEL_COUNTS:
        sums, low, high = (_take_channels(x, channels, method) for x in (sums, low, high))
    valid = counts > 0
    low = low.masked_fill(~valid, 0)
    high = high.masked_fill(~valid, 0)
    if method != "mr4":
        record = encode(sums, low, high, counts.clamp_min(1), "q4" if method == "q4" else "mr25")
        return record if method == "q4" else (*record[:3], counts.to(torch.uint8))

    mean = sums / counts.clamp_min(1)
    code, mean_scale = quantize_vector(mean, 4, 0)
    radius = torch.maximum(high - mean, mean - low)
    radius_scale = (radius.amax(-1, keepdim=True) / 3).clamp_min(2**-24).half()
    radius_code = (radius / radius_scale.float()).round().clamp(0, 3).to(torch.uint8)
    fields = (code.view(torch.uint8) & 15) | (radius_code << 4)
    return pack_codes(fields, 6), mean_scale, radius_scale, counts.to(torch.uint8)


def records_as_bytes(record, method, *, lengths=None):
    """Serialize the encoded fields at the physical record stride for an audit."""
    def scale_bytes(scale):
        return scale.contiguous().view(torch.uint8)

    if method == "q4":
        a, sa, b, sb = record
        fields = (a, b, scale_bytes(sa), scale_bytes(sb))
        if lengths is not None:
            fields += (lengths.to(torch.uint8),)
    else:
        code, sa, sb, count = record
        fields = (code, scale_bytes(sa), scale_bytes(sb), count)
    encoded = torch.cat(fields, dim=-1)
    padding = SUMMARY_BYTES[method] - encoded.shape[-1]
    if padding < 0:
        raise ValueError("encoded summary exceeds the physical record stride")
    return torch.nn.functional.pad(encoded, (0, padding))


def scores(query, record, method, scaling, *, channels=None, alpha=.25, temperature=1.):
    """Score encoded pages with the Round 9 Q8/Q24 convention and full Q8 scale."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaling = float(scaling) / temperature
    if method == "q4" or (method == "mr25" and alpha == .25):
        return scorer_int.scores(query, record, method, scaling)
    if method not in SUMMARY_BYTES:
        raise ValueError(f"unknown summary method: {method}")

    code, mean_scale, radius_scale, counts = record
    if method == "mr4":
        code = unpack_codes(code, 6).view(torch.uint8) & 63
    mean_bits = 4 if method == "mr4" else 6
    sign = 1 << (mean_bits - 1)
    mean = (((code & ((1 << mean_bits) - 1)).short() ^ sign) - sign).to(torch.int8)
    radius = code >> mean_bits

    q = query.float()
    q_scale = q.abs().amax(-1, keepdim=True) / 127
    divisor = torch.where(q_scale > 0, q_scale, 1.)
    if method in _CHANNEL_COUNTS:
        q = _take_channels(q, channels, method)
    q8 = (q / divisor).round().clamp(-127, 127).to(torch.int8)
    scaled = q_scale.double() * scaling
    fa = scaled * mean_scale.transpose(1, 2).double()
    fb = scaled * (radius_scale.transpose(1, 2).double() * alpha)
    coeff_a = (fa * scorer_int.Q24).round().clamp_(0, scorer_int.COEFFICIENT_MAX).long()
    coeff_b = (fb * scorer_int.Q24).round().clamp_(0, scorer_int.COEFFICIENT_MAX).long()
    count = counts.transpose(1, 2)
    bias = count.clamp_min(1).double().log()
    encoded_bias = (bias * scorer_int.Q24).round().long()
    data = scorer_int.ScorerInputs(method, q8, mean, radius, coeff_a, coeff_b, encoded_bias,
                                   count > 0, fa, fb, bias)
    return scorer_int.fixed_scores(data)


def weights(query, record, method, scaling, *, channels=None, alpha=.25, temperature=1.):
    return scores(query, record, method, scaling, channels=channels, alpha=alpha,
                  temperature=temperature).softmax(dim=-1).sum(1)


@dataclass(frozen=True)
class CompactPrefix:
    tokens: int
    page: int
    records: dict
    tail_sum: torch.Tensor
    tail_lo: torch.Tensor
    tail_hi: torch.Tensor
    channels: dict

    @classmethod
    def build(cls, key, page, methods, channels=None):
        if key.ndim != 4 or key.shape[0] != 1 or key.shape[-1] != _HEAD_DIM:
            raise ValueError("expected batch=1 keys with 128 channels")
        if page < 2 or page > 254 or page % 2:
            raise ValueError("summary pages must contain an even number of tokens up to 254")
        methods = tuple(methods)
        channels = {} if channels is None else dict(channels)
        _, groups, tokens, _ = key.shape
        for method in methods:
            if method not in _CHANNEL_COUNTS:
                continue
            ids = channels.get(method)
            if ids is None:
                raise ValueError(f"missing frozen channels for {method}")
            ids = ids.to(device=key.device)
            if ids.shape != (groups, _CHANNEL_COUNTS[method]) or ids.dtype != torch.long:
                raise ValueError(f"invalid frozen channel shape or dtype for {method}")
            if not bool(((ids >= 0) & (ids < _HEAD_DIM)).all() & (ids.diff() > 0).all()):
                raise ValueError("frozen channel IDs must be unique, ordered, and in range")
            channels[method] = ids.clone()

        full, tail = divmod(tokens, page)
        body = key[0, :, :full * page].unflatten(1, (full, page)).float()
        # Round 9 built sealed pages from an intermediate statistic for each pair.
        sums = body.unflatten(2, (page // 2, 2)).sum(3).sum(2)
        low, high = body.amin(2), body.amax(2)
        counts = torch.full((groups, full, 1), page, device=key.device, dtype=torch.long)
        records = {m: encode_summary(sums, low, high, counts, m, channels=channels.get(m)) for m in methods}
        partial = key[0, :, full * page:].float()
        tail_sum = partial.sum(1, keepdim=True)
        tail_lo = partial.amin(1, keepdim=True) if tail else torch.full_like(tail_sum, torch.inf)
        tail_hi = partial.amax(1, keepdim=True) if tail else torch.full_like(tail_sum, -torch.inf)
        return cls(tokens, page, records, tail_sum, tail_lo, tail_hi, channels)


class CompactOnline:
    """Append each newly decoded affine32 key and encode the mutable tail once."""

    def __init__(self, prefix):
        self.page = prefix.page
        self.length = prefix.tokens
        self.data = prefix.records
        self.channels = prefix.channels
        self.sums = prefix.tail_sum
        self.low = prefix.tail_lo
        self.high = prefix.tail_hi

    def append(self, key, tokens):
        if tokens != self.length + 1:
            raise ValueError("nonconsecutive KV append")
        key = key.float().unsqueeze(1)
        self.sums = self.sums + key
        self.low, self.high = torch.minimum(self.low, key), torch.maximum(self.high, key)
        count = (tokens - 1) % self.page + 1
        lengths = torch.full((*key.shape[:2], 1), count, device=key.device, dtype=torch.long)
        result = {}
        for method, sealed in self.data.items():
            tail = encode_summary(self.sums, self.low, self.high, lengths, method,
                                  channels=self.channels.get(method))
            result[method] = tuple(torch.cat((x, y), dim=1) for x, y in zip(sealed, tail))
        self.length = tokens
        if count == self.page:
            self.data = result
            self.sums = torch.zeros_like(self.sums)
            self.low = torch.full_like(self.low, torch.inf)
            self.high = torch.full_like(self.high, -torch.inf)
        return result


@dataclass(frozen=True)
class ChannelStatistics:
    key_abs: torch.Tensor
    score_abs: torch.Tensor
    observations: int


def fit_channel_statistics(query, key, mr25_record, *, alpha=.25):
    """Equal-state means of raw post-RoPE magnitudes and realized score contributions."""
    if key.ndim == 4:
        if key.shape[0] != 1:
            raise ValueError("expected batch=1 keys")
        key = key[0]
    center, radius, counts = decoded(mr25_record, "mr25")
    valid = counts > 0
    page_components = (center.abs() + alpha * radius).masked_fill(~valid, 0)
    # The mean of the product across all query heads and pages factors by dimension.
    summary_mean = page_components.sum(1, dtype=torch.float64) / valid.sum(1).clamp_min(1)
    return ChannelStatistics(key.abs().mean(1, dtype=torch.float64),
                             query.abs().mean(1, dtype=torch.float64) * summary_mean, 1)


class ChannelCalibration:
    def __init__(self):
        self.statistics = {}

    def update(self, layer, query, key, mr25_record):
        stats = fit_channel_statistics(query, key, mr25_record)
        key_abs, score_abs = stats.key_abs.cpu(), stats.score_abs.cpu()
        previous = self.statistics.get(layer)
        observations = 1
        if previous is not None:
            key_abs = key_abs + previous.key_abs
            score_abs = score_abs + previous.score_abs
            observations += previous.observations
        self.statistics[layer] = ChannelStatistics(key_abs, score_abs, observations)

    def freeze(self, count):
        if not self.statistics or not 0 < count <= _HEAD_DIM:
            raise ValueError("channel calibration requires observations and a valid dimension count")
        out = {"k": {}, "score": {}}
        for layer, stats in sorted(self.statistics.items()):
            for name, values in (("k", stats.key_abs), ("score", stats.score_abs)):
                ids = values.argsort(dim=-1, descending=True, stable=True)[..., :count]
                out[name][layer] = ids.sort(dim=-1).values
        return out
