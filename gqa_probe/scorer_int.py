"""Bit-exact integer reference for the Round 9 page scorer.

The hardware consumes per-head Q8 queries and two unsigned 22-bit Q24
coefficients per page. Coefficients already include the query, summary and attention
scales. Mean+Radius stores alpha=1/4 in its second coefficient. Group-level
softmax and ranking use the same FP32 operations as the existing selector.
"""

from dataclasses import dataclass

import torch

from .vskip_summary import decoded, unpack_codes


Q24 = 1 << 24
COEFFICIENT_BITS = 22
COEFFICIENT_MAX = (1 << COEFFICIENT_BITS) - 1


@dataclass(frozen=True)
class ScorerInputs:
    method: str
    query: torch.Tensor
    code_a: torch.Tensor
    code_b: torch.Tensor
    coeff_a: torch.Tensor
    coeff_b: torch.Tensor
    bias: torch.Tensor
    valid: torch.Tensor
    float_coeff_a: torch.Tensor
    float_coeff_b: torch.Tensor
    float_bias: torch.Tensor

    @property
    def saturated_a(self):
        return self.valid & ((self.float_coeff_a * Q24).round() > COEFFICIENT_MAX)

    @property
    def saturated_b(self):
        return self.valid & ((self.float_coeff_b * Q24).round() > COEFFICIENT_MAX)

    @property
    def underflow_a(self):
        return self.valid & (self.float_coeff_a > 0) & (self.coeff_a == 0)

    @property
    def underflow_b(self):
        return self.valid & (self.float_coeff_b > 0) & (self.coeff_b == 0)


def _summary_parts(records, method):
    if method == "q4":
        lo, scale_lo, hi, scale_hi = records
        shape = (lo.shape[0], 1, lo.shape[1])
        return (
            unpack_codes(lo, 4), unpack_codes(hi, 4),
            scale_lo.transpose(1, 2), scale_hi.transpose(1, 2),
            torch.zeros(shape, dtype=torch.float32, device=lo.device),
            torch.ones(shape, dtype=torch.bool, device=lo.device),
        )
    if method == "mr25":
        packed, scale_mean, scale_radius, counts = records
        mean = (((packed & 63).short() ^ 32) - 32).to(torch.int8)
        radius = packed >> 6
        count = counts.transpose(1, 2)
        return (
            mean, radius, scale_mean.transpose(1, 2),
            scale_radius.float().transpose(1, 2) * .25,
            count.clamp_min(1).double().log(), count > 0,
        )
    raise ValueError(f"unsupported integer scorer method: {method}")


def _dot_parts(query, code_a, code_b, method):
    # Widen before abs so a directed RTL test can exercise signed Q8 -128.
    q = query.float()
    qa = q.clamp_max(0) if method == "q4" else q
    qb = q.clamp_min(0) if method == "q4" else q.abs()
    first = torch.bmm(qa, code_a.float().transpose(1, 2))
    second = torch.bmm(qb, code_b.float().transpose(1, 2))
    return first, second


def prepare_inputs(query, records, method, scaling):
    """Quantize once per query head, with round-to-even and visible saturation."""
    code_a, code_b, scale_a, scale_b, bias, valid = _summary_parts(records, method)
    q = query.float()
    if q.ndim != 3 or q.shape[0] != code_a.shape[0] or q.shape[2] != code_a.shape[2]:
        raise ValueError("query and summary group/dimension mismatch")
    # At width 128, the largest possible dot is under 2**20. FP32 bmm on
    # integer operands therefore accumulates exactly even on the CPU reference.
    if q.shape[-1] > 128:
        raise ValueError("integer scorer supports at most 128 channels")
    q_scale = q.abs().amax(-1, keepdim=True) / 127
    divisor = torch.where(q_scale > 0, q_scale, 1.)
    q8 = torch.round(q / divisor).clamp(-127, 127).to(torch.int8)
    combined = q_scale.double() * float(scaling)
    fa = combined * scale_a.double()
    fb = combined * scale_b.double()
    coeff_a = (fa * Q24).round().clamp_(0, COEFFICIENT_MAX).to(torch.int64)
    coeff_b = (fb * Q24).round().clamp_(0, COEFFICIENT_MAX).to(torch.int64)
    encoded_bias = (bias * Q24).round().to(torch.int64)
    return ScorerInputs(method, q8, code_a, code_b, coeff_a, coeff_b, encoded_bias,
                        valid, fa, fb, bias)


def integer_dots(data):
    """Get two exact integer accumulators before the per-page coefficient stage."""
    a, b = _dot_parts(data.query, data.code_a, data.code_b, data.method)
    return a.to(torch.int64), b.to(torch.int64)


def integer_scores(data):
    """Return signed Q24 scores before converting to FP32 for normalization."""
    a, b = integer_dots(data)
    return a * data.coeff_a + b * data.coeff_b + data.bias


def fixed_scores(data):
    """Use negative infinity for padded MR records with a zero token count."""
    return (integer_scores(data).float() / Q24).masked_fill(~data.valid, -torch.inf)


def q8_scores(data):
    """Q8 query ablation with FP64 scales and bias to isolate Q24 rounding."""
    a, b = integer_dots(data)
    out = (a * data.float_coeff_a + b * data.float_coeff_b + data.float_bias).float()
    return out.masked_fill(~data.valid, -torch.inf)


def factored_scores(query, records, method, scaling):
    """Factor each decoded summary scale outside its FP32 dot product."""
    ca, cb, sa, sb, bias, valid = _summary_parts(records, method)
    a, b = _dot_parts(query, ca, cb, method)
    out = (a * sa.float() + b * sb.float()) * scaling + bias.float()
    return out.masked_fill(~valid, -torch.inf)


def float_scores(query, records, method, scaling):
    """Use the existing FP32 operations before its per-head softmax."""
    query = query.float()
    if method == "q4":
        low, high = decoded(records, method)
        return (torch.bmm(query.clamp_max(0), low.transpose(1, 2)) +
                torch.bmm(query.clamp_min(0), high.transpose(1, 2))) * scaling
    if method == "mr25":
        center, radius, counts = decoded(records, method)
        out = (torch.bmm(query, center.transpose(1, 2)) +
               .25 * torch.bmm(query.abs(), radius.transpose(1, 2))) * scaling
        return out + counts.float().log().transpose(1, 2)
    raise ValueError(f"unsupported integer scorer method: {method}")


def scores(query, records, method, scaling):
    return fixed_scores(prepare_inputs(query, records, method, scaling))


def weights(query, records, method, scaling):
    """Drop-in selector scorer for q4 and mr25 with unchanged group-level A1."""
    return scores(query, records, method, scaling).softmax(dim=-1).sum(1)
