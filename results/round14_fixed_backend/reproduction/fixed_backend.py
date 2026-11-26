"""Experimental integer attention reference. Model projections and dense prefill are BF16."""

from dataclasses import dataclass
from functools import lru_cache
import json
from math import ceil, floor, log2, e

import numpy as np
import torch

from gqa_probe import chunked_prefill, round13_models, scorer_int
from gqa_probe.compact_summary import CompactOnline, score_inputs
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.pagepack import Affine32, _encode_affine32
from gqa_probe.summary_decode import physical_union, read_trace, value_ledger
from scripts.run_round11 import WorkpointAttention

Q24 = 1 << 24
LOGIT_FRACTION = 10
LOGIT_LIMIT = 1 << 19
MASS_MAX = 1 << 20
LOG2E_MIN = floor(log2(e) * Q24)
LOG2E_MAX = LOG2E_MIN + 1
LOG2E_RNE = round(log2(e) * Q24)


def round_divide(numerator, denominator):
    quotient = torch.div(numerator, denominator, rounding_mode='floor')
    remainder = numerator - quotient * denominator
    excess = 2 * remainder - denominator
    return quotient + ((excess > 0) | ((excess == 0) & ((quotient & 1) != 0)))


@lru_cache(maxsize=None)
def tables(device):
    values = [MASS_MAX * 2 ** (-i / 16) for i in range(16)]
    reciprocal = [round((1 << 16) / (1 + (j + .5) / 256)) for j in range(256)]
    return tuple(torch.tensor(x, dtype=torch.int64, device=device)
                 for x in (list(map(floor, values)), list(map(ceil, values)), list(map(round, values)), reciprocal))


def gap_from_max(logits, live):
    maximum = logits.masked_fill(~live, -(1 << 61)).amax(-1, keepdim=True)
    return torch.where(live, (maximum - logits).clamp_min(0), 0)


def exponent_bounds(gap):
    """Enclose exp(-gap/1024) on the Q20 lattice, including underflow tails."""
    lo_table, hi_table, _, _ = tables(gap.device)
    divisor = 1 << (LOGIT_FRACTION + 20)
    down = (gap * LOG2E_MIN) // divisor
    up = (gap * LOG2E_MAX + divisor - 1) // divisor
    capped_down, capped_up = down.clamp_max(320), up.clamp_max(320)
    hi_shift = capped_down >> 4
    lower = lo_table[capped_up & 15] >> (capped_up >> 4)
    upper = (hi_table[capped_down & 15] + (1 << hi_shift) - 1) >> hi_shift
    return lower.masked_fill(up > 320, 0), upper


def exponential(index):
    _, _, lut, _ = tables(index.device)
    index_capped = index.clamp(0, 320)
    mass = lut[index_capped & 15] >> (index_capped >> 4)
    return mass.masked_fill(index > 320, 0)


def a1_normalize(score_q24, live):
    base2 = round_divide(score_q24 * LOG2E_RNE, 1 << 44)
    code = exponential(gap_from_max(base2, live)).masked_fill(~live, 0)
    denominator = code.sum(-1, keepdim=True, dtype=torch.int64)
    exponent = torch.zeros_like(denominator)
    shifted = denominator
    for width in (32, 16, 8, 4, 2, 1):
        shift = torch.where(shifted >= 1 << width, width, 0)
        shifted = shifted >> shift
        exponent += shift
    _, _, _, reciprocal = tables(score_q24.device)
    index = (denominator >> (exponent - 8).clamp_min(0)) & 255
    return round_divide(code * reciprocal[index], 1 << (exponent - 4).clamp_min(0))


def query_codes(query):
    scale = query.abs().amax(-1, keepdim=True) / 127
    q8 = (query / torch.where(scale > 0, scale, 1.)).round().clamp(-127, 127).to(torch.int8)
    return q8, scale


def unpack(code):
    return torch.stack((code & 15, code >> 4), dim=-1).flatten(-2).unflatten(-1, (-1, 32))


def integer_qk(q8, q_scale, keys, attention_scale):
    q = q8.unflatten(-1, (-1, 32)).transpose(1, 2).contiguous()
    k = unpack(keys.code).transpose(1, 2).contiguous()
    groups, blocks, heads, _ = q.shape
    # FP64 batched GEMM is an exact simulator for these <=32 integer products.
    raw = torch.bmm(q.flatten(0, 1).double(), k.flatten(0, 1).double().transpose(1, 2))
    raw = raw.long().unflatten(0, (groups, blocks)).transpose(1, 2)
    sum_query = q.sum(-1, dtype=torch.int64).transpose(1, 2)
    corrected = raw - sum_query[..., None] * keys.zero_point.transpose(1, 2)[:, None]
    coefficient = (q_scale.double()[..., None] * keys.scale.double().transpose(1, 2)[:, None]
                   * (attention_scale * Q24)).round().long()
    accumulator = (corrected * coefficient).sum(2)
    wide = round_divide(accumulator, 1 << (24 - LOGIT_FRACTION))
    logits = wide.clamp(-LOGIT_LIMIT, LOGIT_LIMIT - 1)
    return logits, dict(q_sum=sum_query, qk_group_dot=corrected, qk_scale_q24=coefficient,
                        qk_accumulator=accumulator, logit_saturations=logits != wide)


def histogram_gate(logits, valid, pins):
    live = valid[:, None].expand_as(logits).flatten(-2)
    forced = (valid & pins[..., None])[:, None].expand_as(logits).flatten(-2)
    optional = live & ~forced
    gap = gap_from_max(logits.flatten(-2), live)
    lower, upper = exponent_bounds(gap)
    bin_id = (gap >> (LOGIT_FRACTION - 2)).clamp_max(63)
    histogram = logits.new_zeros((*logits.shape[:2], 64))
    histogram.scatter_add_(-1, bin_id, upper.masked_fill(~optional, 0))
    cold = histogram.flip(-1).cumsum(-1).flip(-1)
    # Upper-bound dropped mass and lower-bound total mass, so the 2% inequality
    # also holds for exact exponentials of the candidate's quantized logits.
    denominator = lower.masked_fill(~live, 0).sum(-1, keepdim=True)
    keep_bin = cold * 50 > denominator
    keep = forced | (optional & keep_bin.gather(-1, bin_id))
    torch._assert_async((((~keep & optional) * upper).sum(-1, keepdim=True) * 50 <= denominator).all(),
                        'candidate histogram exceeded the fixed-logit mass guarantee')
    return keep.reshape_as(logits), histogram, denominator


def weighted_pv(logits, mask, values, bits):
    gap = gap_from_max(logits.flatten(-2), mask.flatten(-2)).reshape_as(logits)
    index = round_divide(gap * LOG2E_RNE, 1 << (LOGIT_FRACTION + 20))
    mass = exponential(index).masked_fill(~mask, 0)
    limit = (1 << bits) - 1
    if bits == 8:
        block_max = mass.amax(-1, keepdim=True)
    elif bits == 16:
        block_max = torch.full_like(mass[..., :1], MASS_MAX)
    else:
        raise ValueError('PV supports UINT8 block max and UINT16 global weights')
    p_code = round_divide(mass * limit, block_max.clamp_min(1))
    weight = round_divide(p_code * block_max, limit).flatten(-2)
    denominator = weight.sum(-1, keepdim=True)
    centered = unpack(values.code).int() - values.zero_point[..., None].int()
    scale_q24 = (values.scale.double() * Q24).long()
    v_q24 = (centered * scale_q24[..., None]).flatten(-2)
    numerator = (weight[..., None] * v_q24[:, None]).sum(-2)
    out_q12 = round_divide(numerator, denominator.clamp_min(1) << 12)
    return out_q12.float() / 4096, dict(p_code=p_code, p_block_scale=block_max, pv_denominator=denominator,
                                        pv_accumulator=numerator, pv_q12=out_q12)


class EncodedCache(chunked_prefill.PreallocatedAffineKV4Cache):
    """Preserve original affine bytes beside the same BF16 prefix views."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.encoded_keys = []
        self.encoded_values = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start = self.get_seq_length(layer_idx)
        stop = start + key_states.shape[-2]
        if stop > self.capacity:
            raise ValueError('encoded KV exceeds the reserved generation tail')
        keys, values = _encode_affine32(key_states), _encode_affine32(value_states)
        visible = super().update(key_states, value_states, layer_idx, cache_kwargs)
        for src, bank in ((keys, self.encoded_keys), (values, self.encoded_values)):
            if layer_idx == len(bank):
                buf = (x.new_empty(*x.shape[:-2], self.capacity, x.shape[-1])
                       for x in (src.code, src.scale, src.zero_point))
                bank.append(Affine32(*buf))
            dst = bank[layer_idx]
            for out, inp in ((dst.code, src.code), (dst.scale, src.scale), (dst.zero_point, src.zero_point)):
                out[..., start:stop, :].copy_(inp)
        return visible


def install_cache():
    chunked_prefill.PreallocatedAffineKV4Cache = EncodedCache
    round13_models.PreallocatedAffineKV4Cache = EncodedCache


def select_encoded(record, positions):
    def select(tensor):
        g, _ = positions.shape
        idx = positions[..., None].expand(g, -1, tensor.shape[-1])
        return tensor[0].gather(1, idx)
    return Affine32(select(record.code), select(record.scale), select(record.zero_point))


@dataclass
class GateResult:
    initial: torch.Tensor
    final: torch.Tensor
    histogram: torch.Tensor
    denominator: torch.Tensor
    valid: torch.Tensor
    fallback: torch.Tensor
    prefix: torch.Tensor
    ledger: dict


def select_values(plan, logits, valid, pins):
    initial, hist, denom = histogram_gate(logits, valid, pins)
    ledger = value_ledger(plan, physical_union(plan, initial))
    fallback = ledger['over_bytes'] > 0
    prefix = torch.where(fallback, plan.safe_count, plan.count)
    prefix_valid = plan.full_mask(prefix)
    final = initial
    if fallback.any().item():
        capped, capped_hist, capped_denom = histogram_gate(logits, prefix_valid, pins)
        final = torch.where(fallback[:, None, None, None], capped, initial)
        hist = torch.where(fallback[:, None, None], capped_hist, hist)
        denom = torch.where(fallback[:, None, None], capped_denom, denom)
        ledger = value_ledger(plan, physical_union(plan, final))
    torch._assert_async((ledger['over_bytes'] == 0).all(), 'fixed backend exceeded physical cap')
    return GateResult(initial, final, hist, denom, prefix_valid, fallback, prefix, ledger)


class FixedAttention(WorkpointAttention):
    def __init__(self, *args, cache, bits, fixture_dir, prompt, layers):
        super().__init__(*args)
        self.cache, self.bits = cache, bits
        self.fixture_dir, self.prompt = fixture_dir, prompt
        self.fixture_layers = {0, 1, layers // 2, layers - 1}
        self.saved_fallback = False

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        groups, tokens, dim = key.shape[1:]
        layer = module.layer_idx
        q = query.reshape(groups, -1, dim).float()
        scale = kwargs.get('scaling') or dim ** -.5
        history = page_history(tokens, self.workpoint.page)
        if layer not in self.online:
            self.online[layer] = CompactOnline(self.prefixes[layer])
        method = self.workpoint.method
        records = self.online[layer].append(key[0, :, -1], tokens)[method]
        inputs = score_inputs(q, records, method, scale, channels=self.masks[method][layer])
        scores = scorer_int.integer_scores(inputs)
        normalized = a1_normalize(scores, inputs.valid)
        ranked = ranked_pages(normalized.sum(1).float(), history)
        plan = self.allocate(self.workpoint, ranked, history)
        valid = plan.full_mask()
        positions = plan.ids[..., None] * history.page_size + torch.arange(history.page_size, device=q.device)
        positions = positions.clamp_max(tokens - 1).flatten(1)
        keys = select_encoded(self.cache.encoded_keys[layer], positions)
        values = select_encoded(self.cache.encoded_values[layer], positions)
        q8, q_scale = query_codes(q)
        logits, control = integer_qk(q8, q_scale, keys, scale)
        logits = logits.unflatten(-1, valid.shape[-2:])
        gate = select_values(plan, logits, valid, plan.protected)
        output, pv = weighted_pv(logits, gate.final, values, self.bits)
        for name, field in gate.ledger.items():
            self._add('read' if name == 'total' else name, field)
        self._add('decode_layer_calls', 1)
        self._add('kv_group_calls', groups)
        self._add('cap_bytes', groups * plan.cap)
        self._add('dense_comparator_bytes', groups * plan.dense_bytes)
        self._add('over_budget_group_steps', gate.ledger['over_bytes'] > 0)
        self._add('fallback_group_steps', gate.fallback)
        self._add('qk_mac', gate.ledger['k_records'] * q.shape[1] * dim)
        self._add('pv_mac', gate.final.sum() * dim)
        self._add('fused_updates', groups * q.shape[1] * history.pages * 96)
        self._add('logit_saturations', control['logit_saturations'] & valid.flatten(1)[:, None])
        if self.fixture_dir is not None and layer in self.fixture_layers:
            fallback_group = None
            if not self.saved_fallback and gate.fallback.any().item():
                fallback_group = int(gate.fallback.nonzero()[0, 0])
                self.saved_fallback = True
            if self.step in (0, 4, 12) or fallback_group is not None:
                self.save_fixture(layer, 0 if fallback_group is None else fallback_group, plan, valid, keys,
                                  values, q8, q_scale, scale, scores, inputs.valid, normalized, logits, control, gate, pv)
        return self.output(query, output)

    def save_fixture(self, layer, group, plan, valid, keys, values, q8, q_scale, scale, scores,
                     score_valid, normalized, logits, control, gate, pv):
        sl = slice(group, group + 1)
        record = dict(q8=q8, q_scale=q_scale, k_code=keys.code, k_scale=keys.scale, k_zp=keys.zero_point,
                      v_code=values.code, v_scale=values.scale, v_zp=values.zero_point, score_q24=scores,
                      score_valid=score_valid, a1_weights=normalized, logits=logits, valid=valid, pins=plan.protected,
                      prefix_valid=gate.valid, first_mask=gate.initial, final_mask=gate.final,
                      histogram=gate.histogram, histogram_total_lower=gate.denominator, page_ids=plan.ids,
                      fallback=gate.fallback, prefix_pages=gate.prefix, **control, **pv)
        record = {name: x[sl].cpu().numpy() for name, x in record.items()}
        metadata = dict(bits=self.bits, group=group, step=self.step, layer=layer, prompt=self.prompt,
                        scale=scale, cap=plan.cap, bytes=int(gate.ledger['total'][group]),
                        addresses=read_trace(plan, gate.final, group=group))
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        path = self.fixture_dir / f'p{self.prompt:02d}-s{self.step:03d}-l{layer:02d}-g{group:02d}.npz'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as dest:
            np.savez_compressed(dest, metadata=json.dumps(metadata), **record)
        tmp.replace(path)
