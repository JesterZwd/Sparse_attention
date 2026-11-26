"""Recompute GPU captures with a standalone CPU NumPy reference."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import ceil, e, floor, log2
from pathlib import Path

import numpy as np

Q24, MASS = 1 << 24, 1 << 20
LOG2_MIN = floor(log2(e) * Q24)
LOG2_NEAREST = round(log2(e) * Q24)
TABLE = np.array([[convert(MASS * 2 ** (-idx / 16)) for idx in range(16)]
                  for convert in (floor, ceil, round)], dtype=np.int64)
RECIP = np.array([round(2**16 / (1 + (idx + .5) / 256)) for idx in range(256)], dtype=np.int64)


def nearest_even(n, d):
    quotient, remainder = np.divmod(n, d)
    return quotient + ((2 * remainder > d) | ((2 * remainder == d) & ((quotient & 1) > 0)))


def gaps(values, live):
    maximum = np.max(np.where(live, values, -(1 << 61)), axis=-1, keepdims=True)
    return np.where(live, maximum - values, 0)


def exp_code(index):
    bounded = index.clip(0, 320)
    return np.where(index <= 320, TABLE[2, bounded % 16] >> (bounded // 16), 0)


def page_weights(score, live):
    score = nearest_even(score * LOG2_NEAREST, 1 << 44)
    code = np.where(live, exp_code(gaps(score, live)), 0)
    summation = code.sum(-1, keepdims=True)
    exponent = np.asarray([int(s).bit_length() - 1 for s in summation.flat]).reshape(summation.shape)
    lookup = (summation >> (exponent - 8)) & 255
    return nearest_even(code * RECIP[lookup], 1 << (exponent - 4))


def unpack(packed):
    dense = np.empty((*packed.shape[:-1], 128), dtype=np.int64)
    dense[..., 0::2] = packed & 15
    dense[..., 1::2] = packed >> 4
    return dense.reshape(*packed.shape[:-1], 4, 32)


def value_mask(logits, valid, pins):
    shape = logits.shape
    valid_heads = np.broadcast_to(valid[:, None], shape).reshape(*shape[:2], -1)
    forced = np.broadcast_to((valid & pins[..., None])[:, None], shape).reshape(*shape[:2], -1)
    optional = valid_heads & ~forced
    gap = gaps(logits.reshape(valid_heads.shape), valid_heads)
    divisor = 1 << 30
    up = (gap * (LOG2_MIN + 1) + divisor - 1) // divisor
    down = gap * LOG2_MIN // divisor
    top = np.minimum(down, 320)
    bottom = np.minimum(up, 320)
    upper = (TABLE[1, top % 16] + (1 << (top // 16)) - 1) >> (top // 16)
    lower = np.where(up <= 320, TABLE[0, bottom % 16] >> (bottom // 16), 0)
    bin_ids = np.minimum(gap // 256, 63)
    hist = np.zeros((*shape[:2], 64), dtype=np.int64)
    for group, head in np.ndindex(shape[:2]):
        np.add.at(hist[group, head], bin_ids[group, head], (upper * optional)[group, head])
    denominator = (lower * valid_heads).sum(-1, keepdims=True)
    tail = np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]
    mask = forced | (optional & np.take_along_axis(tail * 50 > denominator, bin_ids, axis=-1))
    dropped = (upper * optional * ~mask).sum(-1, keepdims=True)
    if not (dropped * 50 <= denominator).all():
        raise ValueError('histogram exceeded the mass cap')
    return mask.reshape(shape), hist, denominator


def pv_values(logits, mask, values_q24, bits):
    flat = mask.reshape(*mask.shape[:2], -1)
    gap = gaps(logits.reshape(flat.shape), flat).reshape(mask.shape)
    masses = exp_code(nearest_even(gap * LOG2_NEAREST, 1 << 30)) * mask
    block_scale = masses.max(-1, keepdims=True) if bits == 8 else np.full_like(masses[..., :1], MASS)
    limit = (1 << bits) - 1
    code = nearest_even(masses * limit, np.maximum(block_scale, 1))
    restored = nearest_even(code * block_scale, limit).reshape(flat.shape)
    denominator = restored.sum(-1, keepdims=True)
    accumulator = np.einsum('ght,gtd->ghd', restored, values_q24, dtype=np.int64)
    output = nearest_even(accumulator, np.maximum(denominator, 1) * (1 << 12))
    return code, block_scale, denominator, accumulator, output


def verify(path):
    with np.load(path, allow_pickle=False) as archive:
        x = {name: archive[name] for name in archive.files}
    info = json.loads(str(x.pop('metadata')))

    def eq(field, value):
        np.testing.assert_array_equal(x[field], value, err_msg=f'{path.name}: {field}')

    def centers(prefix):
        return unpack(x[f'{prefix}_code']) - x[f'{prefix}_zp'][..., None]

    q = x['q8'].astype(np.int64).reshape(*x['q8'].shape[:2], 4, 32)
    dots = np.einsum('ghbd,gtbd->ghbt', q, centers('k'), dtype=np.int64)
    coefficient = np.rint(x['q_scale'].astype(np.float64)[..., None] *
                          x['k_scale'].astype(np.float64).swapaxes(1, 2)[:, None] * (info['scale'] * Q24)).astype(np.int64)
    accum = (coefficient * dots).sum(2)
    wide = nearest_even(accum, 1 << 14)
    clipped = wide.clip(-(1 << 19), (1 << 19) - 1)
    logits = clipped.reshape(x['logits'].shape)
    eq('q_sum', q.sum(-1))
    eq('qk_group_dot', dots)
    eq('qk_scale_q24', coefficient)
    eq('qk_accumulator', accum)
    eq('logit_saturations', wide != clipped)
    eq('logits', logits)
    eq('a1_weights', page_weights(x['score_q24'], x['score_valid']))
    initial, _, _ = value_mask(logits, x['valid'], x['pins'])
    mask, hist, total = value_mask(logits, x['prefix_valid'], x['pins'])
    eq('first_mask', initial)
    eq('final_mask', mask)
    eq('histogram', hist)
    eq('histogram_total_lower', total)
    values = (centers('v') * (x['v_scale'].astype(np.float64) * Q24).astype(np.int64)[..., None])
    values = values.reshape(*values.shape[:-2], 128)
    fields = ('p_code', 'p_block_scale', 'pv_denominator', 'pv_accumulator', 'pv_q12')
    for field, value in zip(fields, pv_values(logits, mask, values, info['bits']), strict=True):
        eq(field, value)
    addresses = [block for phase in info['addresses'] for block in phase]
    bytes_read = sum(size for _, size in addresses)
    if not (bytes_read == info['bytes'] <= info['cap'] and all(addr % 32 == size % 32 == 0 for addr, size in addresses)):
        raise ValueError(f'{path.name}: incorrect physical read ledger')

    flat = mask.reshape(*mask.shape[:2], -1)
    gap = gaps(logits.reshape(flat.shape), flat)
    floating_mass = np.exp(-gap / 1024) * flat
    floating_mass /= floating_mass.sum(-1, keepdims=True)
    reference = np.einsum('ght,gtd->ghd', floating_mass, values / Q24)
    denominator = float(np.linalg.norm(reference))
    errors = {}
    for bits in (8, 16):
        result = pv_values(logits, mask, values, bits)[-1] / 4096
        errors[str(bits)] = float(np.linalg.norm(result - reference)) / max(denominator, 1e-12)
    return dict(vector=str(path), sha256=sha256(path.read_bytes()).hexdigest(), **info,
                natural_fallback=bool(x['fallback'].item()), pv_relative_l2_same_state=errors)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob('*.npz'))
    if not paths:
        raise SystemExit('no vectors found')
    records = [verify(path) for path in paths]
    summary = dict(vectors=len(records), natural_fallbacks=sum(r['natural_fallback'] for r in records), checks=records)
    if args.receipt:
        args.receipt.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"CPU verified {summary['vectors']} vectors including {summary['natural_fallbacks']} natural fallback states")


if __name__ == '__main__':
    main()
