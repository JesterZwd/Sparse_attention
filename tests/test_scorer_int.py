from dataclasses import replace
from math import log

import numpy as np
import pytest
import torch

from gqa_probe.scorer_int import (
    COEFFICIENT_MAX,
    Q24,
    factored_scores,
    fixed_scores,
    float_scores,
    integer_dots,
    integer_scores,
    prepare_inputs,
    q8_scores,
    weights,
)
from gqa_probe.vskip_summary import encode, pack_codes, weights as reference_weights


def test_audit_capture_preserves_protection_and_emits_rtl_fixture(tmp_path):
    from scripts.audit_scorer_numerics import audit

    packed = np.zeros((1, 3, 128), dtype=np.uint8)
    scale = np.full((1, 3, 1), .125, dtype=np.float16)
    counts = np.array([[[32], [7], [0]]], dtype=np.uint8)
    for i, v in enumerate((-.5, .5)):
        np.savez(
            tmp_path / f"s{i}.npz",
            query=np.full((1, 2, 128), v, dtype=np.float32),
            method="mr25", scale=128**-.5, protected=[1], k=2,
            r0=packed, r1=scale, r2=scale, r3=counts,
        )
    cases, summary, fixtures = audit(tmp_path, rtl_blocks=2, rtl_pages=8)
    assert len(cases) == 2
    assert summary["mr25"]["groups"] == 2
    assert cases[0]["stages"]["q24"]["minimum_topk_overlap"] == 1.
    assert cases[0]["stages"]["q24"]["minimum_optional_topk_overlap"] == 1.
    block = fixtures["mr25"][0]
    assert block["query"] == [-127] * 128
    assert fixtures["mr25"][1]["query"] == [127] * 128
    assert [p["bias"] for p in block["pages"]] == [round(log(32) * Q24), round(log(7) * Q24)]
    assert all(p["score"] == p["bias"] for p in block["pages"])


def test_negative_query_uses_distinct_quest_scales():
    codes_lo = torch.tensor([[[-8, -3, -1, -2]]], dtype=torch.int8)
    codes_hi = torch.tensor([[[7, 6, 5, 4]]], dtype=torch.int8)
    r = (
        pack_codes(codes_lo, 4),
        torch.tensor([[[2**-9]]], dtype=torch.float16),
        pack_codes(codes_hi, 4),
        torch.tensor([[[2**-10]]], dtype=torch.float16),
    )
    data = prepare_inputs(torch.tensor([[[-128., 32., -24., 0.]]]), r, "q4", .125)
    data = replace(data, query=torch.tensor([[[-128, 32, -24, 0]]], dtype=torch.int8))
    ca, cb = data.coeff_a.item(), data.coeff_b.item()
    assert ca == 2 * cb
    assert integer_scores(data).item() == (128 * 8 + 24) * ca + 32 * 6 * cb


def test_radius_bits_and_nonzero_tail_length_bias():
    pack = torch.tensor([[[0b11000001, 0b10111110, 0b01011111, 0]]], dtype=torch.uint8)
    r = (pack, torch.full((1, 1, 1), .25, dtype=torch.float16),
         torch.full((1, 1, 1), .125, dtype=torch.float16), torch.tensor([[[7]]]))
    data = prepare_inputs(torch.tensor([[[-1., .75, -.25, 0.]]]), r, "mr25", .25)
    data = replace(data, query=torch.tensor([[[-128, 13, -31, 0]]], dtype=torch.int8))
    assert data.code_a.tolist() == [[[1, -2, 31, 0]]]
    assert data.code_b.tolist() == [[[3, 2, 1, 0]]]
    assert data.bias.item() == round(log(7) * Q24)
    expected = data.bias.item()
    expected += (-128 - 26 - 31 * 31) * data.coeff_a.item()
    expected += (3 * 128 + 2 * 13 + 31) * data.coeff_b.item()
    assert integer_scores(data).item() == expected
    torch.testing.assert_close(fixed_scores(data), torch.tensor([[[expected / Q24]]]))


def test_zero_query_and_invalid_mean_radius_page():
    q = torch.zeros(1, 2, 4)
    r = (torch.zeros(1, 3, 4, dtype=torch.uint8),
         torch.ones(1, 3, 1, dtype=torch.float16), torch.ones(1, 3, 1, dtype=torch.float16),
         torch.tensor([[[4], [2], [0]]], dtype=torch.uint8))
    data = prepare_inputs(q, r, "mr25", 1.)
    assert data.coeff_a.count_nonzero() == 0
    assert data.coeff_b.count_nonzero() == 0
    assert not data.saturated_a.any()
    torch.testing.assert_close(weights(q, r, "mr25", 1.), torch.tensor([[4 / 3, 2 / 3, 0.]]))


def test_coefficient_saturation_is_visible_before_rank_comparison():
    q = torch.full((1, 1, 4), 127.)
    r = (torch.zeros(1, 1, 4, dtype=torch.uint8), torch.ones(1, 1, 1),
         torch.full((1, 1, 1), 2**-24), torch.tensor([[[1]]], dtype=torch.uint8))
    data = prepare_inputs(q, r, "mr25", 1.)
    assert data.coeff_a.item() == COEFFICIENT_MAX
    assert data.saturated_a.item() is True
    assert data.coeff_b.item() == 0
    assert data.underflow_b.item() is True


@pytest.mark.parametrize("method", ["q4", "mr25"])
def test_float_factorization_preserves_reference_a1(method):
    rng = torch.Generator().manual_seed(7)
    q = torch.randn(2, 4, 128, generator=rng) * .4
    values = torch.randn(2, 13, 32, 128, generator=rng) * .5
    count = torch.full((2, 13, 1), 32, dtype=torch.uint8)
    records = encode(values.sum(2), values.amin(2), values.amax(2), count, method)
    scale = 128**-.5
    original = reference_weights(q, records, method, scale)
    torch.testing.assert_close(float_scores(q, records, method, scale).softmax(-1).sum(1), original)
    torch.testing.assert_close(factored_scores(q, records, method, scale).softmax(-1).sum(1), original)
    prepared = prepare_inputs(q, records, method, scale)
    assert not (prepared.saturated_a | prepared.saturated_b).any()
    a, b = integer_dots(prepared)
    bound = (a.abs() + b.abs() + 1).float() / (2 * Q24)
    error = (fixed_scores(prepared) - q8_scores(prepared)).abs()
    assert torch.all(error <= bound + 1e-6)
