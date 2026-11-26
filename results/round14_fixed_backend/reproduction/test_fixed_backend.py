import unittest

import torch

from fixed_backend import (
    EncodedCache, a1_normalize, exponent_bounds, histogram_gate, integer_qk, query_codes,
    round_divide, select_values, weighted_pv,
)
from gqa_probe.chunked_prefill import PreallocatedAffineKV4Cache
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.light_a1_reference import light_probabilities
from gqa_probe.pagepack import Affine32, _encode_affine32
from gqa_probe.page_read_budget import page_history
from gqa_probe.summary_decode import plan, read_trace


class FixedBackendTest(unittest.TestCase):
    def test_signed_nearest_even(self):
        values = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
        torch.testing.assert_close(round_divide(values, 2), torch.tensor([-4, -2, -2, 0, 0, 2, 2, 4]))

    def test_a1_reproduces_existing_reference(self):
        raw = torch.tensor([[[0., -0.69314718056, -1.38629436112]]], dtype=torch.float64)
        fixed = (raw * (1 << 24)).round().long()
        expected = torch.tensor(light_probabilities(raw[0].numpy())[1]).unsqueeze(0)
        torch.testing.assert_close(a1_normalize(fixed, torch.ones_like(fixed, dtype=torch.bool)), expected)

    def test_affine_integer_qk_has_expected_signed_zero_correction(self):
        q = torch.tensor([[[1., -1., 0., 0.] * 32]])
        ks = torch.tensor([[[[2., -1., 0., 0.] * 32, [-2., 1., 0., 0.] * 32]]])
        packed = _encode_affine32(ks)
        selected = Affine32(packed.code[0], packed.scale[0], packed.zero_point[0])
        q8, scale = query_codes(q)
        out, detail = integer_qk(q8, scale, selected, 0.125)
        expected = (q @ packed.decode()[0].transpose(-1, -2) * .125 * 1024).round().long()
        # The encoded Q24 coefficient can shift the final Q10 rounding by one.
        torch.testing.assert_close(out, expected, atol=1, rtol=0)
        codes = torch.stack((selected.code & 15, selected.code >> 4), -1).flatten(-2)
        centered = codes.unflatten(-1, (4, 32)).long() - selected.zero_point[..., None]
        exact_dots = (q8.unflatten(-1, (4, 32))[:, :, None].long() * centered[:, None]).sum(-1).transpose(-1, -2)
        torch.testing.assert_close(detail['qk_group_dot'], exact_dots)

    def test_cache_preserves_original_affine_bytes_and_prefill_values(self):
        torch.manual_seed(1)
        source = torch.randn(1, 2, 5, 128).bfloat16()
        baseline, candidate = PreallocatedAffineKV4Cache(5), EncodedCache(5)
        for chunk in (source[..., :3, :], source[..., 3:, :]):
            actual = candidate.update(chunk, -chunk, 0)
            expected = baseline.update(chunk, -chunk, 0)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
        torch.testing.assert_close(candidate.encoded_keys[0].decode(torch.bfloat16), candidate.key_cache[0])

    def test_mass_bounds_and_histogram_budget(self):
        gaps = torch.arange(0, 24 * 1024, dtype=torch.int64)
        lower, upper = exponent_bounds(gaps)
        exact = (-gaps.double() / 1024).exp() * (1 << 20)
        self.assertTrue(bool((lower <= exact).all() & (exact <= upper).all()))
        logits = -torch.arange(96).reshape(1, 1, 3, 32) * 256
        valid = torch.ones((1, 3, 32), dtype=torch.bool)
        valid[0, -1, -1] = False
        pins = torch.tensor([[False, False, True]])
        keep, hist, _ = histogram_gate(logits, valid, pins)
        self.assertEqual(hist.shape[-1], 64)
        self.assertFalse(bool(keep[..., -1, -1].any()))
        self.assertTrue(bool(keep[..., -1, :-1].all()))
        weight = (logits.double() / 1024).masked_fill(~valid[:, None], -torch.inf).flatten(-2).softmax(-1)
        self.assertLessEqual(float((weight * ~keep.flatten(-2)).sum()), .02)

    def test_directed_fallback_reuses_initial_k_and_preserves_pins(self):
        history = page_history(32_001, 32)
        ranks = ranked_pages(torch.zeros(1, history.pages), history)
        allocated = plan(ranks, history, summary_bytes=112, budget_percent=5, v_reserve=.80)
        valid = allocated.full_mask()
        flat_scores = torch.zeros((1, 4, *valid.shape[-2:]), dtype=torch.int64)
        result = select_values(allocated, flat_scores, valid, allocated.protected)
        self.assertTrue(bool(result.fallback.item()))
        self.assertLess(int(result.prefix.item()), int(allocated.count.item()))
        self.assertEqual(int(result.ledger['over_bytes'].item()), 0)
        self.assertEqual(int(result.ledger['k_records'].item()), int(valid.sum()))
        self.assertTrue(bool((result.final | ~(allocated.protected[..., None] & valid)[:, None]).all()))
        address_bytes = sum(size for phase in read_trace(allocated, result.final) for _, size in phase)
        self.assertEqual(address_bytes, int(result.ledger['total'].item()))

    def test_pv_affine_scaling_and_unsigned_weights(self):
        values = torch.tensor([[[[1., -1.] * 64, [3., -3.] * 64]]])
        packed = _encode_affine32(values)
        selected = Affine32(packed.code[0], packed.scale[0], packed.zero_point[0])
        keep = torch.ones(1, 1, 1, 2, dtype=torch.bool)
        logits = torch.zeros_like(keep, dtype=torch.int64)
        expected = packed.decode()[0].mean(1, keepdim=True)
        for width in (8, 16):
            output, detail = weighted_pv(logits, keep, selected, width)
            torch.testing.assert_close(output, expected, atol=2 / 4096, rtol=0)
            self.assertEqual(int(detail['p_code'].max()), (1 << width) - 1)


if __name__ == '__main__':
    unittest.main()
