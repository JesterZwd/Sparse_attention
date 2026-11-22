import pytest
import torch

from gqa_probe.fixed_k_v import keep_mask, probabilities
from gqa_probe.round7_gates import gate_cost, gate_mask
from gqa_probe.shared_refill import State


def state(logits, tail=4, protected=(0,)):
    z = torch.tensor(logits, dtype=torch.float32).reshape(1, -1, len(logits[0]), 4)
    ids = torch.arange(z.shape[-2])[None]
    positions = ids[..., None] * 4 + torch.arange(4)
    valid = positions < (positions.numel() - 4 + tail)
    forced = torch.isin(ids, torch.tensor(protected))
    z = z.masked_fill(~valid[:, None], -torch.inf)
    return State(ids, positions, z, valid, forced, valid[:, None].expand_as(z))


def test_checked_gate_keeps_failed_heads_and_forced_tokens():
    s = state([[[0., -8., -9., -10.], [-8., -6., -7., -8.]],
               [[0., -9., -9., -9.], [-1.1, -1.2, -1.3, -1.4]]], tail=3)
    keep = gate_mask(s, "checked", 1., epsilon=.02)
    expected = torch.tensor([[[[1, 1, 1, 1], [0, 0, 0, 0]],
                              [[1, 1, 1, 1], [1, 1, 1, 0]]]], dtype=torch.bool)
    assert torch.equal(keep, expected)
    assert torch.equal(keep.any(1), s.valid)


def test_checked_threshold_includes_equal_gap_and_all_allowed_drops():
    s = state([[[0., -9., -9., -9.], [-5., -5., -4., -7.]]])
    keep = gate_mask(s, "checked", 5.)
    assert torch.equal(keep[0, 0, 1], torch.tensor([False, False, True, False]))


def test_hist64_keeps_entire_boundary_bin():
    s = state([[[0., -20., -20., -20.], [-4.00, -4.04, -4.08, -10.]]])
    hist = gate_mask(s, "hist64")
    exact = gate_mask(s, "exact")
    assert torch.equal(hist[0, 0, 1], torch.tensor([True, True, True, False]))
    assert torch.equal(exact[0, 0, 1], torch.tensor([True, True, False, False]))


def test_hist64_charges_forced_mass_and_saturated_cold_tokens():
    s = state([[[0., -20., -20., -20.], [-14., -17., -500., -1000.]],
               [[-99., -20., -20., -20.], [0., -17., -500., -1000.]]], tail=3)
    mask = gate_mask(s, "hist64")
    assert torch.equal(mask[0, 0, 1], torch.tensor([False, False, False, False]))
    assert torch.equal(mask[0, 1, 1], torch.tensor([True, False, False, False]))


@pytest.mark.parametrize("gate", ["checked", "hist64", "exact"])
def test_zero_mass_limit_keeps_valid_tokens_even_underflowed(gate):
    s = state([[[0., -2., -3., -5.], [-499., -500., -5., -7.]]], tail=3)
    assert torch.equal(gate_mask(s, gate, 2., epsilon=0), s.mask)


@pytest.mark.parametrize("gate", ["checked", "hist64", "exact"])
def test_checked_and_top_p_hold_mass_bound_for_each_head(gate):
    generator = torch.Generator().manual_seed(21)
    logits = (torch.randn(4, 17, 4, generator=generator) * 2).tolist()
    s = state(logits, tail=1, protected=(0, 16))
    mask = gate_mask(s, gate, 3.)
    assert ((probabilities(s.logits) * ~mask).sum((2, 3)) <= .020001).all()
    assert (mask | ~(s.valid & s.protected[..., None])[:, None]).all()
    assert (~mask | s.valid[:, None]).all()


def test_blasst_preserves_quest_traversal_and_full_preserves_tail():
    s = state([[[0., -1., -2., -3.], [-8., -9., -10., -11.], [-.5, -1., -2., -3.]]], tail=1)
    physical = gate_mask(s, "blasst", .01)
    order = torch.tensor([[1, 0, 2]])
    sorted_gate = gate_mask(s, "blasst", .01, order)
    expected = keep_mask(s.logits, None, s.valid, s.protected, "blasst", .01, order)
    assert torch.equal(sorted_gate, expected)
    assert sorted_gate[0, 0, 1].all() and not physical[0, 0, 1].any()
    assert torch.equal(gate_mask(s, "full"), s.mask)


def test_gate_cost_counts_read_passes_without_double_charging_common_logits():
    hist = gate_cost("hist64", 97, 4, optional_tokens=65, pages=4)
    checked = gate_cost("checked", 97, 4, optional_tokens=65, pages=4)
    assert hist["logits_sram_bytes"] == checked["logits_sram_bytes"] == 97 * 4 * 4
    assert hist["logits_sram_reads"] == 2 * checked["logits_sram_reads"] == 97 * 4 * 2
    assert hist["histogram_updates"] == 65 * 4
    assert hist["histogram_bin_visits"] == 2 * 64 * 4
    assert hist["control_sram_bytes"] > checked["control_sram_bytes"]
    assert hist["gate_ck"] > checked["gate_ck"]
    assert checked["gate_exponentials"] == 97 * 4
