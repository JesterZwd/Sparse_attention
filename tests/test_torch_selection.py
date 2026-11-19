"""Literal bounded-swap contracts, independent of the CPU implementation."""

import numpy as np
import pytest
import torch

from gqa_probe.selection import select_pages
from gqa_probe.torch_selection import select_torch


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_a1_protection_stable_ties_and_existing_reference(device):
    mass = np.array([[.6, 0], [.35, .3], [.05, .25], [0, .45]])
    batch = torch.tensor(np.stack([mass, np.full((4, 2), .25)]), device=device)
    result = select_torch(batch, [3, 3], 2, .5)
    for group in range(2):
        reference = select_pages(batch[group].cpu().numpy(), [3], 2, .5, "A1")
        assert result.mask[group].nonzero().flatten().tolist() == sorted(reference.order)
        np.testing.assert_allclose(result.coverage[group].cpu(), reference.coverage)
    assert result.mask[1].tolist() == [True, False, False, True]


@pytest.mark.parametrize("device", DEVICES)
def test_one_swap_literal_gain_and_guard(device):
    mass = torch.tensor([[[.41, .3], [.405, .304], [.185, .396]]],
                        dtype=torch.float64, device=device)
    result = select_torch(mass, [], 1, .4, "SWAP")
    assert result.mask.tolist() == [[False, True, False]]
    torch.testing.assert_close(result.coverage.cpu(), torch.tensor([[.405, .304]], dtype=torch.float64))
    assert result.swaps.tolist() == [1]
    assert result.candidate_pairs.tolist() == [2]
    assert result.gate_hit.tolist() == [False]
    # δ=0 keeps the initial good head exactly; neither incoming page is admissible.
    guarded = select_torch(mass, [], 1, .4, "SWAP", delta=0.)
    assert guarded.mask.tolist() == [[True, False, False]]
    assert guarded.swaps.tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
def test_initial_anchor_does_not_drift_on_second_round(device):
    rows = [[.17, .1], [.28, .15], [.275, .154], [.269, .159]]
    rows += [[.0012, .0874]] * 5
    mass = torch.tensor([rows], dtype=torch.float64, device=device)
    result = select_torch(mass, [0], 2, [.35, .4], "SWAP", max_swaps=2)
    assert result.mask.nonzero()[:, 1].tolist() == [0, 2]
    assert result.swaps.tolist() == [1]
    torch.testing.assert_close(result.coverage.cpu(), torch.tensor([[.445, .254]], dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
def test_two_swaps_use_current_membership_and_tie_page_ids(device):
    rows = [[.1, .1], [.2, .12], [.199, .12], [.196, .122], [.195, .122]]
    rows += [[.11 / 6, .416 / 6]] * 6
    result = select_torch(torch.tensor([rows], dtype=torch.float64, device=device),
                          [0], 3, [.4, .5], "SWAP", max_swaps=2)
    assert result.mask.nonzero()[:, 1].tolist() == [0, 3, 4]
    assert result.swaps.tolist() == [2]
    torch.testing.assert_close(result.coverage.cpu(), torch.tensor([[.491, .344]], dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget,protected,target,gate", [
    (4, [], 1., True), (2, [0, 1], .8, False), (0, [], .8, False),
    (2, [], .4, True),
])
def test_noop_budget_and_gate_execute_no_pairs(device, budget, protected, target, gate):
    result = select_torch(torch.full((2, 4, 2), .25, dtype=torch.float64, device=device),
                          protected, budget, target, "SWAP", max_swaps=2)
    assert result.mask.sum(dim=1).tolist() == [budget, budget]
    assert result.gate_hit.tolist() == [gate, gate]
    assert result.swaps.tolist() == [0, 0]
    assert result.candidate_pairs.tolist() == [0, 0]
    assert result.metrics["executed_candidate_pairs"].tolist() == [0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_gate_compacts_groups_and_preserves_physical_budget(device):
    mass = torch.tensor([[[.41, .3], [.405, .304], [.185, .396]],
                         [[.8, .8], [.1, .1], [.1, .1]]], dtype=torch.float64, device=device)
    result = select_torch(mass, [], 1, .4, "SWAP")
    assert result.gate_hit.tolist() == [False, True]
    assert result.candidate_pairs.tolist() == [2, 0]
    assert result.metrics["executed_candidate_pairs"].tolist() == [2, 0]
    assert result.mask.sum(dim=1).tolist() == [1, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_zero_pages_and_per_group_protection(device):
    mass = torch.tensor([[[.5, .5], [.5, .5], [0., 0.]],
                         [[.5, .5], [.5, .5], [0., 0.]]], dtype=torch.float64, device=device)
    protected = torch.tensor([[False, False, True], [True, False, False]], device=device)
    result = select_torch(mass, protected, 2, 1., "SWAP")
    assert result.mask.tolist() == [[True, False, True], [True, True, False]]
    assert result.swaps.tolist() == [0, 0]


@pytest.mark.parametrize("kwargs", [{"budget": 4}, {"budget": -1}, {"algorithm": "A3"},
                                    {"delta": -.1}, {"max_swaps": -1}, {"target": 1.1}])
def test_invalid_configuration_rejected(kwargs):
    options = dict(budget=1, target=.8)
    options.update(kwargs)
    with pytest.raises((ValueError, RuntimeError)):
        select_torch(torch.full((1, 3, 2), 1 / 3, dtype=torch.float64), [], **options)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("delta,want_pages,want_coverage,want_swaps", [
    (.05, [1, 2], [.46, .23], 1), (.09, [2, 3], [.42, .26], 2),
])
def test_literal_equal_gain_ties_and_fixed_initial_floor(device, delta, want_pages, want_coverage, want_swaps):
    mass = torch.tensor([[[.25, .10], [.25, .10], [.21, .13], [.21, .13],
                          [.03, .18], [.03, .18], [.02, .18]]], dtype=torch.float64, device=device)
    result = select_torch(mass, [], 2, .3, "SWAP", delta=delta, max_swaps=2)
    assert result.mask.nonzero()[:, 1].tolist() == want_pages
    assert result.swaps.tolist() == [want_swaps]
    torch.testing.assert_close(result.coverage.cpu(), torch.tensor([want_coverage], dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
def test_swap_does_not_expand_last_four_outgoing_boundary(device):
    rows = [[.29, .01]] + [[.01, .10]] * 5 + [[.0989, .011]]
    rows += [[.5611 / 13, .479 / 13]] * 13
    result = select_torch(torch.tensor([rows], dtype=torch.float64, device=device),
                          [], 6, [.1, .8], "SWAP", delta=.5)
    # Page 0 -> 6 would improve deficit, but page 0 is outside the last four
    # selected pages in W order. Its presence must not expand the boundary.
    assert result.mask.nonzero()[:, 1].tolist() == [0, 1, 2, 3, 4, 5]
    assert result.candidate_pairs.tolist() == [32]
    assert result.swaps.tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
def test_swap_does_not_expand_first_eight_incoming_boundary(device):
    rows = [[.3, .1]] + [[.07, .095]] * 8 + [[.001, .12]] + [[.0695, .01]] * 2
    result = select_torch(torch.tensor([rows], dtype=torch.float64, device=device),
                          [], 1, [.001, .8], "SWAP", delta=.5)
    # Page 9 is beneficial, but ranks ninth among unselected pages.
    assert result.mask.nonzero()[:, 1].tolist() == [0]
    assert result.candidate_pairs.tolist() == [8]
    assert result.swaps.tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
def test_independent_numpy_reference_agrees_on_randomized_batched_decisions(device):
    from gqa_probe.online_selection import select_online

    rng = np.random.default_rng(813)
    for pages, heads, budget, protected in [(1, 1, 1, []), (5, 2, 2, [0]),
                                            (21, 4, 10, [0, 20]), (64, 4, 13, [0, 63]),
                                            (7, 2, 2, [0, 1]), (7, 4, 7, [0])]:
        batch = rng.dirichlet(np.full(pages, .4), size=(3, heads)).transpose(0, 2, 1)
        for target, delta, rounds in [(.3, .01, 2), (.8, .0, 1), (.9, .02, 1), (1., .01, 2)]:
            actual = select_torch(torch.tensor(batch, device=device), protected, budget,
                                  target, "SWAP", delta=delta, max_swaps=rounds)
            for group, mass in enumerate(batch):
                expected = select_online(mass, protected, budget, target, "SWAP",
                                         delta=delta, max_swaps=rounds)
                assert actual.mask[group].nonzero().flatten().tolist() == sorted(expected.order)
                np.testing.assert_allclose(actual.coverage[group].cpu(), expected.coverage, atol=2e-14)
                assert actual.swaps[group].item() == expected.metrics["swaps"]
                assert actual.gate_hit[group].item() == expected.metrics["gate_hit"]
                assert actual.candidate_pairs[group].item() == expected.metrics["candidate_pairs"]


@pytest.mark.parametrize("device", DEVICES)
def test_swap_subtracts_outgoing_before_adding_incoming_exactly(device):
    mass = torch.tensor([[[.6, .1], [.3, .3], [.1, .6]]], dtype=torch.float64, device=device)
    result = select_torch(mass, [], 1, [.2, .5], "SWAP", delta=.5)
    assert result.mask.tolist() == [[False, True, False]]
    # Removing the entire one-page coverage first is exact zero. Computing
    # (C + incoming) - outgoing instead introduces an avoidable rounding error.
    assert torch.equal(result.coverage, mass[:, 1, :])
