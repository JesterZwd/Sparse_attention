from itertools import combinations

import numpy as np
import pytest

from gqa_probe.online_selection import select_online
from gqa_probe.selection import select_pages


ANCHOR = np.array([[.25, .10], [.25, .10], [.21, .13], [.21, .13],
                   [.03, .18], [.03, .18], [.02, .18]])


def test_swap_has_literal_gain_and_deterministic_in_out_tie():
    result = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.05, record_audit=True)
    assert result.order == [1, 2]
    np.testing.assert_allclose(result.coverage, [.46, .23])
    assert result.metrics["swaps"] == 1
    assert result.metrics["candidate_pairs"] == 10
    assert result.audit[0]["incoming"] == 2
    assert result.audit[0]["outgoing"] == 0
    assert result.audit[0]["gain"] == pytest.approx(.03)


def test_swap_removes_outgoing_before_adding_incoming_under_float64_cancellation():
    mass = np.array([[.6, .1], [.3, .3], [.1, .6]])
    result = select_online(mass, [], 1, .3, "SWAP", delta=.3, record_audit=True)
    assert result.order == [1]
    # The outgoing page is the entire current coverage. Removing it first
    # leaves exact zero, then inserting page 1 must reproduce its stored mass.
    # Adding first gives [.29999999999999993, .30000000000000004] instead.
    np.testing.assert_array_equal(result.coverage, [.3, .3])
    # The same association is required for candidate scoring, independently of
    # the accepted update: the normative post-swap deficit is exactly zero.
    assert result.audit[0]["deficit_after"] == 0.
    assert result.audit[0]["gain"] == .3 - .1


def test_two_rounds_keep_original_a1_anchor():
    guarded = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.05, max_swaps=2)
    relaxed = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.09, max_swaps=2)
    assert guarded.metrics["swaps"] == 1
    assert set(guarded.order) == {1, 2}
    assert relaxed.metrics["swaps"] == 2
    assert set(relaxed.order) == {2, 3}
    np.testing.assert_allclose(relaxed.coverage, [.42, .26])


def test_gate_and_zero_guard_can_leave_a1_unchanged():
    gated = select_online(ANCHOR, [], 2, .15, "SWAP")
    assert gated.order == [0, 1]
    assert gated.metrics["gate_hit"]
    assert gated.metrics["candidate_pairs"] == 0
    guarded = select_online(ANCHOR, [], 2, .3, "SWAP", delta=0)
    assert guarded.order == [0, 1]
    assert guarded.metrics["swaps"] == 0


def test_gate_a3_uses_a1_on_pass_and_a3_otherwise():
    passed = select_online(ANCHOR, [], 2, .15, "GATE-A3")
    assert passed.order == [0, 1]
    assert passed.metrics["gate_hit"]
    assert passed.metrics["gain_evaluations"] == 0
    failed = select_online(ANCHOR, [], 2, .8, "GATE-A3")
    assert failed.order == select_pages(ANCHOR, [], 2, .8, "A3").order


@pytest.mark.parametrize("algorithm,batch", [("A1", 1), ("A3", 1), ("A3-B", 1), ("A3-B", 8)])
def test_reference_matches_existing_selectors(algorithm, batch):
    rng = np.random.default_rng(48)
    for _ in range(8):
        mass = rng.dirichlet(np.full(21, .4), size=4).T
        actual = select_online(mass, [0, 20], 10, .7, algorithm, batch_size=batch)
        reference = select_pages(mass, [0, 20], 10, .7, algorithm, batch_size=batch)
        assert actual.order == reference.order
        np.testing.assert_allclose(actual.coverage, reference.coverage)


def test_alpha_one_reproduces_original_and_exports_upper():
    rng = np.random.default_rng(17)
    for _ in range(8):
        mass = rng.dirichlet(np.full(17, .4), size=4).T
        result = select_online(mass, [0], 5, .9, "A3-U", alpha=1)
        reference = select_online(mass, [0], 5, .9, "A3")
        assert result.order == reference.order
        assert np.all(np.asarray(result.metrics["upper_per_head"]) >= result.coverage - 1e-12)
        assert result.metrics["upper_ranking_items"] == 16 * 4


def test_alpha_one_upper_identity_holds_for_independently_enumerated_feasible_marginal_gains():
    mass = np.array([[.4, .05], [.25, .15], [.15, .35], [.1, .25], [.1, .2]])
    result = select_online(mass, [0], 3, .9, "A3-U", alpha=1)
    # Adding the best two nonprotected entries independently per head gives
    # U=(.4+.25+.15, .05+.35+.25)=(.8,.65). This checks the bound itself;
    # the policy's alpha=1 objective shortcut is not used to derive gains.
    np.testing.assert_allclose(result.metrics["upper_per_head"], [.8, .65])
    capped_target = np.minimum(.9, np.asarray(result.metrics["upper_per_head"]))
    for size in (0, 1):
        for extra in combinations(range(1, 5), size):
            chosen = (0,) + extra
            coverage = mass[list(chosen)].sum(axis=0)
            for page in set(range(5)) - set(chosen):
                after = coverage + mass[page]
                original_gain = (np.minimum(.9, after) - np.minimum(.9, coverage)).sum()
                capped_gain = (np.minimum(capped_target, after) - np.minimum(capped_target, coverage)).sum()
                assert capped_gain == pytest.approx(original_gain, abs=1e-15)


def test_alpha_below_one_changes_the_actual_greedy_objective():
    mass = np.array([[.3, .1], [.4, .02], [.1, .31], [.1, .285], [.1, .285]])
    original = select_online(mass, [0], 2, .9, "A3")
    adjusted = select_online(mass, [0], 2, .9, "A3-U", alpha=.9)
    # Original gains: page 1=.42, page 2=.41. U=(.7,.41), so adjusted
    # deficits=(.33,.269), gains page 1=.35 and page 2=.369. Tied later
    # pages also gain .369, and page 2 wins the page-ID tie.
    assert original.order == [0, 1]
    assert adjusted.order == [0, 2]
    np.testing.assert_allclose(adjusted.metrics["effective_targets"], [.63, .369])


def test_protection_and_budget_survive_all_swap_rounds():
    rng = np.random.default_rng(2)
    for _ in range(20):
        mass = rng.dirichlet(np.ones(20), size=4).T
        result = select_online(mass, [0, 19, 0], 8, .8, "SWAP", max_swaps=2, record_audit=True)
        assert len(result.order) == len(set(result.order)) == 8
        assert {0, 19}.issubset(result.order)
        assert all(row["gain"] > 1e-12 for row in result.audit)
        a1 = select_online(mass, [0, 19], 8, .8, "A1")
        good = a1.coverage >= .8 - 1e-12
        assert np.all(result.coverage[good] >= np.maximum(.8, a1.coverage[good] - .01) - 1e-12)
        assert np.maximum(.8 - result.coverage, 0).sum() <= np.maximum(.8 - a1.coverage, 0).sum() + 1e-12


def test_all_protected_and_full_budget_are_safe():
    for protected, budget in [([0, 1], 2), ([0], 7)]:
        result = select_online(ANCHOR, protected, budget, .9, "SWAP", max_swaps=2)
        assert len(result.order) == budget
        assert result.metrics["candidate_pairs"] == 0


def test_counted_sort_and_uninstrumented_runtime_choose_same_pages():
    counted = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.05, count_comparisons=True)
    runtime = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.05)
    assert counted.order == runtime.order
    assert counted.metrics["ranking_comparisons"] > 0
    assert runtime.metrics["ranking_comparisons"] is None
    assert counted.metrics["initial_coverage_additions"] == 4
    assert counted.metrics["pair_head_evaluations"] == 20
    assert counted.metrics["coverage_update_arithmetic_ops"] == 4


def test_online_reference_has_no_offline_audits_unless_explicitly_requested():
    result = select_online(ANCHOR, [], 2, .3, "SWAP", delta=.05)
    assert result.audit == []
    assert not {"clip_contributions", "changed_decisions", "first_satisfied_position"} & result.metrics.keys()


def test_shared_and_upper_bound_work_is_counted_with_hand_derived_sizes():
    mass = np.array([[.5, .1], [.3, .3], [.2, .6]])
    result = select_online(mass, [0], 2, .9, "A3-U", alpha=.9, count_comparisons=True)
    # W takes one addition per page, free-list rank visits two pages once;
    # each of two head-wise upper ranks also has two pages. U=(.8,.7).
    assert result.metrics["shared_score_head_additions"] == 3
    assert result.metrics["shared_ranking_calls"] == 1
    assert result.metrics["shared_ranking_items"] == 2
    assert result.metrics["upper_ranking_calls"] == 2
    assert result.metrics["upper_ranking_items"] == 4
    assert result.metrics["upper_accumulation_additions"] == 2
    assert result.metrics["ranking_comparisons"] == 4
    assert result.metrics["ranking_score_comparisons"] == 5
    assert result.metrics["ranking_id_comparisons"] == 0
    np.testing.assert_allclose(result.metrics["upper_per_head"], [.8, .7])
    np.testing.assert_allclose(result.metrics["effective_target_per_head"], [.72, .63])


def test_reference_sort_counts_actual_tie_comparisons_and_candidate_storage_is_bounded():
    tie = select_online(np.array([[.5], [.5]]), [], 1, 0., count_comparisons=True)
    assert tie.order == [0]
    assert tie.metrics["ranking_comparisons"] == 1
    assert tie.metrics["ranking_score_comparisons"] == 2
    assert tie.metrics["ranking_id_comparisons"] == 2
    result = select_online(np.full((8, 2), .125), [], 6, 1., "SWAP")
    assert result.metrics["candidate_pairs"] == 8
    assert result.metrics["candidate_build_out_membership_visits"] == 8
    assert result.metrics["candidate_build_out_buffer_writes"] == 6
    assert result.metrics["candidate_build_out_buffer_evictions"] == 2
    assert result.metrics["candidate_build_out_slice_writes"] == 0
    assert result.metrics["candidate_build_in_membership_visits"] == 8
    assert result.metrics["peak_candidate_page_ids"] == 6


@pytest.mark.parametrize("mass,protected,budget,target", [
    (np.array([[.2], [.2]]), [], 1, .8),
    (np.array([[.5], [-.5], [1.]]), [], 1, .8),
    (np.array([[float("nan")], [1.]]), [], 1, .8),
    (np.array([[.5], [.5]]), [2], 1, .8),
    (np.array([[.5], [.5]]), [0, 1], 1, .8),
    (np.array([[.5], [.5]]), [], 3, .8),
    (np.array([[.5], [.5]]), [], 1, 1.1),
    (np.array([[.5], [.5]]), [], 1, float("nan")),
])
def test_online_reference_rejects_invalid_mass_indices_budgets_and_targets(mass, protected, budget, target):
    with pytest.raises(ValueError):
        select_online(mass, protected, budget, target)


def test_trusted_mass_path_preserves_validated_selection_and_zero_target_uses_a1_fill():
    unchecked = select_online(ANCHOR, [0], 3, 0., "A3", validate=False)
    checked = select_online(ANCHOR, [0], 3, 0., "A3")
    assert unchecked.order == checked.order == [0, 1, 2]
    assert unchecked.metrics["gain_evaluations"] == 0
    assert unchecked.metrics["a1_fill_pages"] == 2


@pytest.mark.parametrize("kwargs", [dict(delta=-.01), dict(max_swaps=0), dict(alpha=1.1)])
def test_bad_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        select_online(ANCHOR, [], 2, .3, "SWAP", **kwargs)
