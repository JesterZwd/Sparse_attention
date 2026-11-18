import numpy as np
import pytest

from gqa_probe.selection import protected_pages, select_pages


MASS = np.array([[.60, 0], [.35, .30], [.05, .25], [0, .45]])


def test_deficit_changes_choice_and_reduces_total_deficit():
    a1 = select_pages(MASS, [], 2, .5, "A1")
    a3 = select_pages(MASS, [], 2, .5, "A3")
    assert a1.order == [1, 0]
    assert a3.order == [1, 2]
    assert np.maximum(.5 - a1.coverage, 0).sum() == pytest.approx(.2)
    assert np.maximum(.5 - a3.coverage, 0).sum() == pytest.approx(.1)


def test_linear_limit_matches_a1():
    for budget in range(1, 5):
        assert select_pages(MASS, [], budget, 1., "A3").order == select_pages(
            MASS, [], budget, 1., "A1"
        ).order


def test_batch_one_is_full_greedy():
    a3 = select_pages(MASS, [], 3, .5, "A3")
    batched = select_pages(MASS, [], 3, .5, "A3-B", batch_size=1)
    assert batched.order == a3.order
    np.testing.assert_allclose(batched.coverage, a3.coverage)


def test_adaptive_checks_each_page_and_protection_can_satisfy():
    assert select_pages(MASS, [], 4, .2, "A1", mode="adaptive").order == [1]
    for method in ["A1", "A3", "A3-B"]:
        assert select_pages(MASS, [1, 1], 4, .2, method, mode="adaptive").order == [1]


def test_fixed_fill_is_excluded_from_dynamic_metrics():
    result = select_pages(MASS, [], 4, .2, "A3")
    assert result.order == [1, 0, 3, 2]
    assert result.metrics["selector_iterations"] == 1
    assert result.metrics["coverage_selection_pages"] == 1
    assert result.metrics["a1_fill_pages"] == 3
    assert len(result.batches) == 1


def test_batch_overshoot_and_redundancy_are_audited():
    result = select_pages(MASS, [], 4, .2, "A3-B", batch_size=4, mode="adaptive")
    batch = result.batches[0]
    assert batch["first_satisfied_position"] == 1
    assert batch["batch_overshoot_pages"] == 3
    # Predicted .20 + .40 + .25 + .20, actual capped gain .40.
    assert batch["redundancy"] == pytest.approx(.65)
    assert batch["actual_batch_size"] == 4


def test_equal_scores_choose_lower_page_id():
    mass = np.full((5, 2), .2)
    for method in ["A1", "A3", "A3-B"]:
        assert select_pages(mass, [], 3, .8, method, batch_size=2).order == [0, 1, 2]


def test_physical_protection_counts_partial_local_page():
    assert protected_pages(16384, 32, 1, 128) == [0, 508, 509, 510, 511]
    assert protected_pages(16385, 32, 1, 128) == [0, 508, 509, 510, 511, 512]
    assert protected_pages(13, 32, 1, 128) == [0]


@pytest.mark.parametrize("budget,protected", [(0, [0]), (5, []), (-1, [])])
def test_invalid_budget_is_rejected(budget, protected):
    with pytest.raises(ValueError):
        select_pages(MASS, protected, budget, .8, "A3")


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), MASS * 2, np.array([[np.nan]])])
def test_mass_must_be_finite_nonnegative_and_normalized(bad):
    with pytest.raises(ValueError):
        select_pages(bad, [], 1, .8, "A1")


def test_budget_protection_and_batch_redundancy_randomized():
    rng = np.random.default_rng(42)
    for _ in range(20):
        mass = rng.dirichlet(np.ones(12), size=4).T
        result = select_pages(mass, [0, 11, 11], 7, .7, "A3-B", batch_size=4)
        assert len(set(result.order)) == len(result.order) == 7
        assert {0, 11}.issubset(result.order)
        assert all(b["redundancy"] >= -1e-12 for b in result.batches)
