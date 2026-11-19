import numpy as np
import pytest

from gqa_probe.replay_round2 import evaluate_replay_trace


def test_replay_uses_original_target_for_relaxed_selector():
    mass = np.array([[.60, 0], [.35, .30], [.05, .25], [0, .45]])
    logits = np.log(np.maximum(mass.T, 1e-300))
    values = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    rows = evaluate_replay_trace(logits, values, [], 2, .9, .5,
                                [{"name": "relaxed", "algorithm": "A3-U", "alpha": .9}],
                                page_size=1)
    assert rows[0]["remaining_deficit_sum"] == pytest.approx(.6)
    assert rows[0]["pages"] == [0, 1]
    assert rows[0]["target"] == .9
    assert rows[0]["selector_work"]["effective_targets"] == pytest.approx([.855, .675])


def test_replay_preserves_shared_v_and_hand_checked_a1_a3_coverage():
    mass = np.array([[.60, 0], [.35, .30], [.05, .25], [0, .45]])
    logits = np.log(np.maximum(mass.T, 1e-300))
    values = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    specs = [{"name": "a1", "algorithm": "A1"}, {"name": "a3", "algorithm": "A3"}]
    a1, a3 = evaluate_replay_trace(logits, values, [], 2, .5, .5, specs, page_size=1)
    assert a1["head_coverage"] == pytest.approx([.95, .3])
    assert a3["head_coverage"] == pytest.approx([.4, .55])
    assert a1["remaining_deficit_sum"] == pytest.approx(.2)
    assert a3["remaining_deficit_sum"] == pytest.approx(.1)
    assert len(a1["head_output_absolute_errors"]) == 2


def test_replay_sparse_output_remains_stable_when_dense_mass_underflows():
    rows = evaluate_replay_trace(np.array([[10000., -10000., -10001.]]),
                                np.array([[100.], [2.], [2.]]), [1], 1, .9, 1/3,
                                [{"name": "a1", "algorithm": "A1"}], page_size=1)
    assert rows[0]["head_output_absolute_errors"] == pytest.approx([98.])
    assert rows[0]["head_output_relative_errors"] == pytest.approx([.98])
