import numpy as np
import pytest
import json
import subprocess
import sys

from gqa_probe.evaluate import evaluate_trace, summarize


def test_evaluation_matches_hand_derived_coverage_and_same_v_error():
    mass = np.array([[.60, 0], [.35, .30], [.05, .25], [0, .45]])
    # Finite logits: numerical zero in the fixture is represented by exp(-10000).
    logits = np.log(np.maximum(mass.T, np.exp(-700)))
    values = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    rows = evaluate_trace(logits, values, page_size=1, budgets=[.5], targets=[.5],
                          batches=[4], sink_pages=0, local_tokens=0)
    a1 = next(r for r in rows if r["algorithm"] == "A1")
    a3 = next(r for r in rows if r["algorithm"] == "A3")
    assert a1["remaining_deficit_sum"] == pytest.approx(.2)
    assert a3["remaining_deficit_sum"] == pytest.approx(.1)
    assert a1["selected_pages"] == a3["selected_pages"] == 2
    assert a3["avg_retained_mass"] < a1["avg_retained_mass"]
    assert len(a3["head_output_relative_errors"]) == 2


def test_summary_pools_heads_and_does_not_average_query_percentiles():
    rows = [dict(algorithm="A1", batch_size=1, budget_fraction=.1, target=.8,
                 head_coverage=[.1, .3], head_output_relative_errors=[1., 3.],
                 head_output_absolute_errors=[2., 4.], remaining_deficit_sum=1.2),
            dict(algorithm="A1", batch_size=1, budget_fraction=.1, target=.8,
                 head_coverage=[.8, 1.], head_output_relative_errors=[2., 4.],
                 head_output_absolute_errors=[3., 5.], remaining_deficit_sum=0.)]
    group = summarize(rows)[0]
    assert group["p05_query_head_mass"] == pytest.approx(.13)
    assert group["output_relative_error_p95"] == pytest.approx(3.85)
    assert group["records"] == 2


def test_summary_pools_dynamic_batches_with_unequal_round_counts():
    base = dict(algorithm="A3-B", batch_size=4, budget_fraction=.1, target=.8,
                head_coverage=[.5], head_output_relative_errors=[.1],
                head_output_absolute_errors=[.2])
    rows = [dict(base, batch_selection_change_rate_mean=1.,
                 batches=[{"batch_selection_change_rate": 1.}]),
            dict(base, batch_selection_change_rate_mean=0.,
                 batches=[{"batch_selection_change_rate": 0.}] * 3)]
    assert summarize(rows)[0]["batch_selection_change_rate_mean"] == pytest.approx(.25)


def test_evaluation_refuses_to_overwrite_existing_result(tmp_path):
    traces, output = tmp_path / "traces", tmp_path / "results"
    traces.mkdir()
    output.mkdir()
    (traces / "manifest.json").write_text(json.dumps({"records": []}))
    original = b"existing scientific observations\n"
    records = output / "records.jsonl"
    records.write_bytes(original)
    process = subprocess.run([sys.executable, "-m", "gqa_probe.evaluate", str(traces),
                              "--output", str(output)], capture_output=True)
    assert process.returncode != 0
    assert records.read_bytes() == original
