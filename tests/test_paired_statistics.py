"""Prompt-cluster inference tests with hand-derived reference answers."""

import pytest


def test_scoring_keeps_official_fractional_credit_separate_from_complete_answers():
    from gqa_probe.paired_statistics import score_prediction

    assert score_prediction("First: ALPHA; second: beta.", ["alpha", "BETA", "gamma", "delta"]) == {
        "official_score": .5,
        "complete_correct": False,
    }
    assert score_prediction("ALPHABET beta", ["alpha", "BETA"]) == {
        "official_score": 1.,
        "complete_correct": True,
    }
    assert score_prediction("no values", ["alpha"]) == {
        "official_score": 0.,
        "complete_correct": False,
    }


def test_scoring_refuses_an_empty_reference_set():
    from gqa_probe.paired_statistics import score_prediction

    with pytest.raises(ValueError, match="reference"):
        score_prediction("anything", [])


def _shadow(**changes):
    """One complete callback row, including runner-provided identity."""
    row = dict(sample_id="prompt-1", task="multi-key", split="calibration",
               spec="A1", config_id="primary", budget_fraction=.2, target=.8,
               layer=0, step=0, kv_group=0, head_coverage=[.1],
               head_output_absolute_errors=[0.], head_output_relative_errors=[0.],
               pages=10, original_target_deficit=.5, gate_hit=True, swaps=1,
               candidate_pairs=32, executed_candidate_pairs=32,
               gold_span_recall=.5, gold_token_recall=.25, gold_page_recall=1.,
               work={"pairs": 32, "heads": 8})
    return dict(row, **changes)


def test_shadow_summary_pools_heads_within_prompt_and_preserves_callback_counts():
    from gqa_probe.paired_statistics import summarize_shadow

    records = [_shadow(),
               _shadow(layer=7, step=3, head_coverage=[.3, .8, 1.],
                       head_output_absolute_errors=[10., 20., 30.],
                       head_output_relative_errors=[.1, .2, .3], pages=20,
                       original_target_deficit=.1, gate_hit=False, swaps=2,
                       candidate_pairs=16, executed_candidate_pairs=24,
                       gold_span_recall=None, gold_token_recall=.75,
                       gold_page_recall=.5, work={"pairs": 24, "heads": 16})]
    summary, = summarize_shadow(records)
    assert {name: summary[name] for name in
            ("sample_id", "task", "split", "spec", "config_id", "budget_fraction", "target")} == {
        "sample_id": "prompt-1", "task": "multi-key", "split": "calibration",
        "spec": "A1", "config_id": "primary", "budget_fraction": .2, "target": .8,
    }
    # The four observed heads give linearly interpolated P95/P99, independently
    # of how many heads appeared in either callback record.
    assert summary["absolute_error_p95"] == pytest.approx(28.5)
    assert summary["absolute_error_p99"] == pytest.approx(29.7)
    assert summary["relative_error_p95"] == pytest.approx(.285)
    assert summary["relative_error_p99"] == pytest.approx(.297)
    assert summary["coverage_p05"] == pytest.approx(.13)
    assert summary["mean_deficit"] == pytest.approx(.3)
    assert summary["mean_pages"] == 15.
    assert summary["gate_hit_rate"] == .5
    assert summary["sum_swaps"] == 3
    assert summary["sum_candidate_pairs"] == 48
    assert summary["sum_executed_candidate_pairs"] == 56
    assert summary["mean_gold_span_recall"] == .5
    assert summary["mean_gold_token_recall"] == .5
    assert summary["mean_gold_page_recall"] == .75
    assert summary["n_records"] == 2
    assert summary["n_heads"] == 4
    assert summary["sum_work"] == {"pairs": 56, "heads": 24}


def test_shadow_summary_separates_prompts_specs_and_configurations():
    from gqa_probe.paired_statistics import summarize_shadow

    records = [_shadow(), _shadow(spec="SWAP"),
               _shadow(sample_id="prompt-2", gold_span_recall=None,
                       gold_token_recall=None, gold_page_recall=None),
               _shadow(config_id="stress", budget_fraction=.1, target=.9)]
    summaries = summarize_shadow(records)
    assert len(summaries) == 4
    assert {(row["sample_id"], row["spec"], row["config_id"]) for row in summaries} == {
        ("prompt-1", "A1", "primary"), ("prompt-1", "SWAP", "primary"),
        ("prompt-2", "A1", "primary"), ("prompt-1", "A1", "stress"),
    }
    row = next(row for row in summaries if row["sample_id"] == "prompt-2")
    assert row["mean_gold_span_recall"] is None
    assert row["mean_gold_token_recall"] is None
    assert row["mean_gold_page_recall"] is None
    assert summarize_shadow([]) == []


def test_shadow_summary_refuses_to_mix_config_metadata_under_the_same_key():
    from gqa_probe.paired_statistics import summarize_shadow

    with pytest.raises(ValueError, match="metadata|target"):
        summarize_shadow([_shadow(), _shadow(target=.9)])


def test_shadow_summary_accepts_page_id_lists_and_varying_decode_observations():
    from gqa_probe.paired_statistics import summarize_shadow

    # Actual backend callbacks contain the selected physical page IDs and
    # changing history lengths; those are observations, not configuration.
    records = [
        _shadow(pages=[0, 8], valid_tokens=321, budget_pages=2, available_pages=11,
                all_gold_spans_retained=False, all_gold_pages_retained=False),
        _shadow(step=31, pages=[0, 3, 8], valid_tokens=352, budget_pages=3,
                available_pages=11, all_gold_spans_retained=True,
                all_gold_pages_retained=True),
    ]
    summary, = summarize_shadow(records)
    assert summary["mean_pages"] == 2.5
    assert summary["n_records"] == 2


def test_shadow_summary_keeps_optional_float64_audit_errors_separate_from_main_errors():
    from gqa_probe.paired_statistics import summarize_shadow

    records = [
        _shadow(head_output_absolute_errors_float64_audit=[10.],
                head_output_relative_errors_float64_audit=[1.]),
        _shadow(step=1, head_output_absolute_errors_float64_audit=[20.],
                head_output_relative_errors_float64_audit=[2.]),
    ]
    summary, = summarize_shadow(records)
    assert summary["absolute_error_p99"] == 0.
    assert summary["absolute_error_p95_float64_audit"] == pytest.approx(19.5)
    assert summary["absolute_error_p99_float64_audit"] == pytest.approx(19.9)
    assert summary["relative_error_p95_float64_audit"] == pytest.approx(1.95)
    assert summary["relative_error_p99_float64_audit"] == pytest.approx(1.99)


def test_shadow_summary_refuses_partial_audits_within_a_prompt():
    from gqa_probe.paired_statistics import summarize_shadow

    with pytest.raises(ValueError, match="audit|head"):
        summarize_shadow([_shadow(head_output_absolute_errors_float64_audit=[1.]), _shadow()])


@pytest.mark.parametrize("changes", [
    {"head_output_relative_errors": [1., 2.]},
    {"head_output_absolute_errors": [float("nan")]},
    {"head_coverage": []},
])
def test_shadow_summary_refuses_incomplete_or_nonfinite_head_observations(changes):
    from gqa_probe.paired_statistics import summarize_shadow

    with pytest.raises(ValueError, match="head|finite"):
        summarize_shadow([_shadow(**changes)])


def test_bootstrap_stratifies_tasks_and_macro_averages_instead_of_weighting_heads_or_task_size():
    from gqa_probe.paired_statistics import paired_bootstrap

    # Task small: mean A1=0, SWAP=2. Task large: mean A1=10, SWAP=11.
    # Equal task weight therefore gives (5, 6.5), delta=1.5, not the
    # prompt-weighted delta 1.25. Repeated heads never enter this function.
    rows = [dict(sample_id="only", task="small", method="A1", metric=0.),
            dict(sample_id="only", task="small", method="SWAP", metric=2.)]
    for sample_id in ("a", "b", "c"):
        rows += [dict(sample_id=sample_id, task="large", method="A1", metric=10.),
                 dict(sample_id=sample_id, task="large", method="SWAP", metric=11.)]
    result = paired_bootstrap(rows, "A1", "SWAP", "metric", repetitions=1000)
    assert result["baseline_mean"] == 5.
    assert result["candidate_mean"] == 6.5
    assert result["point_delta"] == 1.5
    assert result["ci_low"] == result["ci_high"] == 1.5
    assert result["relative_delta"] == pytest.approx(.3)
    assert result["relative_ci_low"] == result["relative_ci_high"] == pytest.approx(.3)
    assert result["n_prompts"] == 4
    assert result["n_tasks"] == 2
    assert result["confidence"] == .975
    assert result["seed"] == 20261118
    assert result["repetitions"] == 1000


def test_bootstrap_uses_same_paired_indices_and_is_reproducible_after_row_reordering():
    from gqa_probe.paired_statistics import paired_bootstrap

    # Every candidate value is its paired baseline plus one: any resampled
    # paired mean delta must remain exactly one. Independently resampling the
    # methods (or pairing by row order) creates spurious uncertainty.
    rows = [dict(sample_id="a", task="key", spec="A1", error=1.),
            dict(sample_id="b", task="key", spec="SWAP", error=10.),
            dict(sample_id="b", task="key", spec="A1", error=9.),
            dict(sample_id="a", task="key", spec="SWAP", error=2.)]
    result = paired_bootstrap(rows, "A1", "SWAP", "error", method_key="spec")
    assert result == paired_bootstrap(list(reversed(rows)), "A1", "SWAP", "error", method_key="spec")
    assert result["baseline_mean"] == 5.
    assert result["candidate_mean"] == 6.
    assert result["point_delta"] == result["ci_low"] == result["ci_high"] == 1.
    # Relative bootstrap denominators are resampled too: two copies of a yield
    # +100%, two copies of b yield +1/9; dividing delta CI by fixed 5 is wrong.
    assert result["relative_delta"] == pytest.approx(.2)
    assert result["relative_ci_low"] == pytest.approx(1 / 9)
    assert result["relative_ci_high"] == 1.


def test_bootstrap_confidence_changes_percentile_interval_of_resampled_means():
    from gqa_probe.paired_statistics import paired_bootstrap

    rows = []
    for sample_id, delta in (("a", 0.), ("b", 0.), ("c", 4.), ("d", 4.)):
        rows += [dict(sample_id=sample_id, task="key", method="A1", metric=1.),
                 dict(sample_id=sample_id, task="key", method="SWAP", metric=1. + delta)]
    # Four paired draws give delta K, K~Binomial(4,.5). Its central 50%
    # interval is [1,3] and its central 97.5% interval is [0,4].
    middle = paired_bootstrap(rows, "A1", "SWAP", "metric", confidence=.5)
    wide = paired_bootstrap(rows, "A1", "SWAP", "metric")
    assert middle["point_delta"] == wide["point_delta"] == 2.
    assert (middle["ci_low"], middle["ci_high"]) == (1., 3.)
    assert (wide["ci_low"], wide["ci_high"]) == (0., 4.)


def test_prompt_level_p99_values_are_not_weighted_by_observed_head_counts():
    from gqa_probe.paired_statistics import paired_bootstrap, summarize_shadow

    records = [
        _shadow(sample_id="short", head_output_absolute_errors=[0.]),
        _shadow(sample_id="short", spec="SWAP", head_output_absolute_errors=[1.]),
        _shadow(sample_id="long", head_coverage=[1.] * 100,
                head_output_absolute_errors=[10.] * 100,
                head_output_relative_errors=[0.] * 100),
        _shadow(sample_id="long", spec="SWAP", head_coverage=[1.] * 100,
                head_output_absolute_errors=[11.] * 100,
                head_output_relative_errors=[0.] * 100),
    ]
    summaries = summarize_shadow(records)
    result = paired_bootstrap(summaries, "A1", "SWAP", "absolute_error_p99", method_key="spec")
    assert result["baseline_mean"] == 5.
    assert result["candidate_mean"] == 6.
    assert result["n_prompts"] == 2


@pytest.mark.parametrize("rows", [
    [dict(sample_id="a", task="key", method="A1", metric=1.)],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="b", task="key", method="SWAP", metric=1.)],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="a", task="value", method="SWAP", metric=1.)],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="a", task="key", method="A1", metric=2.),
     dict(sample_id="a", task="key", method="SWAP", metric=1.)],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="a", task="key", method="SWAP")],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="a", task="key", method="SWAP", metric=float("nan"))],
    [dict(sample_id="a", task="key", method="A1", metric=1.),
     dict(sample_id="a", task="key", method="SWAP", metric=None)],
    [],
])
def test_bootstrap_refuses_unmatched_duplicate_missing_and_nonfinite_values(rows):
    from gqa_probe.paired_statistics import paired_bootstrap

    with pytest.raises(ValueError, match="match|duplicate|missing|finite|empty"):
        paired_bootstrap(rows, "A1", "SWAP", "metric", repetitions=10)


def test_bootstrap_zero_denominators_return_none_and_one_prompt_warns_about_reliability():
    from gqa_probe.paired_statistics import paired_bootstrap

    rows = [dict(sample_id="a", task="key", method="A1", metric=0.),
            dict(sample_id="a", task="key", method="SWAP", metric=1.)]
    result = paired_bootstrap(rows, "A1", "SWAP", "metric", repetitions=10)
    assert result["point_delta"] == result["ci_low"] == result["ci_high"] == 1.
    assert result["relative_delta"] is None
    assert result["relative_ci_low"] is None
    assert result["relative_ci_high"] is None
    assert result["inferential_reliability"] == "low"
    assert "one" in result["warning"].lower()


def test_bootstrap_does_not_drop_undefined_resampled_relative_deltas():
    from gqa_probe.paired_statistics import paired_bootstrap

    rows = [dict(sample_id="a", task="key", method="A1", metric=0.),
            dict(sample_id="b", task="key", method="A1", metric=2.),
            dict(sample_id="a", task="key", method="SWAP", metric=1.),
            dict(sample_id="b", task="key", method="SWAP", metric=3.)]
    result = paired_bootstrap(rows, "A1", "SWAP", "metric", repetitions=1000)
    assert result["relative_delta"] == 1.
    assert result["relative_ci_low"] is None
    assert result["relative_ci_high"] is None


@pytest.mark.parametrize("kwargs", [
    {"confidence": 0.}, {"confidence": 1.}, {"confidence": float("nan")},
    {"repetitions": 0}, {"repetitions": 1.5},
])
def test_bootstrap_validates_inference_parameters(kwargs):
    from gqa_probe.paired_statistics import paired_bootstrap

    rows = [dict(sample_id="a", task="key", method="A1", metric=1.),
            dict(sample_id="a", task="key", method="SWAP", metric=1.)]
    with pytest.raises(ValueError, match="confidence|repetitions"):
        paired_bootstrap(rows, "A1", "SWAP", "metric", **kwargs)
