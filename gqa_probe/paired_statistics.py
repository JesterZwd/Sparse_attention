"""Official RULER scoring and prompt-level paired quality statistics.

Shadow heads are pooled only *within* one prompt/specification to obtain that
prompt's tail error. Inferential inputs are those prompt summaries, never
individual heads, layers, groups, or decode steps.
"""

from collections import defaultdict
from numbers import Integral

import numpy as np


def score_prediction(prediction: str, references: list[str]) -> dict:
    """Score RULER v1 NIAH case-insensitive reference substrings on [0, 1].

    Each reference contributes equally to the official fractional score. The
    additional complete-answer flag requires every reference to match; it is
    not a replacement for the official score. No token-boundary, punctuation,
    or whitespace normalization is applied.
    """
    if not references:
        raise ValueError("references must contain at least one value")
    prediction = prediction.lower()
    matches = [reference.lower() in prediction for reference in references]
    return {"official_score": sum(matches) / len(matches),
            "complete_correct": all(matches)}


_SHADOW_KEY = ("sample_id", "task", "split", "spec", "config_id")
_SHADOW_OBSERVATIONS = {
    "layer", "step", "kv_group", "head_coverage",
    "head_output_absolute_errors", "head_output_relative_errors", "pages",
    "head_output_absolute_errors_float64_audit", "head_output_relative_errors_float64_audit",
    "valid_tokens", "budget_pages", "available_pages",
    "original_target_deficit", "gate_hit", "swaps", "candidate_pairs",
    "executed_candidate_pairs", "gold_span_recall", "gold_token_recall",
    "gold_page_recall", "all_gold_spans_retained", "all_gold_pages_retained", "work",
}


def _finite_scalar(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _head_vector(record, name):
    try:
        values = np.asarray(record[name], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite head observations") from exc
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite head observations")
    return values


def summarize_shadow(records: list[dict]) -> list[dict]:
    """Return one summary per (sample, task, split, spec, optional config_id).

    P95/P99 and coverage P05 pool all observed query heads *inside that
    prompt*. Scalar group observations (deficit, page counts, gates, available
    gold recalls) are averaged across callback records. ``pages`` may be the
    selected page-ID list or its scalar count. Counts and numeric ``work``
    counters are summed. Missing gold recalls produce ``None``, never NaN.

    Other fields are invariant metadata, preserved verbatim; a change within a
    group raises ValueError so two configurations cannot silently be mixed.
    Use the resulting per-prompt P99 values as bootstrap inputs; do not pool
    heads across prompts when evaluating the prespecified hypotheses.
    """
    groups = defaultdict(list)
    for record in records:
        try:
            key = tuple(record[name] for name in _SHADOW_KEY[:-1]) + (
                record.get("config_id"),)
        except KeyError as exc:
            raise ValueError(f"missing shadow identity: {exc.args[0]}") from exc
        groups[key].append(record)

    summaries = []
    for group in groups.values():
        metadata = {name: value for name, value in group[0].items()
                    if name not in _SHADOW_OBSERVATIONS}
        coverage, absolute, relative = [], [], []
        work = defaultdict(float)
        for record in group:
            current_metadata = {name: value for name, value in record.items()
                                if name not in _SHADOW_OBSERVATIONS}
            if current_metadata != metadata:
                raise ValueError("inconsistent shadow metadata within prompt/spec/config group")
            heads = [_head_vector(record, name) for name in (
                "head_coverage", "head_output_absolute_errors", "head_output_relative_errors")]
            if len({values.size for values in heads}) != 1:
                raise ValueError("head coverage and error arrays must have identical lengths")
            coverage.append(heads[0])
            absolute.append(heads[1])
            relative.append(heads[2])
            for name, value in record.get("work", {}).items():
                work[name] += _finite_scalar(value, f"work.{name}")

        coverage = np.concatenate(coverage)
        absolute = np.concatenate(absolute)
        relative = np.concatenate(relative)
        result = dict(metadata)
        result.update(
            absolute_error_p95=float(np.quantile(absolute, .95)),
            absolute_error_p99=float(np.quantile(absolute, .99)),
            relative_error_p95=float(np.quantile(relative, .95)),
            relative_error_p99=float(np.quantile(relative, .99)),
            coverage_p05=float(np.quantile(coverage, .05)),
            n_records=len(group), n_heads=int(coverage.size), sum_work=dict(work))
        for output, source in (("mean_deficit", "original_target_deficit"),
                               ("gate_hit_rate", "gate_hit")):
            result[output] = float(np.mean([_finite_scalar(row[source], source) for row in group]))
        result["mean_pages"] = float(np.mean([
            _finite_scalar(row["pages"], "pages") if np.ndim(row["pages"]) == 0 else len(row["pages"])
            for row in group]))
        for source in ("swaps", "candidate_pairs", "executed_candidate_pairs"):
            result[f"sum_{source}"] = sum(_finite_scalar(row[source], source) for row in group)
        for source in ("gold_span_recall", "gold_token_recall", "gold_page_recall"):
            values = [_finite_scalar(row[source], source) for row in group
                      if row.get(source) is not None]
            result[f"mean_{source}"] = float(np.mean(values)) if values else None
        audit_names = ("head_output_absolute_errors_float64_audit",
                       "head_output_relative_errors_float64_audit")
        if any(name in row for row in group for name in audit_names):
            for label, name in zip(("absolute", "relative"), audit_names):
                values = [_head_vector(row, name) for row in group]
                if any(len(value) != len(row["head_coverage"]) for value, row in zip(values, group)):
                    raise ValueError("audit head errors must match the main head observations")
                pooled = np.concatenate(values)
                for q in (.95, .99):
                    result[f"{label}_error_p{int(q * 100)}_float64_audit"] = float(np.quantile(pooled, q))
        summaries.append(result)
    return summaries


def paired_bootstrap(rows: list[dict], baseline: str, candidate: str, metric: str,
                     confidence: float = .975, repetitions: int = 10000,
                     seed: int = 20261118, method_key: str = "method") -> dict:
    """Task-stratified percentile bootstrap of paired prompt measurements.

    Rows for the two requested methods must match exactly on (task, sample_id)
    and have one finite scalar metric each. Rows for other methods are ignored.
    The caller must filter configuration/split before calling this function;
    duplicate prompt/method rows are rejected rather than silently combined.

    Each task resamples its original number of prompts with replacement. The
    same indices select both methods; each replicate then macro-averages task
    means with equal task weight. Delta is candidate minus baseline. Relative
    changes are fractions, computed against each replicate's baseline mean.
    A zero point denominator gives relative_delta=None. If any resampled
    baseline is zero, both relative interval endpoints are None: undefined
    replicates are never silently discarded. No ratio is replaced with NaN.

    ``inferential_reliability="low"`` warns when a task has only one prompt;
    otherwise it is ``"not_assessed"`` (no sample-size power claim is made).
    """
    confidence = _finite_scalar(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    repetitions = int(repetitions)
    if baseline == candidate:
        raise ValueError("baseline and candidate must be different methods")

    methods = {baseline: {}, candidate: {}}
    for row in rows:
        if method_key not in row:
            raise ValueError(f"missing method field: {method_key}")
        method = row[method_key]
        if method not in methods:
            continue
        try:
            key = (row["task"], row["sample_id"])
            value = row[metric]
        except KeyError as exc:
            raise ValueError(f"missing paired field: {exc.args[0]}") from exc
        if any(value is None or value == "" for value in key):
            raise ValueError("missing task or sample_id in paired row")
        try:
            duplicate = key in methods[method]
        except TypeError as exc:
            raise ValueError("task and sample_id must be hashable identifiers") from exc
        if duplicate:
            raise ValueError(f"duplicate paired prompt for method {method}: {key}")
        methods[method][key] = _finite_scalar(value, metric)

    baseline_values, candidate_values = methods[baseline], methods[candidate]
    if not baseline_values and not candidate_values:
        raise ValueError("empty paired data for requested methods")
    if baseline_values.keys() != candidate_values.keys():
        raise ValueError("baseline and candidate must exactly match task/sample_id pairs")

    # Sorting identities makes a seed reproducible even after input row order
    # changes; both methods are aligned by identity, never by insertion order.
    task_values = defaultdict(list)
    for key in sorted(baseline_values, key=lambda pair: tuple(
            (type(value).__name__, repr(value)) for value in pair)):
        task_values[key[0]].append((baseline_values[key], candidate_values[key]))

    rng = np.random.default_rng(seed)
    task_means = []
    resampled_means = np.zeros((repetitions, 2), dtype=np.float64)
    for pairs in task_values.values():
        values = np.asarray(pairs, dtype=np.float64)
        task_means.append(values.mean(axis=0))
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        resampled_means += values[indices].mean(axis=1)
    means = np.asarray(task_means).mean(axis=0)
    resampled_means /= len(task_values)
    deltas = resampled_means[:, 1] - resampled_means[:, 0]
    point_delta = float(means[1] - means[0])
    tails = ((1 - confidence) / 2, (1 + confidence) / 2)
    ci_low, ci_high = np.quantile(deltas, tails)

    relative_delta = point_delta / float(means[0]) if means[0] != 0 else None
    zero_denominator = np.any(resampled_means[:, 0] == 0)
    if zero_denominator:
        relative_ci_low = relative_ci_high = None
    else:
        relative_ci_low, relative_ci_high = map(float, np.quantile(
            deltas / resampled_means[:, 0], tails))

    n_prompts = len(baseline_values)
    singleton_task = any(len(pairs) == 1 for pairs in task_values.values())
    warnings = []
    if n_prompts == 1:
        warnings.append("Only one observed prompt; the bootstrap cannot estimate sampling variability.")
    elif singleton_task:
        warnings.append("At least one task has only one observed prompt; its within-task variability is not estimable.")
    if zero_denominator:
        warnings.append("Relative interval undefined because a resampled baseline mean is zero.")
    return dict(
        baseline=baseline, candidate=candidate, metric=metric,
        baseline_mean=float(means[0]), candidate_mean=float(means[1]),
        point_delta=point_delta, ci_low=float(ci_low), ci_high=float(ci_high),
        relative_delta=relative_delta, relative_ci_low=relative_ci_low,
        relative_ci_high=relative_ci_high, n_prompts=n_prompts,
        n_tasks=len(task_values), confidence=confidence, repetitions=repetitions,
        seed=int(seed), inferential_reliability="low" if singleton_task else "not_assessed",
        warning=" ".join(warnings) if warnings else None)
