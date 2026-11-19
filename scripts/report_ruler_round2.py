"""Audit complete Round 2 runs and report paired quality and selector cost."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap, score_prediction
from gqa_probe.replay_round2 import benchmark_cases
from gqa_probe.ruler_experiment import FOCUS, SAMPLE_LAYERS, SAMPLE_STEPS, TASKS, _sha, shadow_specs
from gqa_probe.selection import protected_pages


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(path, phase, count):
    completion = read_json(path / "complete.json")
    config = read_json(path / "config.json")
    if (completion["status"], completion["phase"], completion["completed_prompts"]) != ("complete", phase, count):
        raise ValueError(f"incomplete {phase} cohort")
    if config["run_id"] != completion["run_id"]:
        raise ValueError(f"{phase} run identity mismatch")
    for name, expected in completion["files_sha256"].items():
        if _sha(path / name) != expected:
            raise ValueError(f"{phase} file checksum mismatch: {name}")
    with gzip.open(path / "per_prompt_predictions.jsonl.gz", "rt") as stream:
        predictions = [json.loads(line) for line in stream]
    summaries = read_json(path / "per_prompt_shadow.json")
    for row in predictions + summaries:
        if row["split"] != phase or row["run_id"] != completion["run_id"]:
            raise ValueError(f"{phase} record provenance mismatch")
    return config, completion, predictions, summaries


def audit_test(config, completion, predictions, shadows, selection):
    identities = {(row["task"], row["sample_id"]) for row in predictions}
    if Counter(task for task, _ in identities) != Counter({task: 128 for task in TASKS}):
        raise ValueError("test task counts differ from the frozen 512 prompts")
    expected = {("dense", "DenseBF16"), ("dense", "DenseKV4")} | {
        (config_id, method) for config_id in FOCUS for method in ("A1", "SWAP")}
    records = {}
    for row in predictions:
        key = (row["task"], row["sample_id"], row["config_id"], row["method"])
        if key in records:
            raise ValueError(f"duplicate test prediction: {key}")
        records[key] = row
        tokens = row["generated_tokens"]
        stats = row["attention_stats"]
        if (not 1 <= tokens <= 128 or tokens != len(row["token_ids"])
                or row["input_tokens"] + 128 > 16384
                or row["truncated"] != (not row["eos_reached"] and tokens == 128)
                or (not row["eos_reached"] and tokens != 128)
                or stats["decode_layer_calls"] != (tokens - 1) * 36
                or stats["kv_group_calls"] != (tokens - 1) * 36 * 8):
            raise ValueError(f"test decode length or layer accounting mismatch: {key}")
        if any(row[k] != v for k, v in score_prediction(row["text"], row["references"]).items()):
            raise ValueError(f"prediction score mismatch: {key}")
        if row["method"] in {"A1", "SWAP"}:
            target, fraction = FOCUS[row["config_id"]]
            budgets = [int(((row["input_tokens"] + step + 32) // 32) * fraction)
                       for step in range(tokens - 1)]
            if stats["selected_pages"] != sum(budgets) * 36 * 8:
                raise ValueError(f"sparse physical page accounting mismatch: {key}")
            if row["target"] != target or row["budget_fraction"] != fraction:
                raise ValueError(f"sparse configuration mismatch: {key}")
            if row["method"] == "SWAP" and any(row[k] != v for k, v in selection["selected"].items()):
                raise ValueError(f"unfrozen swap candidate: {key}")
    if set(records) != {(task, sample, cfg, method) for task, sample in identities for cfg, method in expected}:
        raise ValueError("test prediction set is incomplete")
    for task, sample in identities:
        kv4 = records[task, sample, "dense", "DenseKV4"]
        for cfg, method in expected - {("dense", "DenseBF16"), ("dense", "DenseKV4")}:
            sparse = records[task, sample, cfg, method]
            if (sparse["token_ids"][0] != kv4["token_ids"][0]
                    or sparse["references"] != kv4["references"]
                    or sparse["input_tokens"] != kv4["input_tokens"]
                    or not sparse["prefill_reused"]):
                raise ValueError(f"KV4 methods did not share the same prefill: {task}/{sample}")
    observed = Counter((r["task"], r["sample_id"], r["spec"]) for r in shadows)
    expected_shadows = {(task, sample, spec["name"]) for task, sample in identities
                        for spec in shadow_specs("test", selection)}
    if set(observed) != expected_shadows or any(count != 1 for count in observed.values()):
        raise ValueError("test shadow set does not exactly match the paired cohort")
    for row in shadows:
        kv4 = records[row["task"], row["sample_id"], "dense", "DenseKV4"]
        expected_count = 48 * sum(step < kv4["generated_tokens"] - 1 for step in SAMPLE_STEPS)
        if row["n_records"] != expected_count or row["n_heads"] != 4 * expected_count:
            raise ValueError("prespecified shadow observations are incomplete")
    if len(predictions) != completion["completed_generations"] or len(shadows) != completion["shadow_prompt_summaries"]:
        raise ValueError("test aggregate counts disagree with completion marker")
    return dict(test_prompts=len(identities), predictions=len(predictions),
                shadow_prompt_summaries=len(shadows), no_decode_prompts=completion["no_decode_prompts"],
                run_id=config["run_id"], status="passed")


def mean_available(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def shadow_details(path, complete, shadows):
    """Audit raw masks and create prompt-level retained-mass and layer summaries."""
    index = {(r["task"], r["sample_id"], r["spec"]): r for r in shadows}
    layer_rows = []
    for key, bundle_hash in complete["prompt_bundle_sha256"].items():
        directory = path / "prompts" / key
        if _sha(directory / "bundle.json") != bundle_hash:
            raise ValueError("prompt bundle hash mismatch")
        bundle = read_json(directory / "bundle.json")
        for name, digest in bundle["files_sha256"].items():
            if _sha(directory / name) != digest:
                raise ValueError(f"prompt bundle payload hash mismatch: {key}/{name}")
        groups = defaultdict(list)
        seen = set()
        with gzip.open(directory / "shadow.jsonl.gz", "rt") as stream:
            for line in stream:
                record = json.loads(line)
                spec = record["spec"]
                sample = record["task"], record["sample_id"], spec
                observation = spec, record["layer"], record["step"], record["kv_group"]
                if (record["run_id"] != complete["run_id"] or sample not in index
                        or sample[:2] != (bundle["task"], bundle["sample_id"])):
                    raise ValueError("raw shadow identity does not match completed run")
                pages = set(record["pages"])
                length = record["valid_tokens"]
                budget = int(((length + 31) // 32) * record["budget_fraction"])
                if (len(record["pages"]) != budget or len(pages) != budget
                        or not set(protected_pages(length)).issubset(pages)
                        or any(page < 0 or page >= (length + 31) // 32 for page in pages)):
                    raise ValueError("raw shadow failed budget and protection audit")
                if observation in seen:
                    raise ValueError("duplicate layer/step/group observation in a prompt")
                seen.add(observation)
                groups[spec].append(record)
        for spec, records in groups.items():
            summary = index[bundle["task"], bundle["sample_id"], spec]
            if len(records) != summary["n_records"]:
                raise ValueError("raw shadow and summary counts differ")
            summary["mean_retained_mass"] = float(np.mean([r["head_coverage"] for r in records]))
            for kind in ("spans", "pages"):
                summary[f"all_gold_{kind}_retained_rate"] = mean_available(
                    [r[f"all_gold_{kind}_retained"] for r in records])
            for layer in SAMPLE_LAYERS:
                layer_records = [r for r in records if r["layer"] == layer]
                if not layer_records:
                    raise ValueError("missing prespecified shadow layer")
                item = dict(task=bundle["task"], sample_id=bundle["sample_id"], spec=spec,
                            setting=summary["config_id"], layer=layer)
                for label in ("absolute", "relative"):
                    errors = np.concatenate([r[f"head_output_{label}_errors"] for r in layer_records])
                    item[f"{label}_error_p95"], item[f"{label}_error_p99"] = map(float, np.quantile(errors, [.95, .99]))
                layer_rows.append(item)
    return layer_rows


def quality_table(predictions, shadows):
    indexed = {(r["task"], r["sample_id"], r["spec"]): r for r in shadows}
    rows = []
    for task in ["all", *TASKS]:
        for config, method in [("dense", "DenseBF16"), ("dense", "DenseKV4"),
                               *[(cfg, m) for cfg in FOCUS for m in ("A1", "SWAP")]]:
            selected = [r for r in predictions if (r["config_id"], r["method"]) == (config, method)
                        and (task == "all" or r["task"] == task)]
            reference = [indexed[r["task"], r["sample_id"], r["spec"]] for r in selected if "spec" in r]
            row = dict(task=task, setting=config, method=method, prompts=len(selected),
                       complete_correct=sum(r["complete_correct"] for r in selected),
                       official_score=mean_available([r["official_score"] for r in selected]),
                       complete_accuracy=mean_available([r["complete_correct"] for r in selected]),
                       truncated=sum(r["truncated"] for r in selected),
                       mean_generated_tokens=mean_available([r["generated_tokens"] for r in selected]))
            for name in ("gold_page_recall", "gold_span_recall", "gold_token_recall"):
                row[f"closed_loop_{name}"] = mean_available([r["attention_stats"][name] for r in selected])
            row["closed_loop_all_gold_spans_retained_rate"] = mean_available([
                stats["all_gold_spans_group_hits"] / stats["gold_group_count"]
                for r in selected if (stats := r["attention_stats"])["gold_group_count"]])
            totals = Counter()
            for prediction in selected:
                totals.update({k: prediction["attention_stats"][k] for k in
                               ("selected_pages", "kv_group_calls", "gate_hits", "swaps",
                                "candidate_pairs", "executed_candidate_pairs")})
            for name in ("selected_pages", "gate_hits", "swaps", "candidate_pairs", "executed_candidate_pairs"):
                row[f"closed_loop_mean_{name}_per_group"] = (
                    totals[name] / totals["kv_group_calls"] if totals["kv_group_calls"] else None)
            for name in ("absolute_error_p95", "absolute_error_p99", "relative_error_p95",
                         "relative_error_p99", "coverage_p05", "mean_deficit", "mean_pages",
                         "mean_gold_page_recall", "mean_gold_span_recall", "mean_gold_token_recall",
                         "mean_retained_mass", "all_gold_spans_retained_rate", "all_gold_pages_retained_rate"):
                row[f"dense_kv4_replay_{name}"] = mean_available([r.get(name) for r in reference])
            rows.append(row)
    return rows


def intervals(predictions, shadows, selection):
    result = {"quantization": {}}
    for metric in ("official_score", "complete_correct"):
        result["quantization"][metric] = paired_bootstrap(predictions, "DenseBF16", "DenseKV4", metric)
    for config in FOCUS:
        subset = [r for r in predictions if r["config_id"] == config]
        stats = {metric: paired_bootstrap(subset, "A1", "SWAP", metric)
                 for metric in ("official_score", "complete_correct")}
        specs = [r["name"] for r in shadow_specs("test", selection) if r["config_id"] == config]
        for metric in ("absolute_error_p95", "absolute_error_p99", "relative_error_p95", "relative_error_p99"):
            stats[metric] = paired_bootstrap(shadows, *specs, metric, method_key="spec")
        with_mass = [r for r in shadows if r.get("mean_retained_mass") is not None]
        if with_mass:
            stats["mean_retained_mass"] = paired_bootstrap(with_mass, *specs, "mean_retained_mass", method_key="spec")
        result[config] = stats
    for config, baseline, candidate in [
        ("quantization", "DenseBF16", "DenseKV4"), *[(c, "A1", "SWAP") for c in FOCUS]
    ]:
        relevant = [r for r in predictions if r["config_id"] == ("dense" if config == "quantization" else config)]
        values = {(r["task"], r["sample_id"], r["method"]): r["complete_correct"] for r in relevant}
        discordance = Counter()
        for task, sample, method in values:
            if method == baseline:
                before, after = values[task, sample, baseline], values[task, sample, candidate]
                discordance["candidate_only" if after and not before else
                            "baseline_only" if before and not after else
                            "both_correct" if before else "both_incorrect"] += 1
        result[config]["complete_correct"]["paired_counts"] = dict(discordance)
        result[config]["complete_correct"]["zero_observed_discordance"] = not (
            discordance["candidate_only"] or discordance["baseline_only"])
    return result


def decide(primary, median_ratio):
    accuracy = primary["complete_correct"]
    tail = primary["absolute_error_p99"]
    tests = dict(accuracy_superiority=accuracy["ci_low"] > 0,
                 absolute_p99_noninferiority=tail["relative_ci_high"] is not None and tail["relative_ci_high"] <= .01,
                 absolute_p99_improvement=tail["relative_delta"] is not None and tail["relative_delta"] <= -.01
                 and tail["ci_high"] < 0,
                 accuracy_noninferiority=accuracy["ci_low"] >= -.01,
                 selector_cost=median_ratio <= 1.25)
    route_a = tests["accuracy_superiority"] and tests["absolute_p99_noninferiority"]
    route_b = tests["absolute_p99_improvement"] and tests["accuracy_noninferiority"]
    return dict(tests=tests, quality_route_accuracy=route_a, quality_route_tail=route_b,
                median_cpu_cost_ratio=median_ratio, passed=tests["selector_cost"] and (route_a or route_b))


def cost_cases(test_path, config, predictions):
    ids = {(r["task"], r["sample_id"]) for r in config["cost_cohort"]}
    if Counter(task for task, _ in ids) != Counter({task: 16 for task in TASKS}):
        raise ValueError("expected 16 frozen cost prompts per task")
    no_decode = {(r["task"], r["sample_id"]) for r in predictions
                 if r["method"] == "DenseKV4" and r["generated_tokens"] == 1}
    visits = set()
    cases = []
    for file in sorted((test_path / "prompts").glob("*/cost_mass/*.npz")):
        with np.load(file, allow_pickle=False) as payload:
            mass = payload["mass"]
            meta = json.loads(payload["metadata_json"].item())
        identity = meta["task"], meta["sample_id"]
        key = (*identity, meta["layer"])
        if (identity not in ids or meta["split"] != "test" or meta["step"] != 0
                or meta["run_id"] != config["run_id"] or key in visits
                or mass.shape != (8, (meta["valid_tokens"] + 31) // 32, 4)
                or not np.isfinite(mass).all() or np.any(mass < 0)
                or not np.allclose(mass.sum(axis=1), 1, atol=1e-9, rtol=0)):
            raise ValueError(f"invalid cost mass sample: {file}")
        visits.add(key)
        for group, matrix in enumerate(mass):
            info = dict(task=meta["task"], sample_id=meta["sample_id"], layer=meta["layer"],
                        step=0, kv_group=group)
            for target, fraction in FOCUS.values():
                cases.append((matrix, meta["protected"], int(len(matrix) * fraction), target, fraction, info))
    expected = {(*identity, layer) for identity in ids - no_decode for layer in SAMPLE_LAYERS}
    if visits != expected or not cases:
        raise ValueError("cost samples do not match the frozen eligible cohort")
    return cases, len(visits), len(ids & no_decode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("results/ruler_round2_calibration"))
    parser.add_argument("--test", type=Path, default=Path("results/ruler_round2_test"))
    parser.add_argument("--output", type=Path, default=Path("results/ruler_round2_report"))
    args = parser.parse_args()
    calibration, _, cp, cs = load_run(args.calibration, "calibration", 128)
    config, complete, predictions, shadows = load_run(args.test, "test", 512)
    selected_file = args.calibration / "selection.json"
    selection = read_json(selected_file)
    if (config["selection_sha256"] != _sha(selected_file) or config["quantizer"] != calibration["quantizer"]
            or {(r["task"], r["sample_id"]) for r in cp} & {(r["task"], r["sample_id"]) for r in predictions}):
        raise ValueError("held-out run must use the frozen calibration candidate and separate prompts")
    audit = audit_test(config, complete, predictions, shadows, selection)
    layer_rows = shadow_details(args.test, complete, shadows)
    costs, npz_count, skipped = cost_cases(args.test, config, predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "quality.csv", quality_table(predictions, shadows))
    if layer_rows:
        write_csv(args.output / "layer_tails_per_prompt.csv", layer_rows)
    report_intervals = intervals(predictions, shadows, selection)
    write_json(args.output / "paired_intervals.json", report_intervals)
    write_json(args.output / "audit.json", dict(audit, cost_npz_files=npz_count,
                                               cost_no_decode_prompts=skipped,
                                               cost_group_matrices=npz_count * 8,
                                               calibration_shadow_summaries=len(cs)))
    benchmark_cases(costs, [dict(name="a1", algorithm="A1"),
                           dict(name="swap", algorithm="SWAP", **selection["selected"])], args.output)
    timing = read_json(args.output / "timing_summary.json")["summary"]
    target, fraction = FOCUS["primary"]
    median_ratio = next(r["median_ratio_to_a1"] for r in timing if r["spec"] == "swap"
                        and (r["target"], r["budget_fraction"]) == (target, fraction))
    write_json(args.output / "decision.json", decide(report_intervals["primary"], median_ratio))
    print(f"Round 2 verification and report complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
