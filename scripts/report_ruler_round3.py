"""Summarize frozen round-three runs by task and paired prompt."""

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean

from gqa_probe.page_read_budget import page_history, read_ledger, summary_record
from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.summary_attention import Policy


SCORERS = ("quest_softmax", "quest_max", "centroid8")
BUDGETS = (10, 15, 20)


def ratio(a, b):
    return a / b if b else None


def optional_mean(values):
    available = [v for v in values if v is not None]
    return mean(available) if available else None


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def capacity_table(sram_total_bytes=4 * 1024 * 1024):
    rows = []
    # Every possible tail length near the nominal context.
    for length in (16384, 32768):
        for size in (32, 64):
            for tokens in range(length - size + 1, length + 1):
                history = page_history(tokens, size)
                for percent in BUDGETS:
                    for method in ("quest_softmax", "centroid8"):
                        record = summary_record(method)
                        for residency, part in (("full_scan", 0), ("static_sram", sram_total_bytes // 288)):
                            ledger = read_ledger(history, method, percent=percent, resident_bytes=part)
                            rows.append(dict(context=length, page_size=size, tokens=tokens, policy=method,
                                             percent=percent, residency=residency, sram_slice_bytes=part,
                                             record_encoded_bytes=record.total - record.summary_alignment,
                                             record_layout_bytes=record.total,
                                             resident_actual_bytes=ledger["resident_records"] * record.total,
                                             minimum_fraction_of_dense=ledger["minimum_read_bytes"] / history.dense.total,
                                             **ledger))
    return rows


def study_quality(rows):
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row["context"], row["task"], row["policy"]].append(row)
    result = []
    for (context, task, policy), samples in sorted(cohorts.items()):
        valid = [r for r in samples if r["status"] == "complete"]
        representative = samples[0]
        summary = dict(context=context, task=task, policy=policy, method=representative["method"],
                       page_size=representative["page_size"], percent=representative["percent"],
                       resident_bytes=representative["resident_bytes"], prompts=len(valid), infeasible=len(samples)-len(valid))
        if valid:
            traffic = [r["byte_ledger"].get("total_read_bytes") for r in valid]
            summary.update(official_score=mean(r["official_score"] for r in valid),
                           complete_accuracy=mean(r["complete_correct"] for r in valid),
                           complete_count=sum(r["complete_correct"] for r in valid),
                           truncated=sum(r["truncated"] for r in valid),
                           gold_page_recall=optional_mean(r["gold_page_recall"] for r in valid),
                           decode_tokens=mean(r["decode_tokens"] for r in valid),
                           pages_per_group=optional_mean(ratio(r["selected_pages"], r["kv_group_calls"]) for r in valid),
                           read_bytes_per_answer=optional_mean(traffic),
                           read_bytes_per_decode=optional_mean(ratio(b, r["decode_tokens"])
                                                                for r, b in zip(valid, traffic) if b is not None),
                           ratio_to_dense=optional_mean(ratio(r["byte_ledger"]["total_read_bytes"],
                                                              r["byte_ledger"]["dense_read_bytes"])
                                                        for r in valid if "total_read_bytes" in r["byte_ledger"]))
        result.append(summary)
    return result


def calibration_findings(quality, diagnostic_rows):
    by_cohort = defaultdict(dict)
    for row in quality:
        if row["prompts"]:
            by_cohort[row["context"], row["task"]][row["policy"]] = row
    findings = []
    for (context, task), rows in by_cohort.items():
        scores = {k: r["official_score"] for k, r in rows.items()}
        dense = scores["dense_bf16_P32_p100"]
        kv = scores["dense_kv4_P32_p100"]
        oracle = {5: scores["oracle_P32_p5"], 10: scores["oracle_P32_p10"]}
        workpoints = []
        for name, centroid in rows.items():
            if centroid["method"] != "centroid8" or centroid["resident_bytes"]:
                continue
            size, percent = centroid["page_size"], centroid["percent"]
            variants = [f"{quest}_P{size}_b{percent}" for quest in SCORERS[:2]]
            if any(k not in scores or rows[k]["infeasible"] for k in (*variants, name)):
                continue
            quest = max(variants, key=scores.__getitem__)
            baseline, candidate = scores[quest], scores[name]
            viable = dense >= .90 and kv >= dense - .05 and max(oracle.values()) >= kv - .05
            discriminating = .95 <= baseline <= .98 or abs(candidate - baseline) >= .01
            workpoints.append(dict(page_size=size, percent=percent, quest=quest, candidate=name,
                                   quest_score=baseline, centroid_score=candidate,
                                   observed_improvement=candidate-baseline, oracle_viable=viable,
                                   distinguishing=discriminating,
                                   equal_byte_quality_signal=viable and discriminating and candidate-baseline >= .01))
        effects = {}
        for size in (32, 64):
            for fraction in (5, 10):
                selected = [r for r in diagnostic_rows if r["context"] == context and r["task"] == task]
                masses = {m: optional_mean(r["retained_mass"] for r in selected
                                          if r["policy"] == f"{m}_P{size}_p{fraction}")
                          for m in ("oracle", "centroid_fp32", "centroid8")}
                if all(v is not None for v in masses.values()):
                    representation = masses["oracle"] - masses["centroid_fp32"]
                    quantization = masses["centroid_fp32"] - masses["centroid8"]
                    effects[f"P{size}_p{fraction}"] = dict(mass=masses,
                                                         oracle_to_float=representation,
                                                         float_to_int8=quantization)
        findings.append(dict(context=context, task=task, dense_bf16=dense, dense_kv4=kv,
                             oracle_page_fraction=oracle, same_page_centroid_diagnosis=effects, workpoints=workpoints))
    return findings


def read_completed(root, phase):
    rows, receipts = [], {}
    for done in sorted((root / phase).glob("*/complete.json")):
        receipt = json.loads(done.read_text())
        data = done.with_name("per_prompt.json")
        if _sha(data) != receipt["per_prompt_sha256"] or _sha(done.with_name("config.json")) != receipt["config_sha256"]:
            raise ValueError(f"corrupt completed run: {done.parent}")
        cohort_rows = json.loads(data.read_text())
        if len(cohort_rows) != receipt["num_predictions"]:
            raise ValueError("prediction count mismatch")
        rows.extend(cohort_rows)
        receipts[str(done.relative_to(root))] = _sha(done)
    if not rows:
        raise ValueError("no completed Round 3 cohorts")
    return rows, receipts


def validate_frozen_formal(root, formal):
    if formal["phase"] != "test" or not formal["comparison"]:
        raise ValueError("missing formal comparison")
    for path, digest in formal["calibration_receipts"].items():
        if _sha(root / path) != digest:
            raise ValueError("formal config has stale calibration receipts")
    for name, digest in formal["input_manifests"].items():
        data = root / "data" / name
        if _sha(data / "manifest.json") != digest:
            raise ValueError("held-out manifest changed since freeze")
        manifest = json.loads((data / "manifest.json").read_text())
        if manifest["splits"]["test"]["count_per_task"] != formal["per_task_n"]:
            raise ValueError("held-out size differs from frozen sample size")
        cal_manifest = json.loads((root / "data" / formal["cohort"] / "manifest.json").read_text())
        cal_seeds = cal_manifest["splits"]["calibration"]["seeds"].values()
        if set(cal_seeds) & set(manifest["splits"]["test"]["seeds"].values()):
            raise ValueError("test seeds overlap calibration")


def formal_decision(rows, formal):
    criteria = formal["criteria"]
    baseline = formal["comparison"]["baseline"]
    candidate = formal["comparison"]["candidate"]
    data = [{**row, "method_id": row["policy"]} for row in rows if row["status"] == "complete"]
    intervals = {metric: paired_bootstrap(data, baseline, candidate, metric, method_key="method_id",
                                          confidence=criteria["two_sided_confidence"],
                                          repetitions=criteria["bootstrap_repetitions"],
                                          seed=criteria["bootstrap_seed"])
                 for metric in ("official_score", "complete_correct")}
    quality = intervals["official_score"]
    passed = quality["point_delta"] >= criteria["quality_gain"] and quality["ci_low"] > 0
    method_bytes = {}
    for method in (baseline, candidate):
        branch = [r for r in data if r["policy"] == method]
        method_bytes[method] = {label: mean(r["byte_ledger"].get("total_read_bytes", 0) / max(1, denominator(r))
                                           for r in branch)
                                for label, denominator in (("answer", lambda _: 1), ("decode_token", lambda r: r["decode_tokens"]))}
    saving = {m: 1 - method_bytes[candidate][m] / method_bytes[baseline][m] for m in ("answer", "decode_token")}
    indices = {(r["task"], r["sample_id"], r["policy"]): r["official_score"] for r in data}
    losses_by_task = defaultdict(list)
    for task, sample, method in indices:
        if method == baseline:
            losses_by_task[task].append(indices[task, sample, candidate] < indices[task, sample, baseline])
    counts = {k: dict(n=len(v), observed_regressions=sum(v)) for k, v in losses_by_task.items()}
    # A no-regression sample gives a nonzero exact one-sided bound. Require a
    # demonstrably adequate interval before taking the noninferiority route.
    zero_loss_upper = max((1 - (criteria["noninferiority_alpha"] / len(counts)) ** (1 / len(v))
                           for v in losses_by_task.values()), default=1.) if all(not any(v) for v in losses_by_task.values()) else None
    lower = quality["ci_low"] if zero_loss_upper is None else min(quality["ci_low"], -zero_loss_upper)
    savings_pass = (formal["comparison"]["route"] == "noninferiority" and
                    min(saving.values()) >= criteria["bytes_reduction"] and lower >= criteria["noninferiority_margin"])
    return dict(intervals=intervals, bytes=method_bytes, reduction=saving, regression_counts=counts,
                zero_loss_95_upper=zero_loss_upper, quality_pass=passed and formal["comparison"]["route"] == "quality",
                savings_pass=savings_pass, passed=(passed and formal["comparison"]["route"] == "quality") or savings_pass)


def report(root, *, formal=None):
    phase = "test" if formal else "calibration"
    rows, receipts = read_completed(root, phase)
    output = root / f"{phase}_report"
    output.mkdir(parents=True, exist_ok=True)
    quality = study_quality(rows)
    diagnostics = [dict(context=r["context"], task=r["task"], sample_id=r["sample_id"], **d)
                   for r in rows if r["status"] == "complete" for d in r["same_state_diagnostics"]]
    write_csv(output / "quality.csv", quality)
    write_csv(output / "same_state_per_prompt.csv", diagnostics)
    write_csv(output / "capacity.csv", capacity_table())
    write_csv(output / "byte_ledger_per_prompt.csv", [
        dict(context=r["context"], task=r["task"], sample_id=r["sample_id"], policy=r["policy"],
             decode_tokens=r["decode_tokens"], **r["byte_ledger"])
        for r in rows if r["status"] == "complete" and r["byte_ledger"]])
    _write_json(output / "per_prompt.json", rows)
    if formal:
        result = formal_decision(rows, formal)
    else:
        result = dict(findings=calibration_findings(quality, diagnostics),
                      status="calibration_only",
                      gather_max_abs_difference=max((r.get("gather_max_abs_difference") or 0 for r in rows)),
                      gather_prompt_count=sum(r.get("gather_max_abs_difference") is not None for r in rows))
    _write_json(output / "decision.json", dict(phase=phase, receipts=receipts, **result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--formal", type=Path)
    options = parser.parse_args()
    report(options.root, formal=json.loads(options.formal.read_text()) if options.formal else None)
