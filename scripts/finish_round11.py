"""Freeze the 64K hardware screen and classify the equal-resource work points."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

import numpy as np
import torch

from gqa_probe.round11_energy import DESIGN_AXES, PRIMARY
from gqa_probe.round11_workpoints import A, BASELINE
from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.round11_energy import run as energy_run
from scripts.run_round11 import ROOT, admitted, freeze, model_info, read_json, run_phase


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def primary(row):
    return tuple(type(value)(row[field]) for field, value in zip(DESIGN_AXES, PRIMARY, strict=True)) == PRIMARY


def timing_screen(path):
    by = defaultdict(list)
    for row in records(path):
        if primary(row) and row["scope"] == "regular":
            by[row["policy"]].append(row)
    if "A" not in by:
        raise ValueError("equal-resource timing reference absent")
    all_keys = {(r["sample_id"], r["step"]) for r in by["A"]}
    results = {}
    for policy, rows in by.items():
        if {(r["sample_id"], r["step"]) for r in rows} != all_keys or len(rows) != len(all_keys):
            raise ValueError("timing screen requires matched full-step states")
        elapsed = np.array([float(r["completion_ns"]) for r in rows])
        if not np.isfinite(elapsed).all() or (elapsed <= 0).any():
            raise ValueError("invalid replay duration")
        logits = [float(r["peak_logits_sram_bytes"]) for r in rows if r.get("peak_logits_sram_bytes")]
        results[policy] = dict(steps=len(rows), mean_ns=float(elapsed.mean()), p95_ns=float(np.quantile(elapsed, .95)),
                               peak_sram_bytes=max(int(float(r["peak_sram_bytes"])) for r in rows),
                               peak_logits_sram_bytes=max(logits, default=None),
                               fits=all(r["feasible_sram"].lower() in ("1", "true") for r in rows))
    a = results["A"]
    for p in results.values():
        p.update(mean_ratio=p["mean_ns"] / a["mean_ns"], p95_ratio=p["p95_ns"] / a["p95_ns"])
    return results


def freeze64(root):
    output = root / "64k"
    entry_path = output / "entry.json"
    entry = read_json(entry_path)
    if not entry["eligible"]:
        return
    timing = output / "replay" / "step_timing.csv"
    stats = timing_screen(timing)
    selected = [policy for policy in entry["eligible"] if entry["calibrated"][policy] and stats[policy]["fits"]]
    result = dict(selected=selected, qualification=entry["calibrated"], full_step_timing=stats,
                  timing_sha256=_sha(timing), entry_sha256=_sha(entry_path),
                  resources=dict(zip(DESIGN_AXES, PRIMARY, strict=True)))
    _write_json(output / "hardware_screen.json", result)
    if selected:
        freeze(root, "64k", (A, *(BASELINE[p] for p in selected)), entry["model_sha256"], hardware=result)


def classify(root, cohort):
    stage = root / cohort
    independent = stage / "independent"
    results = independent if (independent / "complete.json").exists() else stage / "calibration"
    if not (results / "complete.json").exists():
        return
    rows = read_json(results / "predictions.json")
    policies = sorted({r["policy"] for r in rows})
    timing_path = stage / "replay" / "step_timing.csv"
    energy_run(results / "predictions.json", timing_path, results, cohort, policies=policies)
    sw = read_json(stage / "decision.json")["candidates"] if results == independent else {}
    timing = timing_screen(timing_path)
    energy = defaultdict(dict)
    for r in records(results / "energy.csv"):
        if primary(r):
            energy[r["policy"]][r["scenario"]] = float(r["energy_uj_per_decode_step"])
    checks = {}
    for policy in policies:
        if policy == "A":
            continue
        ratios = {scenario: e / energy["A"][scenario] for scenario, e in energy[policy].items()}
        if len(ratios) != 3 or not all(math.isfinite(x) for x in ratios.values()):
            raise ValueError("missing energy sensitivity")
        t = timing[policy]
        latency_and_capacity = t["fits"] and t["mean_ratio"] <= 1. and t["p95_ratio"] <= 1.
        sensitivity_support = all(x <= 1. for x in ratios.values())
        checks[policy] = dict(algorithm_gate=sw.get(policy), stage=results.name,
                              co_design=bool(sw.get(policy, {}).get("passes") and latency_and_capacity
                                             and sensitivity_support),
                              latency_and_capacity=latency_and_capacity,
                              energy_ratios=ratios, energy_scenarios_support=sensitivity_support, **t)
    _write_json(stage / "hardware_decision.json", dict(comparisons=checks, timing=timing, primary_resource=PRIMARY,
                                                      timing_sha256=_sha(timing_path),
                                                      interpretation="Modeled Attention decoder only; paired complete steps, calibration traces."))
    print(cohort, [(p, v["co_design"], v["mean_ratio"], v["p95_ratio"]) for p, v in checks.items()])


@torch.inference_mode()
def follow_up_low_signal_vt(root):
    cross = root / "cross_task"
    vt = cross / "32k_vt"
    summary = records(vt / "quality_cost.csv")
    paired = read_json(vt / "paired.json")
    no_disagreement = all(delta["deltas"]["official_score"] == 0
                          for pair in paired for delta in pair["prompt_deltas"])
    fwe = cross / "16k_fwe"
    if no_disagreement and min(float(r["official_score"]) for r in summary) >= .95:
        if not (fwe / "complete.json").is_file():
            opts = (A, *(BASELINE[name] for name in (admitted(root) or ["B"])))
            _, result = run_phase(root / "data" / "16k_fwe", fwe, opts, model_info(root, "16k_fwe"))
        else:
            result = records(fwe / "quality_cost.csv")
        _write_json(cross / "vt_low_signal_followup.json",
                    dict(vt_all_prompt_scores_identical=True, vt_score_floor=.95,
                         vt_pairs_sha256=_sha(vt / "paired.json"), fwe=result,
                         fwe_receipt_sha256=_sha(fwe / "complete.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze64", "audit32", "audit64", "fwe"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze64":
        freeze64(args.root)
    elif args.action == "fwe":
        follow_up_low_signal_vt(args.root)
    else:
        classify(args.root, args.action.removeprefix("audit") + "k")


if __name__ == "__main__":
    main()
