"""Offline, paired evaluation of full-pool selectors on identical traces."""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .attention import attention_output, page_mass
from .selection import protected_pages, select_pages


def evaluate_trace(logits, values, page_size=32, budgets=(.05, .1, .2),
                   targets=(.7, .8, .9), batches=(1, 4, 8, 16), sink_pages=1,
                   local_tokens=128, mode="fixed"):
    mass = page_mass(logits, page_size)
    protected = protected_pages(logits.shape[1], page_size, sink_pages, local_tokens)
    dense = attention_output(logits, values)
    rows = []
    methods = [("A1", 1), ("A3", 1)] + [("A3-B", b) for b in batches]
    for fraction in budgets:
        budget = int(np.floor(mass.shape[0] * fraction))
        for target in targets:
            for algorithm, batch_size in methods:
                started = time.perf_counter()
                selected = select_pages(mass, protected, budget, target, algorithm,
                                        batch_size=batch_size, mode=mode)
                elapsed = time.perf_counter() - started
                output = attention_output(logits, values, selected.order, page_size)
                absolute = np.linalg.norm(output - dense, axis=1)
                relative = absolute / (np.linalg.norm(dense, axis=1) + 1e-12)
                coverage = selected.coverage
                # Single-head necessary upper bound, not joint feasibility.
                mask = np.ones(mass.shape[0], dtype=bool)
                mask[protected] = False
                free = budget - len(protected)
                best = np.sort(mass[mask], axis=0)[::-1][:free].sum(axis=0)
                upper = mass[protected].sum(axis=0) + best
                row = dict(algorithm=algorithm, batch_size=batch_size, mode=mode,
                           budget_fraction=fraction, target=target, budget_pages=budget,
                           available_pages=mass.shape[0], protected_pages=len(protected),
                           selected_pages=len(selected.order),
                           valid_tokens=logits.shape[1], heads=mass.shape[1],
                           avg_retained_mass=float(coverage.mean()),
                           min_head_mass=float(coverage.min()),
                           unsatisfied_head_ratio=float(np.mean(target - coverage > 1e-12)),
                           remaining_deficit_sum=float(np.maximum(target - coverage, 0).sum()),
                           individually_infeasible_head_ratio=float(np.mean(upper < target - 1e-12)),
                           output_relative_error_mean=float(relative.mean()),
                           output_absolute_error_mean=float(absolute.mean()),
                           head_coverage=coverage.tolist(),
                           head_output_relative_errors=relative.tolist(),
                           head_output_absolute_errors=absolute.tolist(),
                           page_order=selected.order, batches=selected.batches,
                           selector_python_seconds=elapsed,
                           **selected.metrics)
                rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["batch_size"], row["budget_fraction"], row["target"])
        groups[key].append(row)
    summary = []
    for key, records in sorted(groups.items()):
        result = dict(zip(("algorithm", "batch_size", "budget_fraction", "target"), key))
        result["records"] = len(records)
        for name in records[0]:
            vals = [r[name] for r in records if isinstance(r.get(name), (int, float))]
            if vals and name not in result:
                result[name] = float(np.mean(vals))
        coverage = np.concatenate([r["head_coverage"] for r in records])
        rel = np.concatenate([r["head_output_relative_errors"] for r in records])
        absolute = np.concatenate([r["head_output_absolute_errors"] for r in records])
        result["p05_query_head_mass"] = float(np.quantile(coverage, .05))
        batch_changes = [b["batch_selection_change_rate"] for r in records
                         for b in r.get("batches", [])]
        result["batch_selection_change_rate_mean"] = float(np.mean(batch_changes)) if batch_changes else None
        for label, errors in [("relative", rel), ("absolute", absolute)]:
            for q in [.95, .99]:
                result[f"output_{label}_error_p{int(q * 100)}"] = float(np.quantile(errors, q))
        for numerator, denominator, name in [
            ("clip_contributions", "evaluated_contributions", "clip_rate"),
            ("changed_decisions", "selector_iterations", "decision_change_rate"),
            ("batch_changed_pages", "coverage_selection_pages", "batch_selection_change_rate"),
        ]:
            den = sum(r.get(denominator, 0) for r in records)
            result[name] = sum(r.get(numerator, 0) for r in records) / den if den else None
        summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed")
    args = parser.parse_args()
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    with (args.output / "records.jsonl").open("w") as stream:
        for index, record in enumerate(manifest["records"]):
            with np.load(args.trace_dir / record["file"], allow_pickle=False) as trace:
                rows = evaluate_trace(trace["logits"], trace["values"],
                                      page_size=args.page_size, mode=args.mode)
            for row in rows:
                row.update(sample_id=record["sample_id"], layer=record["layer"],
                           step=record["step"], kv_group=record["kv_group"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            all_rows.extend(rows)
            print(f"evaluated {index + 1}/{len(manifest['records'])}", flush=True)
    summary = summarize(all_rows)
    payload = {"source_manifest": str((args.trace_dir / "manifest.json").resolve()),
               "source_kind": manifest.get("source_kind"), "mode": args.mode,
               "gate1_status": "not_adjudicated", "summary": summary}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    if summary:
        fields = sorted(set().union(*(r.keys() for r in summary)))
        with (args.output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    print(f"wrote {len(all_rows)} paired results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
