"""Round 2 fixed-state replay and separately instrumented CPU selector costs."""

import argparse
import csv
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .attention import attention_output, page_mass
from .online_selection import select_online
from .paired_statistics import summarize_shadow
from .selection import protected_pages


def variant_specs():
    return [
        dict(name="a1", algorithm="A1"),
        dict(name="a3", algorithm="A3"),
        dict(name="a3_b8", algorithm="A3-B", batch_size=8),
        dict(name="gate_a3", algorithm="GATE-A3"),
        dict(name="swap_d01_s1", algorithm="SWAP", delta=.01, max_swaps=1),
        dict(name="swap_d00_s1", algorithm="SWAP", delta=0., max_swaps=1),
        dict(name="swap_d02_s1", algorithm="SWAP", delta=.02, max_swaps=1),
        dict(name="swap_d01_s2", algorithm="SWAP", delta=.01, max_swaps=2),
        dict(name="a3_u100", algorithm="A3-U", alpha=1.),
        dict(name="a3_u095", algorithm="A3-U", alpha=.95),
        dict(name="a3_u090", algorithm="A3-U", alpha=.90),
    ]


def _kwargs(spec):
    return {k: v for k, v in spec.items() if k != "name"}


def _gold_recall(pages, spans, length, page_size):
    if not spans:
        return dict(gold_span_recall=None, gold_token_recall=None, gold_page_recall=None)
    selected = set(pages)
    tokens = set(i for start, end in spans for i in range(max(0, start), min(length, end)))
    gold_pages = {i // page_size for i in tokens}
    covered = sum(i // page_size in selected for i in tokens)
    complete = sum(all(i // page_size in selected for i in range(start, end)) for start, end in spans)
    return dict(gold_span_recall=complete / len(spans), gold_token_recall=covered / len(tokens),
                gold_page_recall=len(selected & gold_pages) / len(gold_pages))


def evaluate_replay_trace(logits, values, protected, budget, target, budget_fraction,
                          specs, page_size=32, gold_spans=None, mass=None, dense=None):
    mass = page_mass(logits, page_size) if mass is None else mass
    dense = attention_output(logits, values) if dense is None else dense
    norm = np.linalg.norm(dense, axis=1)
    outputs = {}
    rows = []
    for spec in specs:
        started = time.perf_counter_ns()
        selected = select_online(mass, protected, budget, target, validate=False,
                                 count_comparisons=False, record_audit=False, **_kwargs(spec))
        elapsed = (time.perf_counter_ns() - started) * 1e-9
        pages = tuple(sorted(selected.order))
        if pages not in outputs:
            outputs[pages] = attention_output(logits, values, pages, page_size)
        absolute = np.linalg.norm(outputs[pages] - dense, axis=1)
        relative = absolute / (norm + 1e-12)
        original_deficit = float(np.maximum(target - selected.coverage, 0.).sum())
        rows.append(dict(spec=spec["name"], algorithm=spec["algorithm"],
                         config_id=f"p{target:.2f}_b{budget_fraction:.2f}", target=target,
                         budget_fraction=budget_fraction, budget_pages=budget,
                         available_pages=len(mass), valid_tokens=logits.shape[1],
                         protected_count=len(set(protected)), pages=list(pages),
                         page_order=selected.order, head_coverage=selected.coverage.tolist(),
                         head_output_absolute_errors=absolute.tolist(),
                         head_output_relative_errors=relative.tolist(),
                         head_dense_norm=norm.tolist(), remaining_deficit_sum=original_deficit,
                         original_target_deficit=original_deficit,
                         avg_retained_mass=float(selected.coverage.mean()),
                         gate_hit=bool(selected.metrics["gate_hit"]),
                         swaps=selected.metrics["swaps"],
                         candidate_pairs=selected.metrics["candidate_pairs"],
                         executed_candidate_pairs=selected.metrics["candidate_pairs"],
                         online_cpu_seconds=elapsed, selector_work=selected.metrics,
                         **_gold_recall(pages, gold_spans, logits.shape[1], page_size)))
    return rows


def _shadow_row(row):
    names = ["sample_id", "task", "split", "spec", "config_id", "target", "budget_fraction",
             "layer", "step", "kv_group", "head_coverage", "head_output_absolute_errors",
             "head_output_relative_errors", "pages", "original_target_deficit", "gate_hit",
             "swaps", "candidate_pairs", "executed_candidate_pairs", "gold_span_recall",
             "gold_token_recall", "gold_page_recall"]
    return {k: row[k] for k in names if k in row}


def _summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["spec"], row["target"], row["budget_fraction"]].append(row)
    output = []
    for key, records in sorted(groups.items()):
        absolute = np.concatenate([r["head_output_absolute_errors"] for r in records])
        relative = np.concatenate([r["head_output_relative_errors"] for r in records])
        coverage = np.concatenate([r["head_coverage"] for r in records])
        item = dict(spec=key[0], target=key[1], budget_fraction=key[2], records=len(records),
                    independent_prompts=len({r["sample_id"] for r in records}),
                    absolute_error_p95=float(np.quantile(absolute, .95)),
                    absolute_error_p99=float(np.quantile(absolute, .99)),
                    relative_error_p95=float(np.quantile(relative, .95)),
                    relative_error_p99=float(np.quantile(relative, .99)),
                    coverage_p05=float(np.quantile(coverage, .05)),
                    avg_retained_mass=float(coverage.mean()),
                    mean_deficit=float(np.mean([r["original_target_deficit"] for r in records])),
                    mean_pages=float(np.mean([len(r["pages"]) for r in records])),
                    gate_hit_rate=float(np.mean([r["gate_hit"] for r in records])),
                    mean_swaps=float(np.mean([r["swaps"] for r in records])),
                    mean_candidate_pairs=float(np.mean([r["candidate_pairs"] for r in records])))
        for metric in ["gold_span_recall", "gold_token_recall", "gold_page_recall"]:
            valid = [r[metric] for r in records if r[metric] is not None]
            item[metric] = float(np.mean(valid)) if valid else None
        output.append(item)
    return output


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_cases(cases, specs, output, repetitions=5, seed=20261118):
    """Mass load/normalization and expensive comparator audits are outside timings."""
    rng = random.Random(seed)
    timings = []
    costs = []
    for i, case in enumerate(cases):
        mass, protected, budget, target, fraction, meta = case
        for spec in specs:
            select_online(mass, protected, budget, target, validate=False, **_kwargs(spec))
            counted = select_online(mass, protected, budget, target, validate=False,
                                    count_comparisons=True, **_kwargs(spec))
            costs.append(dict(**meta, target=target, budget_fraction=fraction,
                              spec=spec["name"], work=counted.metrics))
        for repeat in range(repetitions):
            order = list(specs)
            rng.shuffle(order)
            for spec in order:
                start = time.perf_counter_ns()
                result = select_online(mass, protected, budget, target, validate=False,
                                       count_comparisons=False, record_audit=False, **_kwargs(spec))
                elapsed = time.perf_counter_ns() - start
                timings.append(dict(**meta, target=target, budget_fraction=fraction,
                                    spec=spec["name"], repeat=repeat, nanoseconds=elapsed))
                assert len(result.order) == budget
        if (i + 1) % 32 == 0:
            print(f"benchmark {i + 1}/{len(cases)}", flush=True)
    grouped = defaultdict(list)
    for row in timings:
        grouped[row["spec"], row["target"], row["budget_fraction"]].append(row["nanoseconds"] * 1e-9)
    summary = [dict(spec=k[0], target=k[1], budget_fraction=k[2], calls=len(v),
                    median_seconds=float(np.median(v)), p95_seconds=float(np.quantile(v, .95)))
               for k, v in sorted(grouped.items())]
    a1 = {(r["target"], r["budget_fraction"]): r for r in summary if r["spec"] == "a1"}
    for row in summary:
        reference = a1[row["target"], row["budget_fraction"]]
        row["median_ratio_to_a1"] = row["median_seconds"] / reference["median_seconds"]
        row["p95_ratio_to_a1"] = row["p95_seconds"] / reference["p95_seconds"]
    _write_csv(output / "timings.csv", timings)
    _write_csv(output / "timing_summary.csv", summary)
    with gzip.open(output / "cost_records.jsonl.gz", "wt") as f:
        for row in costs:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    payload = dict(timing_scope="trusted normalized mass through returned decision; lightweight online counters included; no comparator/clip/change/first-satisfied audits, mass scoring or I/O",
                   comparison_scope="actual cmp_to_key TimSort calls in a separate counted run, not ASIC cycles",
                   warmup_calls_per_case_method=1, repetitions=repetitions, seed=seed,
                   cases=len(cases), summary=summary)
    (output / "timing_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/pilot_16k/traces"))
    parser.add_argument("--output", type=Path, default=Path("results/round2_replay"))
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    specs = variant_specs()
    manifest = json.loads((args.source / "manifest.json").read_text())
    config = dict(source=str(args.source.resolve()), methods=specs, targets=[.7, .8, .9],
                  fractions=[.05, .10, .20], page_size=32, sink_pages=1, local_tokens=128,
                  focus=[[.8, .2], [.9, .1]], new_quality_evidence=False,
                  independent_prompts=2, scope="replay on old exploratory data only")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B-Instruct-2507", local_files_only=True)
    gold = {}
    for sample in manifest["samples"]:
        text = f"\nKEY {sample['target_key']} VALUE {sample['expected_answer']}\n"
        count = len(tokenizer.encode(text, add_special_tokens=False))
        gold[sample["sample_id"]] = [[sample["needle_token_offset"], sample["needle_token_offset"] + count]]
    all_rows, cases = [], []
    with gzip.open(args.output / "records.jsonl.gz", "wt") as stream:
        for i, meta in enumerate(manifest["records"]):
            with np.load(args.source / meta["file"], allow_pickle=False) as trace:
                logits, values = trace["logits"], trace["values"]
            mass, dense = page_mass(logits), attention_output(logits, values)
            protection = protected_pages(logits.shape[1])
            info = {k: meta[k] for k in ["sample_id", "layer", "step", "kv_group"]}
            for fraction in config["fractions"]:
                budget = int(np.floor(len(mass) * fraction))
                for target in config["targets"]:
                    rows = evaluate_replay_trace(logits, values, protection, budget, target,
                                                 fraction, specs, gold_spans=gold[meta["sample_id"]],
                                                 mass=mass, dense=dense)
                    baseline = rows[0]
                    for row in rows:
                        row.update(info, task="pilot_custom", split="exploration")
                        assert len(row["pages"]) == budget
                        assert set(protection).issubset(row["pages"])
                        assert row["avg_retained_mass"] <= baseline["avg_retained_mass"] + 1e-12
                        if row["spec"] == "a3_u100":
                            reference = next(r for r in rows if r["spec"] == "a3")
                            assert row["page_order"] == reference["page_order"]
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    all_rows.extend(rows)
                    if (target, fraction) in {(.8, .2), (.9, .1)}:
                        cases.append((mass, protection, budget, target, fraction, info))
            if (i + 1) % 16 == 0:
                print(f"replayed {i + 1}/{len(manifest['records'])}", flush=True)
    summary = _summaries(all_rows)
    prompts = summarize_shadow([_shadow_row(row) for row in all_rows])
    (args.output / "summary.json").write_text(json.dumps(dict(scope=config["scope"], summary=summary), indent=2, allow_nan=False))
    _write_csv(args.output / "summary.csv", summary)
    (args.output / "per_prompt.json").write_text(json.dumps(prompts, indent=2, allow_nan=False))
    bench_specs = [s for s in specs if s["algorithm"] != "A3-U"]
    benchmark_cases(cases, bench_specs, args.output, repetitions=args.benchmark_repetitions)
    (args.output / "complete.json").write_text(json.dumps(dict(status="complete", traces=len(manifest["records"]),
                                                              result_rows=len(all_rows), methods=len(specs))))
    print(f"Round 2 replay complete: {len(all_rows)} results", flush=True)


if __name__ == "__main__":
    main()
