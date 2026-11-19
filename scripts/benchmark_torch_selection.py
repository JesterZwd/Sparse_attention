"""Measure A1 and the frozen swap on batched, normalized RULER page masses."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gqa_probe.ruler_experiment import FOCUS, _sha
from gqa_probe.torch_selection import select_torch
from scripts.report_ruler_round2 import write_csv, write_json


def benchmark(test, selection, output):
    completion = json.loads((test / "complete.json").read_text())
    config = json.loads((test / "config.json").read_text())
    if completion["status"] != "complete" or completion["run_id"] != config["run_id"]:
        raise ValueError("requires the completed held-out run")
    if config["selection_sha256"] != _sha(selection):
        raise ValueError("candidate hash differs from the frozen test configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU selector benchmark")
    candidate = json.loads(selection.read_text())["selected"]
    paths = sorted((test / "prompts").glob("*/cost_mass/*.npz"))
    if len(paths) != completion["cost_npz_files"] or not paths:
        raise ValueError("cost matrix manifest differs from saved files")
    device = torch.device("cuda")
    rng = random.Random(20261118)
    timing = []
    counts = []
    statistics = defaultdict(lambda: {"wall": [], "cuda": []})
    with torch.inference_mode():
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as payload:
                mass = torch.from_numpy(payload["mass"]).to(device)
                meta = json.loads(payload["metadata_json"].item())
            torch.cuda.synchronize(device)
            for name, (target, fraction) in FOCUS.items():
                budget = int(mass.shape[1] * fraction)

                def select(method):
                    kwargs = candidate if method == "SWAP" else {}
                    return select_torch(mass, meta["protected"], budget, target, method, **kwargs)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                identity = dict(task=meta["task"], sample_id=meta["sample_id"], layer=meta["layer"],
                                step=meta["step"], setting=name)
                for method in ("A1", "SWAP"):
                    for _ in range(5):
                        result = select(method)
                    torch.cuda.synchronize(device)
                    counts.append(dict(**identity, method=method, groups=mass.shape[0], pages=mass.shape[1],
                                       budget_pages=budget, swaps=int(result.swaps.sum().item()),
                                       gate_hits=int(result.gate_hit.sum().item()),
                                       logical_pairs=int(result.candidate_pairs.sum().item()),
                                       executed_pairs=int(result.metrics["executed_candidate_pairs"].sum().item())))
                for repetition in range(10):
                    methods = ["A1", "SWAP"]
                    rng.shuffle(methods)
                    for method in methods:
                        started = time.perf_counter_ns()
                        start_event.record()
                        select(method)
                        end_event.record()
                        end_event.synchronize()
                        wall_us = (time.perf_counter_ns() - started) / 1000
                        elapsed_us = start_event.elapsed_time(end_event) * 1000
                        stats = statistics[name, method]
                        stats["wall"].append(wall_us)
                        stats["cuda"].append(elapsed_us)
                        timing.append(dict(**identity, method=method, repetition=repetition,
                                           wall_us=wall_us, cuda_event_us=elapsed_us))
            if (index + 1) % 32 == 0:
                print(f"GPU selector benchmark {index + 1}/{len(paths)}", flush=True)
    summary = []
    for (setting, method), times in sorted(statistics.items()):
        reference = statistics[setting, "A1"]
        summary.append(dict(setting=setting, method=method, calls=len(times["wall"]),
                            median_wall_us=float(np.median(times["wall"])),
                            p95_wall_us=float(np.quantile(times["wall"], .95)),
                            median_cuda_event_us=float(np.median(times["cuda"])),
                            p95_cuda_event_us=float(np.quantile(times["cuda"], .95)),
                            median_wall_ratio_to_a1=float(np.median(times["wall"]) / np.median(reference["wall"])),
                            median_cuda_ratio_to_a1=float(np.median(times["cuda"]) / np.median(reference["cuda"]))))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "timings.csv", timing)
    write_csv(output / "logical_and_executed_counts.csv", counts)
    write_json(output / "summary.json", dict(
        source_test_run=config["run_id"], selection_sha256=_sha(selection), benchmark_files=len(paths),
        gpu=torch.cuda.get_device_name(device), torch=torch.__version__, groups_per_call=8,
        warmup=5, repetitions=10, summaries=summary,
        scope="Torch selector from device-resident normalized mass to returned decision and coverage. "
              "Validation scans and selector counters are included. Timers exclude input transfer, "
              "mass scoring, output metric transfers and full-model work. CUDA event time can include "
              "host launch gaps and synchronizations. Each call processes eight KV groups."))
    print(f"GPU selector benchmark complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("results/ruler_round2_test"))
    parser.add_argument("--selection", type=Path, default=Path("results/ruler_round2_calibration/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("results/round2_torch_selector_benchmark"))
    args = parser.parse_args()
    benchmark(args.test, args.selection, args.output)


if __name__ == "__main__":
    main()
