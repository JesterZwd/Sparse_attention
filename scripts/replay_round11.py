"""Replay matched Round 11 group captures and complete serial decode steps on CPU."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from hashlib import file_digest
from io import TextIOWrapper
from itertools import combinations, product
import json
from math import ceil
import os
from pathlib import Path
from statistics import fmean
import tempfile

from gqa_probe.round11_timing import (
    BASE_SCORER, BANK_ROW_BYTES, BURST_BYTES, CK_NS, FAST_SCORER, GROUP_REGION_BYTES,
    LANES, MODES, PRIMARY_PE_NS, QUEUE_DEPTH, ROUND9_SCORER_REFERENCE_NS, SRAM_BYTES,
    BankSession, group_address_offset, prepare_group, replay_group,
)

IDENTITY = ("cohort", "sample_id", "step", "layer", "group")
SCENARIO = ("lanes", "pe_period_ns", "scorer_variant", "mode", "coefficient_ii",
            "coefficient_latency_pe_cycles", "sram_capacity_bytes")
TIME_AND_ACTIVITY = (
    "completion_ns", "first_v_command_ns", "first_v_read_ns", "post_scorer_ns", "post_scorer_compute_ns",
    "dependency_lower_bound_ns", "scorer_wait_ns", "qk_wait_ns", "pv_wait_ns",
    "scorer_busy_ns", "qk_busy_ns", "pv_busy_ns", "k_unpack_ns", "v_unpack_ns",
    "dram_command_util", "dram_data_util", "dram_outstanding_util", "backend_active_util",
    "array_util", "scorer_active_util", "summary_bytes", "k_bytes", "v_bytes", "total_bytes",
    "rd", "act", "pre", "descriptor_runs", "peak_sram_bytes", "peak_live_sram_bytes", "peak_logits_bytes",
    "peak_inflight_bursts", "peak_pending_descriptors", "peak_summary_slots", "peak_k_slots",
    "peak_v_slots", "peak_packed_tile_bytes", "peak_unpacked_tile_bytes",
    "score_data_overlap_ns", "qk_data_overlap_ns", "pv_data_overlap_ns", "data_compute_overlap_ns",
    "request_window_wait_ns", "summary_slot_wait_ns", "k_slot_wait_ns", "v_slot_wait_ns",
    "sram_port_contention_wait_ns", "peak_pv_weight_sram_bytes", "peak_scorer_score_sram_bytes",
)
ADDITIVE = (
    "completion_ns", "post_scorer_ns", "post_scorer_compute_ns", "dependency_lower_bound_ns", "scorer_wait_ns",
    "qk_wait_ns", "pv_wait_ns", "scorer_busy_ns", "qk_busy_ns", "pv_busy_ns", "pv_weights_ns",
    "k_unpack_ns", "v_unpack_ns", "dram_outstanding_ns", "coefficient_busy_ns", "command_ck",
    "rd", "act", "pre", "row_hits", "summary_rd", "k_rd", "v_rd", "descriptor_runs",
    "summary_bytes", "k_bytes", "v_bytes", "total_bytes", "summary_payload_bytes",
    "summary_alignment_bytes", "k_code_bytes", "k_scale_bytes", "k_zero_bytes",
    "k_zero_point_bytes", "v_code_bytes", "v_scale_bytes", "v_zero_bytes", "v_zero_point_bytes",
    "k_records", "v_records", "scorer_terms", "fused_updates", "qk_mac", "pv_mac",
    "histograms", "histogram_updates", "histogram_bin_visits", "gate_exponentials",
    "gate_comparisons", "gate_reduce_adds", "mask_sram_reads", "mask_sram_writes",
    "logits_sram_reads", "logits_sram_writes", "packed_sram_read_bytes", "packed_sram_write_bytes",
    "unpack_sram_read_bytes", "unpack_sram_write_bytes", "k_unpack_elements", "v_unpack_elements",
    "coefficient_records", "coefficient_scale_values", "pv_weight_exponentials",
    "pv_weight_sram_read_bytes", "pv_weight_sram_write_bytes",
    "score_data_overlap_ns", "qk_data_overlap_ns", "pv_data_overlap_ns", "data_compute_overlap_ns",
    "request_window_wait_ns", "summary_slot_wait_ns", "k_slot_wait_ns", "v_slot_wait_ns",
    "sram_port_contention_wait_ns",
)
PEAKS = (
    "peak_sram_bytes", "peak_live_sram_bytes", "peak_summary_sram_bytes", "peak_k_sram_bytes", "peak_gate_sram_bytes",
    "peak_v_sram_bytes", "peak_logits_bytes", "peak_logits_sram_bytes", "peak_inflight_bursts", "peak_pending_descriptors",
    "peak_summary_slots", "peak_k_slots", "peak_v_slots", "peak_packed_tile_bytes",
    "peak_unpacked_tile_bytes", "peak_pv_accumulator_sram_bytes", "histogram_sram_bytes",
    "gate_control_sram_bytes", "peak_pv_weight_sram_bytes", "peak_scorer_score_sram_bytes",
)
CANDIDATE_MODES = ("serial_one_slot", "summary_stream", "double_buffer")


def digest(path):
    with open(path, "rb") as source:
        return file_digest(source, "sha256").hexdigest()


def load_states(paths, policies, layers, groups):
    isolated, steps, seen = defaultdict(list), defaultdict(list), defaultdict(set)
    observations = Counter()
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            for line in source:
                row = json.loads(line)
                policy = row["policy"]
                if policies and policy not in policies:
                    continue
                identity = tuple(row[key] for key in IDENTITY)
                if policy in seen[identity]:
                    raise ValueError(f"duplicate capture of {identity} for {policy}")
                seen[identity].add(policy)
                observations[row["cohort"], policy, row["scope"], row["whole_step"]] += 1
                if row["whole_step"]:
                    steps[(*identity[:3], policy)].append(row)
                else:
                    isolated[identity[:2]].append(row)
    if not seen:
        raise ValueError("no eligible group captures")
    expected_cells = set(product(range(layers), range(groups)))
    for key, rows in steps.items():
        actual = {(row["layer"], row["group"]) for row in rows}
        if actual != expected_cells:
            raise ValueError(f"incomplete whole step {key}: observed {len(actual)} of {len(expected_cells)} cells")
    # The selected whole-step endpoint must be complete for every policy observed
    # at any layer/group in that endpoint, including simultaneous policy fallbacks.
    step_policies = defaultdict(set)
    for cohort, sample, step, policy in steps:
        step_policies[cohort, sample, step].add(policy)
    for identity, present in seen.items():
        complete = step_policies.get(identity[:3])
        if complete and present != complete:
            raise ValueError(f"whole-step policy pairing is incomplete at {identity}")
    return isolated, steps, observations


def scenario_rows(prepared, config):
    for row in prepared:
        for settings in scenarios(row, config):
            yield replay_group(row, **settings)


def scenarios(row, config):
    variant = (BASE_SCORER,)
    if row.raw["scorer_channels"] == 96 and config["ii6"]:
        variant += (FAST_SCORER,)
    for lanes, period, scorer, mode in product(config["lanes"], config["periods"], variant, config["modes"]):
        yield dict(lanes=lanes, pe_period_ns=period, scorer_variant=scorer, mode=mode,
                   coefficient_ii=config["coefficient_ii"], coefficient_latency=config["coefficient_latency"],
                   sram_capacity_bytes=config["sram_capacity_bytes"])


def join_step(group_rows, layers, groups):
    first = group_rows[0]
    row = {key: first[key] for key in ("cohort", "sample_id", "step", "policy", *SCENARIO)}
    for key in ADDITIVE:
        row[key] = sum(item[key] for item in group_rows)
    for key in PEAKS:
        row[key] = max(item[key] for item in group_rows)
    total = row["completion_ns"]
    # Relative first V comes from the first hardware group using V.
    first_v = next((item for item in group_rows if item["first_v_command_ns"] is not None), None)
    row.update(
        scope="regular", whole_step=True, groups=len(group_rows), layers=layers,
        kv_groups_per_layer=groups, fallback_groups=sum(item["fallback"] for item in group_rows),
        phased_sram_counterfactual=first["mode"] == "serial",
        feasible_sram=all(item["feasible_sram"] for item in group_rows),
        first_v_command_ns=first_v["session_start_ns"] + first_v["first_v_command_ns"] if first_v else None,
        first_v_read_ns=first_v["session_start_ns"] + first_v["first_v_read_ns"] if first_v else None,
        per_group_mean_first_v_command_ns=fmean(item["first_v_command_ns"] for item in group_rows
                                               if item["first_v_command_ns"] is not None) if first_v else None,
        last_group_end_ns=group_rows[-1]["session_start_ns"] + group_rows[-1]["completion_ns"],
        dram_command_util=row["command_ck"] * CK_NS / total,
        dram_data_util=2 * row["rd"] * CK_NS / total,
        dram_outstanding_util=row["dram_outstanding_ns"] / total,
        backend_active_util=(row["qk_busy_ns"] + row["pv_busy_ns"]) / total,
        array_util=(row["qk_mac"] + row["pv_mac"]) * row["pe_period_ns"] / (row["lanes"] * total),
        scorer_active_util=row["scorer_busy_ns"] / total,
    )
    if abs(row["last_group_end_ns"] - total) > max(1e-5, total * 1e-12):
        raise ValueError("whole-step critical sequence did not remain on one timebase")
    return row


def replay_step(prepared, config):
    for settings in scenarios(prepared[0], config):
        with BankSession() as session:
            yield [
                replay_group(row, session=session, address_offset=group_address_offset(ordinal), **settings)
                for ordinal, row in enumerate(prepared)
            ]


def summarize(by):
    for keys, values in sorted(by.items()):
        cohort, scope, policy, *scenario = keys
        row = dict(cohort=cohort, scope=scope, policy=policy, **dict(zip(SCENARIO, scenario, strict=True)))
        row.update(states=len(values), samples=len({x["sample_id"] for x in values}),
                   feasible_states=sum(x["feasible_sram"] for x in values),
                   all_feasible_sram=all(x["feasible_sram"] for x in values))
        for metric in TIME_AND_ACTIVITY:
            entries = sorted(x[metric] for x in values if x.get(metric) is not None)
            if entries:
                row.update({f"{metric}_mean": fmean(entries),
                            f"{metric}_p95": entries[ceil(.95 * len(entries)) - 1],
                            f"{metric}_max": entries[-1]})
        yield row


def paired_summaries(pairs):
    by = defaultdict(list)
    for (cohort, _, _, *tail), states in pairs.items():
        left_scope, lane, period, variant, mode, coeff, latency, sram = tail[-8:]
        for (name_a, a), (name_b, b) in combinations(sorted(states.items()), 2):
            if name_a != "A" and (name_a, name_b) != ("B", "C"):
                continue
            by[cohort, left_scope, name_a, name_b, lane, period, variant, mode, coeff, latency, sram].append((a, b))
    for key, values in sorted(by.items()):
        cohort, scope, ref, candidate, *scenario = key
        row = dict(cohort=cohort, scope=scope, reference=ref, candidate=candidate,
                   pairs=len(values), **dict(zip(SCENARIO, scenario, strict=True)))
        for metric in ("completion_ns", "first_v_command_ns", "post_scorer_ns", "post_scorer_compute_ns",
                       "peak_sram_bytes", "peak_logits_bytes"):
            ds = sorted(b[metric] - a[metric] for a, b in values
                        if a[metric] is not None and b[metric] is not None)
            if ds:
                row[f"{metric}_delta_mean"] = fmean(ds)
                row[f"{metric}_delta_p95"] = ds[ceil(.95 * len(ds)) - 1]
                row[f"{metric}_reference_mean"] = fmean(a[metric] for a, b in values if
                                                       a[metric] is not None and b[metric] is not None)
                row[f"{metric}_candidate_mean"] = fmean(b[metric] for a, b in values if
                                                       a[metric] is not None and b[metric] is not None)
        yield row


def paired_modes(pairs):
    # The existing pair keys already identify matched captures across modes.
    across_modes = defaultdict(dict)
    for key, by_policy in pairs.items():
        identity, settings = key[:-len(SCENARIO)], key[-len(SCENARIO):]
        lane, period, variant, mode, coeff, latency, sram = settings
        for policy, row in by_policy.items():
            across_modes[identity, policy, lane, period, variant, coeff, latency, sram][mode] = row
    by = defaultdict(list)
    for (identity, policy, lane, period, variant, coeff, latency, sram), modes in across_modes.items():
        for ref, candidate in combinations(MODES, 2):
            if ref in modes and candidate in modes:
                by[identity[0], identity[-1], policy, lane, period, variant, ref, candidate, coeff,
                   latency, sram].append((modes[ref], modes[candidate]))
    for key, rows in sorted(by.items()):
        cohort, scope, policy, lane, period, variant, ref, candidate, coeff, latency, sram = key
        a, b = (fmean(row[name]["completion_ns"] for row in rows) for name in (0, 1))
        ds = sorted(candidate["completion_ns"] - reference["completion_ns"] for reference, candidate in rows)
        yield dict(cohort=cohort, scope=scope, policy=policy, lanes=lane, pe_period_ns=period,
                   scorer_variant=variant, reference_mode=ref, candidate_mode=candidate,
                   coefficient_ii=coeff, coefficient_latency_pe_cycles=latency,
                   sram_capacity_bytes=sram, pairs=len(rows), reference_mean_ns=a, candidate_mean_ns=b,
                   candidate_speedup=a / b, mean_delta_ns=fmean(ds), p95_delta_ns=ds[ceil(.95 * len(ds)) - 1],
                   reference_fits=all(row[0]["feasible_sram"] for row in rows),
                   candidate_fits=all(row[1]["feasible_sram"] for row in rows))


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fields)
        writer.writeheader()
        writer.writerows(rows)


def measurement_rows(rows):
    keys = ("sample_id", "feasible_sram", *TIME_AND_ACTIVITY)
    return [{k: row[k] for k in keys if k in row} for row in rows]


def replay_task(task):
    index, kind, states, config, directory = task
    ordered = sorted(states, key=lambda row: (row["step"], row["layer"], row["group"], row["policy"]))
    prepared = [prepare_group(row) for row in ordered]
    joined = []
    if kind == "whole":
        rows = []
        for captured in replay_step(prepared, config):
            rows.extend(captured)
            joined.append(join_step(captured, config["layers"], config["groups"]))
    else:
        rows = list(scenario_rows(prepared, config))
    # Persist raw wide rows in bounded worker batches. Return only compact
    # columns needed for the p95, paired and whole-step reports.
    path = Path(directory) / f"{index:06d}.csv.gz"
    # Identical columns for every task. IDs and metrics are fixed by the replay.
    fields = list(rows[0])
    with path.open("wb") as stream:
        compressed = gzip.GzipFile(fileobj=stream, filename="", mode="wb", compresslevel=3, mtime=0)
        with TextIOWrapper(compressed, encoding="utf-8", newline="") as chunk:
            csv.DictWriter(chunk, fields).writerows(rows)
    labels = [(tuple(row[key] for key in IDENTITY), row["scope"], row["policy"],
               tuple(row[key] for key in SCENARIO)) for row in rows]
    return path, fields, joined, list(zip(labels, measurement_rows(rows), strict=True))


def analyze(paths, output, *, lanes=LANES, periods=(PRIMARY_PE_NS, ROUND9_SCORER_REFERENCE_NS),
            modes=MODES, ii6=True, coefficient_ii=1, coefficient_latency=2,
            sram_capacity_bytes=SRAM_BYTES, policies=(), layers=36, groups=8, workers=1):
    isolated, complete, observations = load_states(paths, policies, layers, groups)
    config = dict(lanes=lanes, periods=periods, modes=modes, ii6=ii6, coefficient_ii=coefficient_ii,
                  coefficient_latency=coefficient_latency, sram_capacity_bytes=sram_capacity_bytes,
                  policies=policies, layers=layers, groups=groups)
    pending = [("isolated", rows) for _, rows in sorted(isolated.items())]
    pending.extend(("whole", rows) for _, rows in sorted(complete.items()))
    output.mkdir(parents=True, exist_ok=True)
    group_stats, step_stats, pairs = defaultdict(list), defaultdict(list), defaultdict(dict)
    result_rows = []
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=output) as directory:
        tasks = ((index, kind, rows, config, directory) for index, (kind, rows) in enumerate(pending))
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            # Concatenated gzip members carry all rows with one CSV header.
            compressed = Path(directory) / "group_timing.csv.gz"
            with open(compressed, "wb") as dest:
                fields = None
                done = 0
                completed = pool.map(replay_task, tasks) if pool else map(replay_task, tasks)
                for path, chunk_fields, joined, summary in completed:
                    if fields is None:
                        fields = chunk_fields
                        with gzip.GzipFile(fileobj=dest, mode="wb", mtime=0) as header:
                            header.write((",".join(fields) + "\r\n").encode())
                    if fields != chunk_fields:
                        raise ValueError("group timing fields changed within the replay")
                    with open(path, "rb") as chunk:
                        for block in iter(lambda: chunk.read(1 << 20), b""):
                            dest.write(block)
                    path.unlink()
                    result_rows.extend(joined)
                    for row in joined:
                        scenario = tuple(row[key] for key in SCENARIO)
                        step_stats[row["cohort"], "whole_step", row["policy"], *scenario].append(row)
                        pairs[row["cohort"], row["sample_id"], row["step"], "whole_step", *scenario][row["policy"]] = row
                    for (identity, scope, policy, scenario), row in summary:
                        if joined:
                            scope = "whole_step_group_" + scope
                        group_stats[identity[0], scope, policy, *scenario].append(row)
                        pairs[*identity, scope, *scenario][policy] = row
                    done += 1
                    if done % 6 == 0 or done == len(pending):
                        print(f"Round 11 replay {done}/{len(pending)} prompt or whole-step tasks", flush=True)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)

        destination = Path(directory)
        files = {
            "group_timing.csv.gz": None,
            "timing_summary.csv": summarize(group_stats),
            "step_timing.csv": result_rows,
            "step_timing_summary.csv": summarize(step_stats),
            "paired_timing.csv": paired_summaries(pairs),
            "paired_modes.csv": paired_modes(pairs),
        }
        for name, rows in files.items():
            if rows is not None:
                write_csv(destination / name, rows)
        receipt = {
            "scope": "CPU event projection of attention only",
            "inputs": [{"file": str(p), "sha256": digest(p)} for p in paths],
            "model_source_sha256": {name: digest(Path(__file__).resolve().parent.parent / name)
                                    for name in ("sim/lpddr5_banked.cpp", "sim/round11_timing.cpp",
                                                 "gqa_probe/round11_timing.py", "scripts/replay_round11.py")},
            "configuration": config,
            "primary": dict(lanes=64, pe_period_ns=PRIMARY_PE_NS, mode="double_buffer", scorer_variant=BASE_SCORER),
            "group_captures": [dict(cohort=c, policy=p, scope=s, whole_step=w, groups=n)
                               for (c, p, s, w), n in sorted(observations.items())],
            "whole_steps": len({key[:3] for key in complete}),
            "whole_step_policy_captures": len(complete),
            "group_timing_rows": sum(map(len, group_stats.values())),
            "step_timing_rows": len(result_rows),
            "ports": {
                "LPDDR": "one x16 LPDDR5-6400 channel, 32 B burst, 1.25 ns CK, 16 banks, four groups, 2 KiB rows",
                "read_window_bursts": QUEUE_DEPTH, "descriptor_interlaunch_ck": 2,
                "bank_scheduler": "FIFO per bank, ready RD priority, then oldest ready bank head",
                "physical_limits": "read-only, open-row commands, no refresh, writes, WCK switches, or co-issued queries",
                "packed_slots": "independent 1R1W banks, 32 B per DRAM CK write and 32 B per PE cycle read",
                "unpacked_slots": "independent 1R1W banks, 64 B/PE cycle write, 64/128/256 B/PE cycle PV read for 32/64/128 lanes",
                "qk_broadcast": "one K element can serve four head lanes, with 16/32/64 B per PE cycle unpacked read",
                "score_and_histogram": "16 B logits per PE cycle for gate, one bank per head and one bin per head per PE cycle",
                "port_contention": "zero under independent slot banks with the listed widths; no timed SRAM macros",
                "coefficient_supply": "two scale operands per prepared record, one separate frontend issue and configurable result latency",
            },
            "modes": {
                "serial": "all summary reads before scoring, all K or V reads before backend work; reports bulk-buffer requirement",
                "serial_one_slot": "single summary slot waits through scorer result, one tile waits through QK or PV",
                "summary_stream": "two summary records, release at scorer input service completion; K and V use one tile each",
                "double_buffer": "two summary records and two K or V tile slots, held through QK or PV; one shared MAC backend",
                "dependency_bound": "same bank replay with all slots admissible; relaxed_dependency is the separate strict resource lower bound",
            },
            "candidate_modes": CANDIDATE_MODES,
            "dependency_lower_bound": "per phase max of ideal burst bandwidth, MAC issue, and unpack service; preserves normalization, TopK, gate and optional rerun",
            "peak_sram": "simultaneous score, logits, masks, histograms, rank IDs, tiles, raw/quantized query, PV weights/accumulators and queue stage footprint; finite tile banks reserve capacity for a full page each, and the independently reported live peak counts only occupied bytes",
            "stage_sram_reuse": "the summary/score, K, gate and V stages can reuse SRAM after their predecessor retires; packed and unpacked buffering occupy separately ported banks",
            "queue_accounting": "32 16-byte request slots and 16-byte per outstanding descriptor without duplicated summary bursts",
            "whole_step_execution": "all captured layers and groups execute in layer/group order, on one shared resource and continuous bank/clock state",
            "whole_step_address_layout": dict(region_bytes=GROUP_REGION_BYTES, cyclic_bank_rotation_row_bytes=BANK_ROW_BYTES),
            "whole_step_population": "all groups in each selected endpoint, including naturally occurring fallback; excludes isolated pressure enrichment",
            "isolated_population": "ordinary and fallback-pressure group captures from a cold command controller; reports strata independently",
            "metrics": "relative group nanoseconds, earliest actual V command/RD, nearest-rank observed p95; no population weighting",
            "post_scorer_compute_ns": "sum of planner, normalization, ranking, histogram and byte guard, unpack, MAC, PV weight and finalization service; can include simultaneous service on distinct engines",
            "overlap_and_backpressure": "overlap intersects scheduled burst data intervals with scorer or MAC activity; queue wait is elapsed time at the request cap with an eligible next descriptor; slot wait is elapsed time with an unissued item behind a full slot window",
            "fallback_workset": "the hist64 rerun reuses full initial logits and the same mask/histogram scratch; it does not reload K or issue provisional V",
            "pe_clock_reference": "1.25 ns is an 800 MHz model scenario; 2.385448 ns is a rounded Round 9 pre-layout scorer-only clock reference",
            "files": {name: digest(destination / name) for name in files},
        }
        name = "timing_receipt.json"
        (destination / name).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        for filename in (*files, name):
            os.replace(destination / filename, output / filename)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, nargs="+", choices=LANES, default=LANES)
    parser.add_argument("--pe-period-ns", type=float, nargs="+",
                        default=(PRIMARY_PE_NS, ROUND9_SCORER_REFERENCE_NS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--no-ii6", action="store_true")
    parser.add_argument("--coefficient-ii", type=int, default=1)
    parser.add_argument("--coefficient-latency", type=int, default=2)
    parser.add_argument("--sram-capacity-bytes", type=int, default=SRAM_BYTES)
    parser.add_argument("--policies", nargs="+", default=())
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    receipt = analyze(args.states, args.output, lanes=args.lanes, periods=args.pe_period_ns,
                      modes=args.modes, ii6=not args.no_ii6,
                      coefficient_ii=args.coefficient_ii, coefficient_latency=args.coefficient_latency,
                      sram_capacity_bytes=args.sram_capacity_bytes, policies=args.policies,
                      layers=args.layers, groups=args.groups, workers=args.workers)
    print(json.dumps({k: receipt[k] for k in ("group_timing_rows", "step_timing_rows", "whole_steps")}))


if __name__ == "__main__":
    main()
