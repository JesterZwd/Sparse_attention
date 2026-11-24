"""Project the integrated Round 12 summary frontend on paired whole-step captures."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
import json
from math import ceil
import os
from pathlib import Path
from statistics import fmean

from gqa_probe.round11_energy import SCENARIOS
from gqa_probe.round11_timing import (
    BASE_SCORER, FAST_SCORER, PRIMARY_PE_NS, SRAM_BYTES, BankSession,
    FrontendTiming, group_address_offset, prepare_group, replay_group, saturated_frontend_schedule,
)
from scripts.replay_round11 import digest, join_step, load_states, write_csv


POLICIES = ("A", "B", "M", "N80", "N64")
FRONTEND_ADDITIVE = (
    "frontend_ingress_queue_wait_ns", "frontend_record_service_wait_ns",
    "frontend_result_cdc_ns", "frontend_ingress_port_busy_ns",
)
TIMING_METRICS = (
    "completion_ns", "first_v_command_ns", "first_v_read_ns", "per_group_mean_first_v_command_ns",
    "frontend_phase_ns", "post_scorer_ns", "frontend_score_ready_ns", "scorer_busy_ns", "coefficient_busy_ns",
    "ranking_phase_ns", "k_qk_phase_ns", "gate_phase_ns", "v_pv_phase_ns",
    "frontend_summary_read_ns", "frontend_saturated_span_ns", "frontend_excess_span_ns",
    "frontend_observed_record_ns", "frontend_summary_bytes_per_ns", "score_data_overlap_ns", "dram_data_util",
    "scorer_wait_ns", "qk_wait_ns", "pv_wait_ns", "summary_slot_wait_ns", "k_slot_wait_ns",
    "v_slot_wait_ns", "request_window_wait_ns", *FRONTEND_ADDITIVE, "peak_sram_bytes",
    "peak_live_sram_bytes", "peak_summary_sram_bytes", "peak_frontend_reserved_bytes",
    "peak_k_sram_bytes", "peak_gate_sram_bytes", "peak_v_sram_bytes",
    "peak_summary_slots", "summary_bytes", "k_bytes", "v_bytes", "total_bytes", "act", "pre", "rd",
    "fallback_groups",
)
ENERGY_METRICS = (
    "ddr_commands_uj", "ddr_io_uj", "ddr_background_uj", "ddr_uj", "front_dot_uj",
    "front_coefficient_uj", "front_packed_sram_uj", "front_score_sram_uj", "front_clock_uj",
    "front_uj", "front_plus_ddr_uj",
)
EXTRA_ENERGY = {
    "lower": dict(coeff_pj_per_product=.30, front_mw=.5),
    "nominal": dict(coeff_pj_per_product=1.50, front_mw=2),
    "upper": dict(coeff_pj_per_product=6.00, front_mw=8),
}


@dataclass(frozen=True)
class Condition:
    name: str
    settings: dict
    legacy_reference_channels: tuple = ()
    ideal: bool = False
    legacy_scorer_period_ns: float | None = None
    legacy_burst_fifo_slots: int = 0

    def select(self, channels, stride):
        if channels in self.settings:
            expected_stride, timing = self.settings[channels]
            if expected_stride != stride:
                raise ValueError(f"{self.name} expects {expected_stride} B at {channels} channels, found {stride} B")
            return BASE_SCORER, timing, None, 0
        if channels in self.legacy_reference_channels:
            return BASE_SCORER, None, self.legacy_scorer_period_ns, self.legacy_burst_fifo_slots
        if self.ideal:
            return (FAST_SCORER if channels == 96 else BASE_SCORER), None, None, 0
        if not self.settings:
            return BASE_SCORER, None, None, 0
        raise ValueError(f"{self.name} has no replay calibration for {channels} channels")


def read_conditions(path):
    result = [Condition("round11_common_ii8", {}), Condition("round11_ideal_ii6", {}, ideal=True)]
    if path is None:
        return result, None
    spec = json.loads(path.read_text())
    for item in spec["conditions"]:
        settings = {int(channels): (value["stride_bytes"], FrontendTiming(**value["timing"]))
                    for channels, value in item["records"].items()}
        if not settings or not item.get("evidence"):
            raise ValueError("measured conditions require calibrated record timing and evidence")
        result.append(Condition(item["name"], settings, tuple(item.get("legacy_reference_channels", ())),
                                legacy_scorer_period_ns=item.get("legacy_scorer_period_ns"),
                                legacy_burst_fifo_slots=item.get("legacy_burst_fifo_slots", 0)))
    if len(set(c.name for c in result)) != len(result):
        raise ValueError("duplicate replay condition names")
    return result, spec


def measured_record(rtl, mapping, profile, clock_name, fifo_slots):
    design = rtl[profile]
    capture = design["continuous"]
    channels, stride = design["record_channels"], design["stride_bytes"]
    metrics = capture["edge_metrics"]
    cadence = capture["exact_steady_record_cadence"]
    slots, heads = capture["result_fifo_slots"], capture["group_heads"]
    edge = capture["raw_dot_service_cycles"]
    timing = FrontendTiming(
        profile=profile,
        clock_period_ns=mapping[clock_name]["critical_period_ns"],
        record_interval_cycles=cadence["effective_ii_cycles"],
        record_to_dot_cycles=metrics["complete_to_dot_first"]["min"],
        record_to_result_cycles=metrics["complete_to_output"]["min"],
        record_release_cycles=metrics["complete_to_dot_last"]["min"] + 1,
        dot_ii_cycles=edge,
        dot_tail_cycles=metrics["dot_last_to_output"]["min"],
        coefficient_service_cycles=capture["coefficient_record_issue_cycles"],
        coefficient_latency_cycles=metrics["coefficient_first_issue_to_all_ready"]["min"],
        record_slots=capture["record_slots"],
        ingress_bytes_per_cycle=32,
        ingress_cdc_cycles=2,
        result_cdc_pe_cycles=2,
        auxiliary_bytes=heads * (channels + 8) + slots * 32,
        result_fifo_slots=slots,
        query_load_cycles=edge + 1,
        burst_fifo_slots=fifo_slots,
    )
    # Both test epochs contain a bounded startup/transient region. Compare a
    # long no-stall sequence with the measured periodic region up to rotation.
    pattern = cadence["interval_pattern"]
    complete, dot, release, output = saturated_frontend_schedule(timing, 64, stride).T
    steady = (complete[9:] - complete[8:-1]).tolist()
    mirrored = any(all(delta == pattern[(index + offset) % len(pattern)]
                       for index, delta in enumerate(steady)) for offset in range(len(pattern)))
    observed = {"complete_to_dot_first": Counter((dot - complete).tolist()),
                "complete_to_dot_last": Counter((release - complete - 1).tolist()),
                "complete_to_output": Counter((output - complete).tolist())}
    if (not mirrored or fmean(pattern) != timing.record_interval_cycles or
            any(set(hist) != set(map(int, metrics[name]["histogram"])) for name, hist in observed.items())):
        raise ValueError(f"the no-stall event schedule disagrees with {profile} RTL")
    return dict(stride_bytes=stride, timing=asdict(timing),
                timing_evidence=dict(rtl_profile=profile, rtl_log_sha256=capture["log_sha256"],
                                     clock_profile=clock_name,
                                     clock_basis="mapped pre-layout STA extrapolation" if clock_name == profile else
                                     "other profile clock proxy; parameter timing is unmeasured",
                                     record_interval_pattern=pattern,
                                     complete_to_first_dot_cycles=metrics["complete_to_dot_first"],
                                     complete_to_last_dot_cycles=metrics["complete_to_dot_last"],
                                     complete_to_result_cycles=metrics["complete_to_output"],
                                     first_coefficient_issue_to_all_ready_cycles=metrics["coefficient_first_issue_to_all_ready"],
                                     saturated_model_checked=True))


def write_calibration(rtl_path, mapping_path, destination, narrow_rtl_path=None, narrow_mapping_path=None):
    rtl = json.loads(rtl_path.read_text())["designs"]
    mapping = json.loads(mapping_path.read_text())["designs"]
    receipt_paths = [rtl_path, mapping_path]
    if narrow_rtl_path is not None and narrow_mapping_path is not None:
        rtl.update(json.loads(narrow_rtl_path.read_text())["designs"])
        mapping.update(json.loads(narrow_mapping_path.read_text())["designs"])
        receipt_paths.extend((narrow_rtl_path, narrow_mapping_path))
    old = mapping["old128"]
    sources = [dict(file=str(path), sha256=digest(path)) for path in receipt_paths]
    output = dict(
        sources=sources,
        timing_convention="Accepted scorer edges, period from a common-library pre-layout STA path. Record release is the edge after the measured last dot. No-stall timing assumes continuously ready result consumer.",
        clock_domains="Measured scorer clock and independent 1.25 ns backend PE. The A projection requires four copies of the old native one-head preloaded scorer. Its separate two-cycle coefficient producer continues at the PE clock under the Round 11 proxy.",
        old128_projection=dict(mapped_period_ns=old["critical_period_ns"], mapped_one_head_area_um2=old["area_um2"],
                               four_parallel_raw_area_um2=4 * old["area_um2"],
                               replay_assumptions="old128 RTL uses precomputed coefficients and 16-byte chunks; projected A keeps Round 11 packed-summary ingress and 1.25 ns coeff timing", **rtl["old128"]["continuous"]),
        shared_sram="Each real record slot reserves stride bytes plus 32 bytes for coefficients, tag and derived metadata. Reserve 32 bytes for a shared burst, 32 bytes for four FP64 gains, q8 query storage already provisioned in the Round 11 ledger, extra q8 magnitude storage, and two rounded 32 B result entries. Optional 16 × 32 B return FIFO uses the same 64 KiB allowance. Per-group phase reuse is allowed.",
        cdc="Ingress and query add two scorer cycles after scorer-edge alignment. Scoreboard return adds two PE cycles after PE-edge alignment. Credit feedback uses the modeled last-dot release edge and the next LPDDR scheduling edge. These are replay assumptions.",
        conditions=[],
    )
    configs = [
        ("mapped_dual_tight_a125", "full96_dual", "dual", 0, None, 0),
        ("mapped_dual_fifo16_a125", "full96_dual", "dual", 16, None, 0),
        ("mapped_dual_fifo16_a128", "full96_dual", "dual", 16, old["critical_period_ns"], 0),
        ("single_fifo16_a128_clock_proxy", "full96_single", "single", 16, old["critical_period_ns"], 0),
    ]
    if "patched96_dual" in mapping and "patched96_dual" in rtl:
        configs.extend((name, "patched96_dual", "dual", 16, clock, legacy_fifo) for name, clock, legacy_fifo in (
            ("mapped_patched_fifo16_a125", None, 0),
            ("mapped_patched_fifo16_a128", old["critical_period_ns"], 0),
            ("mapped_patched_equalfifo16_a125", None, 16),
            ("mapped_patched_equalfifo16_a128", old["critical_period_ns"], 16),
        ))
    else:
        output["unavailable_conditions"] = ["patched replay requires a verified RTL cycle trace and its own mapped clock"]
    for name, profile, lanes, fifo, clock, legacy_fifo in configs:
        records = {}
        for channels in (96, 80, 64):
            patched = f"patched{channels}_{lanes}"
            chosen = (profile if channels == 96 else patched if profile == "patched96_dual" and patched in mapping
                      else f"full{channels}_{lanes}")
            if chosen not in rtl:
                continue
            clock_name = chosen if chosen in mapping else "full96_dual"
            records[str(channels)] = measured_record(rtl, mapping, chosen, clock_name, fifo)
        output["conditions"].append(dict(
            name=name, legacy_reference_channels=[128], legacy_scorer_period_ns=clock,
            legacy_burst_fifo_slots=legacy_fifo, records=records,
            evidence="Integrated continuous-ready coefficient, dot, result and slot-release edges. The narrow and single-issue profiles use a proxy period where no same-profile map is available.",
        ))
    patched = next((item for item in output["conditions"] if item["name"] == "mapped_patched_fifo16_a128"), None)
    if patched and "full128_dual" in rtl:
        output["conditions"].append(dict(
            name="integrated_a128_clock_proxy_fifo16", legacy_reference_channels=[],
            records={**patched["records"], "128": measured_record(rtl, mapping, "full128_dual", "old128", 16)},
            evidence="All formats include measured integrated coefficients, slots, and burst ingress at equal receive FIFO depth. The full128 cycle trace uses the separately mapped preloaded one-head old128 period as a clock proxy.",
        ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name)
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


@lru_cache(maxsize=128)
def saturated_span(frontend, count, stride):
    cycles = saturated_frontend_schedule(frontend, count, stride)
    return (int(cycles[-1, 3]) - int(cycles[0, 0])) * frontend.config()[0] / 1_000_000


def work(task):
    state_set, rows, conditions, layers, groups = task
    prepared = [prepare_group(row) for row in sorted(rows, key=lambda row: (row["layer"], row["group"]))]
    first = prepared[0].raw
    channels, stride, heads = (first[key] for key in ("scorer_channels", "summary_stride", "heads"))
    if any((p.raw["scorer_channels"], p.raw["summary_stride"], p.raw["heads"]) != (channels, stride, heads)
           or p.raw.get("answer_start") != first.get("answer_start") for p in prepared):
        raise ValueError("whole-step summaries disagree on geometry or answer timestamp")
    results, reused = [], {}
    for condition in conditions:
        variant, frontend, projection, legacy_fifo = condition.select(channels, stride)
        key = variant, frontend, projection, legacy_fifo
        if key in reused:
            results.append(dict(reused[key], condition=condition.name))
            continue
        with BankSession() as session:
            captured = [replay_group(p, lanes=64, pe_period_ns=PRIMARY_PE_NS, scorer_variant=variant,
                                     frontend=frontend, mode="double_buffer", sram_capacity_bytes=SRAM_BYTES,
                                     legacy_scorer_period_ns=projection,
                                     legacy_burst_fifo_slots=legacy_fifo,
                                     session=session, address_offset=group_address_offset(ordinal))
                        for ordinal, p in enumerate(prepared)]
        row = join_step(captured, layers, groups)
        total_front = sum(item["score_done_ns"] for item in captured)
        phases = {
            phase: sum(item[end] - item[start] for item in captured)
            for phase, start, end in (
                ("ranking_phase_ns", "score_done_ns", "topk_done_ns"),
                ("k_qk_phase_ns", "topk_done_ns", "qk_done_ns"),
                ("gate_phase_ns", "qk_done_ns", "gate_done_ns"),
                ("v_pv_phase_ns", "gate_done_ns", "completion_ns"),
            )
        }
        row.update(
            state_set=state_set, condition=condition.name, scorer_channels=channels, summary_stride=stride,
            answer_start=first.get("answer_start"), heads=heads,
            timestamp="decode_start" if row["step"] == 0 else "answer_start" if first.get("answer_start") else "later",
            frontend_phase_ns=total_front,
            frontend_score_ready_ns=captured[0]["score_done_ns"],
            frontend_summary_read_ns=sum(item["summary_read_done_ns"] for item in captured),
            frontend_summary_bytes_per_ns=row["summary_bytes"] / total_front,
            frontend_model=("integrated_cycle_recurrence" if frontend else
                            "old128_four_head_parallel_projection" if projection else "round11_preloaded_proxy"),
            scorer_clock_period_ns=captured[0].get("scorer_clock_period_ns", PRIMARY_PE_NS),
            scorer_ii=captured[0]["scorer_ii"],
            frontend_record_interval_cycles=frontend.record_interval_cycles if frontend else None,
            scorer_latency_cycles=captured[0].get("scorer_latency_cycles", captured[0]["scorer_latency_pe_cycles"]),
            coefficient_latency_cycles=frontend.coefficient_latency_cycles if frontend else row["coefficient_latency_pe_cycles"],
            frontend_record_to_result_cycles=frontend.record_to_result_cycles if frontend else None,
            frontend_record_slots=frontend.record_slots if frontend else 2,
            frontend_burst_fifo_slots=captured[0].get("frontend_burst_fifo_slots", 0),
            peak_frontend_reserved_bytes=(max(item["peak_frontend_reserved_bytes"] for item in captured)
                                          if frontend or legacy_fifo else None),
            **phases,
            **{metric: sum(item[metric] for item in captured) if frontend else None for metric in FRONTEND_ADDITIVE},
        )
        if frontend:
            saturated = sum(saturated_span(frontend, item["history_pages"], stride) for item in captured)
            actual = sum(item["frontend_last_result_ns"] - item["frontend_first_record_ns"] for item in captured)
            intervals = sum(item["history_pages"] - 1 for item in captured)
            row.update(frontend_saturated_span_ns=saturated, frontend_excess_span_ns=actual - saturated,
                       frontend_observed_record_ns=(sum(item["frontend_last_record_ns"] - item["frontend_first_record_ns"]
                                                        for item in captured) / intervals if intervals else None))
        for energy_case in SCENARIOS:
            row.update({f"{energy_case}_{name}": value for name, value in energy(row, energy_case).items()})
        results.append(row)
        reused[key] = row
    return results


def energy(row, case):
    p, extra = SCENARIOS[case], EXTRA_ENERGY[case]
    ddr_commands = sum(row[field] * p[coeff] for field, coeff in (
        ("act", "act_pj_per_command"), ("pre", "pre_pj_per_command"), ("rd", "rd_core_pj_per_burst")))
    ddr_io = row["total_bytes"] * p["read_io_pj_per_byte"]
    ddr_background = row["completion_ns"] * p["dram_background_mw"]
    ddr = ddr_commands + ddr_io + ddr_background
    front_dot = (row["scorer_terms"] * p["scorer_pj_per_term"] +
                 row["fused_updates"] * p["scorer_pj_per_fused_update"])
    front_coefficient = 2 * row["coefficient_records"] * row["heads"] * extra["coeff_pj_per_product"]
    front_packed_sram = 2 * row["summary_bytes"] * p["sram_pj_per_byte"]
    front_score_sram = 8 * row["coefficient_records"] * row["heads"] * p["sram_pj_per_byte"]
    front_clock = row["frontend_phase_ns"] * extra["front_mw"]
    front = front_dot + front_coefficient + front_packed_sram + front_score_sram + front_clock
    return {name: value / 1e6 for name, value in zip(ENERGY_METRICS, (
        ddr_commands, ddr_io, ddr_background, ddr, front_dot, front_coefficient,
        front_packed_sram, front_score_sram, front_clock, front, front + ddr,
    ), strict=True)}


def descriptive(numbers):
    numbers = sorted(numbers)
    return dict(mean=fmean(numbers), p95=numbers[ceil(.95 * len(numbers)) - 1], max=numbers[-1]) if numbers else {}


def named_stats(metric, numbers):
    return {f"{metric}_{name}": value for name, value in descriptive(numbers).items()}


def summarize(rows):
    sets = defaultdict(list)
    for row in rows:
        # Ordinarily describe effects over all captured steps. Also show the
        # timestamp strata to expose the small endpoint sample and heterogeneity.
        for timestamp in ("all", row["timestamp"]):
            sets[row["state_set"], row["cohort"], row["condition"], row["policy"], timestamp].append(row)
    for (state_set, cohort, condition, policy, timestamp), entries in sorted(sets.items()):
        first = entries[0]
        summary = dict(state_set=state_set, cohort=cohort, condition=condition, policy=policy, timestamp=timestamp,
                       steps=len(entries), prompts=len({entry["sample_id"] for entry in entries}),
                       steps_with_fallback=sum(entry["fallback_groups"] > 0 for entry in entries),
                       frontend_model=first["frontend_model"], scorer_variant=first["scorer_variant"],
                       scorer_clock_period_ns=first["scorer_clock_period_ns"], scorer_ii=first["scorer_ii"],
                       frontend_record_interval_cycles=first["frontend_record_interval_cycles"],
                       feasible_sram=all(entry["feasible_sram"] for entry in entries))
        for metric in TIMING_METRICS:
            summary.update(named_stats(metric, (entry[metric] for entry in entries if entry.get(metric) is not None)))
        yield summary


def collect_pairs(rows):
    index = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in ("state_set", "cohort", "condition", "sample_id", "step"))
        index[key][row["policy"]] = row
    pairs = defaultdict(list)
    for (state_set, cohort, condition, _, _), candidates in index.items():
        for left, right in combinations((name for name in POLICIES if name in candidates), 2):
            reference, candidate = candidates[left], candidates[right]
            if reference["timestamp"] != candidate["timestamp"]:
                raise ValueError("paired policies disagree on the capture timestamp")
            for timestamp in ("all", reference["timestamp"]):
                pairs[state_set, cohort, condition, left, right, timestamp].append((reference, candidate))
            if reference["fallback_groups"] or candidate["fallback_groups"]:
                pairs[state_set, cohort, condition, left, right, "fallback_present"].append((reference, candidate))
    return pairs


def comparison_rows(pairs, metrics):
    for (state_set, cohort, condition, reference, candidate, timestamp), entries in sorted(pairs.items()):
        row = dict(state_set=state_set, cohort=cohort, condition=condition, reference=reference,
                   candidate=candidate, timestamp=timestamp, pairs=len(entries),
                   prompts=len({a["sample_id"] for a, _ in entries}))
        row.update(all_reference_sram_fit=all(a["feasible_sram"] for a, _ in entries),
                   all_candidate_sram_fit=all(b["feasible_sram"] for _, b in entries))
        for metric in metrics:
            eligible = [(a[metric], b[metric]) for a, b in entries if a.get(metric) is not None and b.get(metric) is not None]
            row.update(named_stats(f"{metric}_reference", (a for a, _ in eligible)))
            row.update(named_stats(f"{metric}_candidate", (b for _, b in eligible)))
            row.update(named_stats(f"{metric}_delta", (b - a for a, b in eligible)))
        detail = "completion_ns" if "completion_ns" in metrics else f"{metrics[0].split('_')[0]}_front_plus_ddr_uj"
        row["candidate_to_reference_mean_ratio"] = row[f"{detail}_candidate_mean"] / row[f"{detail}_reference_mean"]
        if detail.endswith("_uj"):
            excess = [(a[detail] - b[detail]) * 1e6 / b["coefficient_records"] for a, b in entries]
            row["additional_candidate_front_pj_per_record_to_tie_mean"] = fmean(excess)
        yield row


def verify_archive(rows, archive):
    key_fields = ("cohort", "sample_id", "step", "policy", "lanes", "pe_period_ns", "scorer_variant", "mode",
                  "coefficient_ii", "coefficient_latency_pe_cycles", "sram_capacity_bytes")
    with archive.open(newline="") as stream:
        expected = {tuple(row[key] for key in key_fields): row for row in csv.DictReader(stream)}
    checked = 0
    table = {}
    for row in rows:
        if not row["condition"].startswith("round11_"):
            continue
        key = tuple(str(row[name]) for name in key_fields)
        other = expected[key]
        mismatch = {name: (other[name], row.get(name)) for name in other
                    if other[name] != ("" if row.get(name) is None else str(row[name]))}
        if mismatch:
            raise ValueError(f"Round 11 reproduction mismatch for {key}: {mismatch}")
        table[key] = len(other)
        checked += 1
    if checked == 0:
        raise ValueError(f"no baseline rows to compare to {archive}")
    return dict(file=str(archive), sha256=digest(archive), unique_rows=len(table),
                comparisons=checked, fields_per_comparison=sorted(set(table.values())))


def analyze(collections, output, *, calibration=None, archives=None, layers=36, groups=8, workers=1):
    conditions, spec = read_conditions(calibration)
    tasks, populations, inputs = [], [], []
    for name, paths in sorted(collections.items()):
        isolated, complete, counts = load_states(paths, POLICIES, layers, groups)
        if not complete:
            raise ValueError(f"{name} does not contain whole-step capture endpoints")
        endpoints = defaultdict(set)
        for (cohort, sample_id, step, policy), rows in sorted(complete.items()):
            endpoints[cohort, sample_id, step].add(policy)
            tasks.append((name, rows, conditions, layers, groups))
        for cohort, sample, step in endpoints:
            if not {"A", "B", "M"} <= endpoints[cohort, sample, step]:
                raise ValueError(f"missing A/B/M at {name}/{cohort}/{sample}/{step}")
        inputs.extend(dict(state_set=name, file=str(path), sha256=digest(path)) for path in paths)
        populations.append(dict(state_set=name, whole_steps=len(endpoints), whole_step_policy_captures=len(complete),
                                isolated_groups_excluded=sum(map(len, isolated.values())),
                                captured_groups=[dict(cohort=cohort, policy=policy, scope=scope,
                                                      whole_step=whole, groups=count)
                                                 for (cohort, policy, scope, whole), count in sorted(counts.items())]))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(work, tasks))
    else:
        completed = list(map(work, tasks))
    rows = [row for batch in completed for row in batch]
    matched = collect_pairs(rows)
    reproduction = []
    for label, path in sorted((archives or {}).items()):
        state_set, cohort = label.split("/", 1)
        archive_rows = [row for row in rows if row["state_set"] == state_set and row["cohort"] == cohort]
        reproduction.append(dict(state_set=state_set, cohort=cohort, **verify_archive(archive_rows, path)))

    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "step_timing.csv": rows,
        "step_timing_summary.csv": summarize(rows),
        "paired_timing.csv": comparison_rows(matched, TIMING_METRICS),
        "paired_energy.csv": (dict(energy_case=case, **row) for case in SCENARIOS
                              for row in comparison_rows(matched, tuple(f"{case}_{metric}" for metric in ENERGY_METRICS))),
    }
    for filename, data in tables.items():
        temporary = output / f".{filename}"
        write_csv(temporary, data)
        os.replace(temporary, output / filename)
    result = dict(
        scope="CPU event projection of decode attention only",
        geometry=dict(layers=layers, kv_groups=groups, backend_lanes=64, pe_period_ns=PRIMARY_PE_NS,
                      shared_sram_bytes=SRAM_BYTES, mode="double_buffer"),
        inputs=inputs, populations=populations, archived_baseline_reproduction=reproduction,
        calibration=None if calibration is None else dict(file=str(calibration), sha256=digest(calibration), **spec),
        pairing="Matched on state set, cohort, prompt and captured decode step. The group schedule preserves one LPDDR state across the complete layer/group traversal.",
        timestamp="decode_start is step zero. answer_start is the trace answer-start marker on nonzero steps. later is the remaining nonzero steps. fallback_present is the matched subset where either policy fell back.",
        population="Each captured whole step carries equal weight within its cohort and timestamp stratum. P95 is the nearest-rank observed value. A P95 of paired differences and each side's own P95 are both reported.",
        front_timing="The in-order physical region is consumed at a finite scorer-clock beat port. A straddling burst is fetched once and needs both record slots at RTL ingress. Coefficient issue, result readiness, dot issue, last dot slot release, and two output credits follow the measured RTL edge recurrence. The no-stall cadence is checked against RTL and emerges from these stages. The final result crosses to the PE domain before normalization. Optional burst FIFO space is reserved inside the same 64 KiB stage SRAM accounting. The backend phases mark normalization/ranking/planner, K/QK, gate/budget, and V/PV/finalization in order.",
        approximation="The calibrated ingress uses the cycle recurrence with continuous result ready as in the RTL reference test. LPDDR credit feedback, quantized CDC edges, bounded staging, and score writes are replay assumptions. A projected old128 clock uses the Round 11 summary ingress and coefficient preparation at the PE clock and an optimistic four-head parallel instance of the old one-head preloaded dot core. It preserves eight cycles per four-head record and assumes about four times the old dot core area before shared logic. The equal-FIFO scenarios give A the same burst-return budget with the Round 11 atomic ingress model. All backend compute stays at 1.25 ns.",
        stall_metrics="Scorer wait sums idle dot time, including initial service and input starvation. Record service wait measures delay beyond the isolated complete-to-first-dot minimum per record. Ingress queue wait sums visible-to-consumed intervals per burst and includes in-order reassembly and slot blocking. These residency sums may overlap across terms. Result CDC measures final scorer-to-PE alignment and modeled synchronization. Slot and request-window wait retain the Round 11 elapsed blocked-device definitions. Excess span compares observed first complete through last result against the same timed scorer with continuous ready input.",
        energy=dict(scope="Illustrative sampled decode attention frontend plus DRAM energy, in µJ/step. DDR terms use actual projected ACT/PRE/RD counts and all summary/K/V physical bytes. Backend compute and its SRAM, unmetered logic and leakage are outside this increment.",
                    coefficients=SCENARIOS, additional_coefficients=EXTRA_ENERGY,
                    front="Dot terms/fused updates, two coefficient products per head and record, two packed payload accesses, write/read of each result, and illustrative active frontend power. The same cost assumptions apply to all policies. Positive extra-front pJ/record to tie means the candidate may spend that additional energy before losing its modeled advantage."),
        model_source_sha256={name: digest(Path(__file__).resolve().parent.parent / name) for name in (
            "sim/lpddr5_banked.cpp", "sim/round11_timing.cpp", "gqa_probe/round11_timing.py",
            "scripts/replay_round11.py", "gqa_probe/round11_energy.py", "scripts/replay_round12.py")},
        files={filename: digest(output / filename) for filename in tables},
        step_timing_rows=len(rows), paired_timing_rows=len(matched),
    )
    temporary = output / ".replay_receipt.json"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "replay_receipt.json")
    return result


def labeled_path(value):
    label, separator, path = value.partition("=")
    if not label or not separator or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=labeled_path, action="append", required=True, metavar="SET=PATH")
    parser.add_argument("--verify-archive", type=labeled_path, action="append", metavar="SET/COHORT=PATH")
    parser.add_argument("--frontend-calibration", type=Path)
    parser.add_argument("--rtl-report", type=Path)
    parser.add_argument("--mapping-report", type=Path)
    parser.add_argument("--narrow-rtl-report", type=Path)
    parser.add_argument("--narrow-mapping-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if bool(args.rtl_report) != bool(args.mapping_report) or (
        args.frontend_calibration and args.rtl_report
    ):
        parser.error("provide either --frontend-calibration or both --rtl-report and --mapping-report")
    if bool(args.narrow_rtl_report) != bool(args.narrow_mapping_report) or (
        args.narrow_rtl_report and not args.rtl_report
    ):
        parser.error("both narrow reports require the primary RTL and mapping reports")
    if args.rtl_report:
        args.frontend_calibration = args.output / "frontend_calibration.json"
        write_calibration(args.rtl_report, args.mapping_report, args.frontend_calibration,
                          args.narrow_rtl_report, args.narrow_mapping_report)
    collections = defaultdict(list)
    for name, path in args.states:
        collections[name].append(path)
    receipt = analyze(collections, args.output, calibration=args.frontend_calibration,
                      archives=dict(args.verify_archive or ()), workers=args.workers)
    print(json.dumps({name: receipt[name] for name in ("step_timing_rows", "paired_timing_rows",
                                                     "archived_baseline_reproduction")}))


if __name__ == "__main__":
    main()
