"""Cycle-model sensitivity to record slot depth, checked against the Round 12 RTL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_round12_rtl import TOP, between, gap, oracle, repeated_cadence, sha256, summarize, synthetic


def schedule(q, g, records, expected, depth, lanes, tail, coefficient_latency, result_credits):
    """Same beat/slot ordering as the RTL, with a parameterized record capacity."""
    dim = len(q[0])
    stride = dim + 16
    chunks = dim // 16
    pages = len(records)
    payload = b"".join(records)
    offset = 0
    slot = {}
    ready = {}
    operation = {}
    completions, issues, starts, releases, results = [], [], [], [], []
    coefficient_page = 0
    coefficient_free = 0
    dot_free = 0
    input_blocked = 0
    peak_slots = 0
    active_dot = None
    for cycle in range(20_000):
        if operation.get(len(results)) == cycle:
            results.append(cycle)
        if len(results) == pages:
            break

        # Slot credit is evaluated before this edge's last-dot release.
        if offset < len(payload):
            limit = min(offset + 32, len(payload))
            ids = range(offset // stride, (limit - 1) // stride + 1)
            fresh = sum(page not in slot for page in ids)
            if len(slot) + fresh <= depth:
                for page in ids:
                    first = max(offset, page * stride)
                    last = min(limit, (page + 1) * stride)
                    data = slot.setdefault(page, bytearray())
                    data.extend(payload[first:last])
                    if len(data) == stride:
                        if bytes(data) != records[page] or page != len(completions):
                            raise AssertionError("misassembled ordered record")
                        completions.append(cycle)
                offset = limit
                peak_slots = max(peak_slots, len(slot))
            else:
                input_blocked += 1

        if (coefficient_page < len(completions) and cycle >= coefficient_free and
                cycle > completions[coefficient_page]):
            issues.append(cycle)
            issue_cycles = 8 // lanes
            ready[coefficient_page] = cycle + issue_cycles - 1 + coefficient_latency
            coefficient_free = cycle + issue_cycles
            coefficient_page += 1

        next_dot = len(starts)
        if (next_dot in ready and cycle > ready[next_dot] and cycle >= dot_free and
                len(starts) - len(results) < result_credits):
            value, _ = oracle(q, g, [slot[next_dot]], dim)
            if value != [expected[next_dot]]:
                raise AssertionError(f"Q24 score differs at record {next_dot}")
            active_dot = next_dot
            starts.append(cycle)
            dot_free = cycle + chunks
            operation[next_dot] = dot_free - 1 + tail
        if active_dot is not None and cycle == dot_free - 1:
            slot.pop(active_dot)
            active_dot = None
            releases.append(cycle)
    else:
        raise AssertionError("cycle model did not retire every record")
    return dict(complete=completions, issue=issues, dot_first=starts, dot_last=releases,
                output=results, blocked_cycles=input_blocked, occupied_slots=peak_slots)


def summarize_depth(depth, tail, reference, credits):
    model = [schedule(q, g, records, scores, depth, 2, tail, 9, credits)
             for q, g, records, scores, _coeffs in synthetic(96)]
    summary = {}
    for key, a, b in (
        ("complete_to_dot_first", "dot_first", "complete"),
        ("complete_to_dot_last", "dot_last", "complete"),
        ("complete_to_output", "output", "complete"),
        ("dot_last_to_output", "output", "dot_last"),
        ("coefficient_complete_to_first_issue", "issue", "complete"),
    ):
        summary[key] = summarize([x for epoch in model for x in between(epoch[a], epoch[b])])
    intervals = [gap(epoch["complete"])[2:-1] for epoch in model]
    cadence = repeated_cadence(intervals)
    if not cadence:
        cadence = dict(interval_pattern=None,
                       effective_ii_cycles=sum(sum(x) for x in intervals)/sum(len(x) for x in intervals))
    if reference:
        rtl = reference["continuous"]
        for key, metric in summary.items():
            if metric != rtl["edge_metrics"][key]:
                raise AssertionError(f"two-slot model disagrees with RTL for {key}")
        if cadence != rtl["exact_steady_record_cadence"]:
            raise AssertionError("two-slot model cadence differs from RTL")
        if sum(epoch["blocked_cycles"] for epoch in model) != rtl["stalled_input_cycles"]:
            raise AssertionError("two-slot model ingress backpressure differs from RTL")
    return dict(
        model_checked_against_rtl=bool(reference), record_slots=depth, result_credits=credits,
        verified_scores=4*sum(len(x["output"]) for x in model),
        first_burst_to_first_output=[x["output"][0] for x in model],
        observed_peak_record_slots=max(x["occupied_slots"] for x in model),
        blocked_ingress_cycles=sum(x["blocked_cycles"] for x in model),
        exact_or_sampled_cadence=cadence,
        steady_completion_intervals=summarize([x for epoch in intervals for x in epoch]),
        edge_metrics=summary,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = TOP / "results/round12_frontend/hardware"
    parser.add_argument("--rtl-validation", type=Path, default=base / "rtl_validation.json")
    parser.add_argument("--result", type=Path, default=base / "depth_sensitivity.json")
    opts = parser.parse_args()
    profiles = json.loads(opts.rtl_validation.read_text())["designs"]
    result = {}
    for tag in ("full96_dual", "patched96_dual"):
        tail = profiles[tag]["continuous"]["edge_metrics"]["dot_last_to_output"]["min"]
        result[tag] = dict(
            main_two_slots=summarize_depth(2, tail, profiles[tag], 2),
            four_record_slots_same_result_credits=summarize_depth(4, tail, None, 2),
            four_record_slots_three_result_credits=summarize_depth(4, tail, None, 3),
        )
    opts.result.write_text(json.dumps(dict(
        scope="Calibrated cycle model using exact burst bytes and integer scores; no added-slot RTL synthesis or placement.",
        rtl_validation_sha256=sha256(opts.rtl_validation), model_source_sha256=sha256(Path(__file__)),
        designs=result,
    ), indent=2, sort_keys=True) + "\n")
    for tag, rows in result.items():
        print(tag, {k: v["exact_or_sampled_cadence"]["effective_ii_cycles"] for k, v in rows.items()})


if __name__ == "__main__":
    main()
