"""Calibrate whole-step replay to the measured, matched Round 13 frontend clocks."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from gqa_probe.round11_timing import FrontendTiming, saturated_frontend_schedule
from scripts.replay_round11 import digest as sha256


ROOT = Path("results/round13_comrad/hardware")


def check_recurrence(path, timing, stride):
    with path.open(newline="") as source:
        epochs = {}
        for row in csv.DictReader(source):
            epochs.setdefault(row["epoch"], []).append([int(row[key]) for key in
                                                       ("complete", "dot_first", "dot_last", "output")])
    total = 0
    for epoch, rows in epochs.items():
        array = np.array(rows, dtype=np.int64)
        array[:, 2] += 1  # Last dot edge retires its slot for the next edge.
        array -= array[0, 0]
        model = saturated_frontend_schedule(timing, len(rows), stride)
        model -= model[0, 0]
        if not np.array_equal(array, model):
            mismatches = np.flatnonzero(np.any(array != model, axis=1))
            start = int(mismatches[0])
            raise ValueError(f"{path.name} epoch {epoch} non-stalled recurrence diverged at record {start}. "
                             f"RTL {array[start].tolist()}, replay {model[start].tolist()}")
        total += len(rows)
    return total


def calibrated_record(dir_, case, rtl, mapped):
    measure = rtl[case]["continuous"]
    design = mapped[case]
    dim = design["dimension"]
    slots, credits = design["record_slots"], design["result_credits"]
    metrics = measure["edge_metrics"]
    trace = dir_/"cycles"/f"{case}_continuous.csv"
    if sha256(trace) != measure["cycle_trace_sha256"]:
        raise ValueError(f"RTL cycle trace hash differs for {case}")
    if slots != measure["record_slots"] or credits != measure["result_fifo_slots"]:
        raise ValueError(f"RTL and mapping use different capacity for {case}")
    timing = FrontendTiming(
        profile=case,
        clock_period_ns=design["critical_period_ns"],
        record_interval_cycles=measure["steady_complete_cadence"]["effective_ii_cycles"],
        record_to_dot_cycles=metrics["complete_to_dot_first"]["min"],
        record_to_result_cycles=metrics["complete_to_output"]["min"],
        record_release_cycles=metrics["complete_to_dot_last"]["min"]+1,
        dot_ii_cycles=measure["raw_dot_service_cycles"],
        dot_tail_cycles=metrics["dot_last_to_output"]["min"],
        coefficient_service_cycles=measure["coefficient_record_issue_cycles"],
        coefficient_latency_cycles=metrics["coefficient_first_issue_to_all_ready"]["min"],
        record_slots=slots,
        ingress_bytes_per_cycle=32,
        ingress_cdc_cycles=3,
        result_cdc_pe_cycles=2,
        auxiliary_bytes=4*(dim+8) + credits*32 + 2,
        result_fifo_slots=credits,
        query_load_cycles=measure["raw_dot_service_cycles"]+1,
        burst_fifo_slots=design["receive_fifo_slots"],
    )
    n = check_recurrence(trace, timing, dim+16)
    return dict(stride_bytes=dim+16,timing=asdict(timing),
                timing_evidence=dict(rtl_profile=case, rtl_log_sha256=measure["log_sha256"],
                                     cycle_trace_sha256=sha256(trace), matched_steady_trace_records=n,
                                     record_interval_pattern=measure["steady_complete_cadence"]["interval_pattern"],
                                     saturated_model_checked=True, scope="Integrated frontend including actual receive FIFO, dual issue, tail, and result credits"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware",type=Path,default=ROOT)
    opts=parser.parse_args()
    dir_=opts.hardware
    rtl_path, mapped_path = dir_/"rtl_validation.json",dir_/"synthesis.json"
    rtl = json.loads(rtl_path.read_text())
    mapped = json.loads(mapped_path.read_text())
    if mapped["rtl_validation_sha256"]!=sha256(rtl_path):
        raise ValueError("Mapped design references a different RTL validation receipt")
    conditions = []
    for s,c in ((2,2),(4,2),(4,3)):
        name=f"full_match_s{s}_c{c}"
        records={str(dim):calibrated_record(dir_,f"full{dim}_s{s}_c{c}",rtl["designs"],mapped["designs"])
                 for dim in (128,96)}
        conditions.append(dict(name=name,records=records,
                               evidence="Both scored widths use their own mapped scorer period and matched dual issue, tail, FIFO depth, slots, and result credits."))
    path = dir_/"frontend_calibration.json"
    payload = dict(sources=[dict(file=str(p),sha256=sha256(p)) for p in (rtl_path,mapped_path)],
                 conditions=conditions,
                 clock_domains="Separate per-width measured pre-layout scorer clock and common 1.25 ns backend PE.",
                 cdc="Two modeled scorer cycles for the DRAM-side crossing and one demonstrated synchronous receive FIFO cycle. Two PE cycles for the modeled scorer-result crossing.",
                 shared_sram="512 B receive payload, 16 LAST bits, rounded 32 B output per credit, packed record stride and 32 B prepared metadata per slot, Q magnitude and gains; input/result stages share the 64 KiB workset under the established replay.",
                 port_assumptions="One receive FIFO write and read per scorer edge, ordered 32 B burst feed, up to two independent 16 B record destinations per straddle, and one result FIFO write/read. Clocking and SRAM macro ports follow model assumptions.",
                 verification="Every continuous-test record crosses the same modeled complete, first-dot, slot-reuse, and output edge relative to the first assembled record. Source recovery and CDC timing are separately modeled.")
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(sum(y["timing_evidence"]["matched_steady_trace_records"] for x in conditions
              for y in x["records"].values()),"exact record schedules")


if __name__=="__main__":
    main()
