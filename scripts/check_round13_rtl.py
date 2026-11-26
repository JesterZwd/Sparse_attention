"""Check full CoMRAD front ends at equal FIFO boundaries and three credit depths."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re

from check_round12_rtl import between, gap, load_fixtures, receipt, repeated_cadence, summarize, synthetic, write_mem
from synthesize_score import TOP, path_to, replay, sha256


SOURCES = tuple(TOP / "rtl" / f for f in ("round13_frontend.sv", "round12_gqa_core.sv", "round12_coeff.sv"))
BENCH = TOP / "sim/round13_frontend_tb.sv"
FIXTURES = TOP / "results/round12_frontend/local_inputs"
HARDWARE = TOP / "results/round13_comrad/hardware"
PROFILES = {f"full{dim}_s{slots}_c{credits}": (dim, slots, credits)
            for dim in (128, 96) for slots, credits in ((2, 2), (4, 2), (4, 3))}
RX_DEPTH = 16


def trace_rows(stdout):
    epochs = []
    for match in re.finditer(r"^EV (\w+) (\d+)$", stdout, re.MULTILINE):
        name, cycle = match.groups()
        if name == "begin":
            epochs.append({})
        epochs[-1].setdefault(name, []).append(int(cycle))
    out = []
    for index, epoch in enumerate(epochs):
        for field in ("record_start", "dot_first", "dot_last", "output", "complete", "coefficient", "issue"):
            assert field in epoch
        for offset, (start, complete, coef, dot, last, score) in enumerate(zip(
            epoch["record_start"], epoch["complete"], epoch["coefficient"], epoch["dot_first"],
            epoch["dot_last"], epoch["output"], strict=True
        )):
            out.append(dict(epoch=index, record=offset, epoch_first_burst=epoch["external_input"][0],
                            start=start, complete=complete, coefficient_first_issue=epoch["issue"][4*offset],
                            coefficient_ready=coef, dot_first=dot, dot_last=last, output=score))
    return epochs, out


def simulate(name, work, epochs, stall):
    dim, slots, credits = PROFILES[name]
    work.mkdir(parents=True, exist_ok=True)
    words, pages = write_mem(work, epochs, dim)
    iv, vvp = (path_to(tool, None) for tool in ("iverilog", "vvp"))
    params = dict(CHUNKS=dim//16, RECORD_SLOTS=slots, RESULT_CREDITS=credits,
                  NUM_EPOCHS=len(pages), TOTAL_PAGES=sum(pages), TOTAL_WORDS=words)
    replay([iv, "-g2012", "-s", "tb", "-o", "run.vvp",
            *[f"-Ptb.{k}={v}" for k, v in params.items()], *SOURCES, BENCH],
           work, work/"iverilog.log")
    stdout = replay([vvp, "run.vvp", f"+stall={int(stall)}"], work, work/"vvp.log")
    if f"PASS {sum(pages)}\n" not in stdout:
        raise AssertionError("RTL did not finish every epoch")
    metrics = receipt(stdout, pages, 2, False, dim//16)
    trace_epochs, trace = trace_rows(stdout)
    peak_rx = int(re.search(r"^FIFOPEAK (\d+)$", stdout, re.MULTILINE)[1])
    stalls = tuple(map(int, re.search(r"^WAITS (\d+) (\d+) (\d+)$", stdout, re.MULTILINE).groups()))
    assert metrics["observed_peak_record_slots"] <= slots
    assert metrics["observed_peak_inflight_or_buffered_results"] <= credits
    assert peak_rx <= RX_DEPTH
    assert sum(len(e["external_input"]) for e in trace_epochs) == words == metrics["accepted_input_beats"]
    assert all(min(between(e["input"], e["external_input"])) >= 1 for e in trace_epochs)
    assert stalls == tuple(sum(len(e.get(event, [])) for e in trace_epochs)
                           for event in ("blocked_slot", "wait_coefficient", "wait_credit"))
    edge = work/"cycles.csv"
    with edge.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=trace[0])
        writer.writeheader()
        writer.writerows(trace)
    metrics.update(record_slots=slots, result_fifo_slots=credits, rx_fifo_entries=RX_DEPTH,
                   observed_peak_rx_fifo_entries=peak_rx, blocked_record_slot_cycles=stalls[0],
                   waited_coefficient_cycles=stalls[1], blocked_result_credit_cycles=stalls[2],
                   rx_accept_to_assembly_cycles=summarize([value for e in trace_epochs
                                                           for value in between(e["input"], e["external_input"])]),
                   external_first_burst_to_result_cycles=[e["output"][0]-e["external_input"][0]
                                                          for e in trace_epochs],
                   log_sha256=sha256(work/"vvp.log"), cycle_trace_sha256=sha256(edge))
    if not stall:
        # The four-slot ring has extra initial credits. At record 8 it has
        # entered its periodic interval. Exclude the final, possibly odd beat.
        startup = 8
        for field in ("complete", "dot_first", "output"):
            slices = [gap(e[field])[startup:-1] for e in trace_epochs]
            cadence = repeated_cadence(slices)
            if cadence is None:
                raise AssertionError(f"Missing periodic {field} cadence: {name}")
            metrics[f"steady_{field}_cadence"] = cadence
        metrics["exact_steady_record_cadence"] = metrics["steady_complete_cadence"]
        metrics["steady_warmup_intervals"] = startup
    return metrics, edge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=PROFILES, action="append")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, default=HARDWARE)
    parser.add_argument("--work", type=Path, default=TOP/".cache/round13/rtl")
    parser.add_argument("--jobs", type=int, default=3)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    cycle_dir = opts.output/"cycles"
    cycle_dir.mkdir(exist_ok=True)
    fixtures = sorted(opts.fixtures.glob("*.npz"))

    def check(name):
        dim, slots, credits = PROFILES[name]
        tests = synthetic(dim)
        saved = load_fixtures(fixtures, dim)
        fixtures_by_name = {p.name: sha256(p) for p, _ in saved}
        design = dict(record_channels=dim, stride_bytes=dim+16, record_slots=slots,
                      result_credits=credits, rx_fifo_entries=RX_DEPTH, fixture_hashes=fixtures_by_name)
        for label, epochs, stall in (("continuous", tests, False), ("stalled", tests, True),
                                     ("real", [item for _, item in saved], True)):
            if not epochs:
                continue
            result, edges = simulate(name, opts.work/name/label, epochs, stall)
            design[label] = result
            (cycle_dir/f"{name}_{label}.csv").write_bytes(edges.read_bytes())
        print(name, design["continuous"]["exact_steady_record_cadence"],
              [design["continuous"][k] for k in ("blocked_record_slot_cycles",
                                                "waited_coefficient_cycles", "blocked_result_credit_cycles")],
              flush=True)
        return name, design

    with ThreadPoolExecutor(max_workers=opts.jobs) as workers:
        designs = dict(workers.map(check, opts.case or PROFILES))
    sources = (*SOURCES, BENCH, Path(__file__).absolute(), TOP/"scripts/check_round12_rtl.py")
    payload = dict(sources_sha256={str(p.relative_to(TOP)): sha256(p) for p in sources}, designs=designs,
                   scope="Integer scores, Q24 coefficients, record/page ID ordering, query epochs, and an internal ordered 16×32 B receive FIFO. Continuous and seeded randomized input/output pressure use the same implementation parameters.",
                   wait_buckets=dict(blocked_record_slot_cycles="a buffered beat cannot acquire its record slot or both straddling banks",
                                     waited_coefficient_cycles="the next assembled record is at the dot head and awaits its coefficient writes",
                                     blocked_result_credit_cycles="the next prepared record cannot start its first dot because every output credit is held"),
                   wait_interpretation="The ingress stall and dot stall can overlap. The two dot-head stall counters are exclusive.")
    (opts.output/"rtl_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
