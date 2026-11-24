"""Replay Round 12 synthetic and frozen calibration vectors in Icarus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import runpy
import struct

from synthesize_score import TOP, path_to, replay, sha256


RTL = TOP / "rtl"
SOURCES = (RTL / "round12_frontend.sv", RTL / "round12_gqa_core.sv",
           RTL / "round12_coeff.sv")
BENCH = TOP / "sim/round12_frontend_tb.sv"
PROFILES = {
    "old128": (128, 0, 0, True),
    "full128_dual": (128, 2, 0, False),
    "preloaded96": (96, 2, 0, True),
    "full96_dual": (96, 2, 0, False),
    "patched96_dual": (96, 2, 1, False),
    "full96_single": (96, 1, 0, False),
    "full80_dual": (80, 2, 0, False),
    "full80_single": (80, 1, 0, False),
    "full64_dual": (64, 2, 0, False),
    "full64_single": (64, 1, 0, False),
    "patched80_dual": (80, 2, 1, False),
    "patched64_dual": (64, 2, 1, False),
}
DEFAULT_PROFILES = tuple(name for name in PROFILES if name not in ("patched80_dual", "patched64_dual"))
Q24 = 1 << 24
CMAX = (1 << 22) - 1


def little(fields, width):
    mask = (1 << width) - 1
    return sum((int(value) & mask) << (i * width) for i, value in enumerate(fields))


def floats_to_bits(values):
    return [struct.unpack("<Q", struct.pack("<d", x))[0] for x in values]


def bias(count):
    return round(Q24 * math.log(max(1, count)))


def oracle(q8, g64, records, dim):
    """Independent score reference using the existing binary64 round-to-even convention."""
    scores, coefficients = [], []
    for record in records:
        a, b = struct.unpack_from("<ee", record, dim)
        count = record[dim + 4]
        page_coeffs = []
        page_scores = []
        for head, g in zip(q8, g64, strict=True):
            ca = min(CMAX, round(g * a * Q24))
            cb = min(CMAX, round(g * (b * 0.25) * Q24))
            dot_a = sum(x * (((y & 63) ^ 32) - 32) for x, y in zip(head, record[:dim], strict=True))
            dot_b = sum(abs(x) * (y >> 6) for x, y in zip(head, record[:dim], strict=True))
            page_scores.append(dot_a * ca + dot_b * cb + bias(count))
            page_coeffs.extend((ca, cb))
        scores.append(page_scores)
        coefficients.append(page_coeffs)
    return scores, coefficients


def synthetic(dim):
    rng = random.Random(120012 + dim)
    epochs = []
    for epoch, n in enumerate((91, 40)):
        q8 = [[rng.randrange(-128, 128) for _ in range(dim)] for _ in range(4)]
        q8[0][:4] = [-128, 127, 0, 1]
        g64 = [2**-9, 0.007809101475122599, 0.0, 0.052173944101]
        if epoch:
            g64 = g64[::-1]
        records = []
        for page in range(n):
            fill = [0, 0x88, 0xFF, 0x40, 0x80, 0xE0]
            codes = (bytes([fill[page]]) * dim if page < len(fill) else
                     bytes(rng.randrange(256) for _ in range(dim)))
            # Include zero/subnormal scales, saturated coefficients, padded
            # and full records, and nonzero alignment bytes.
            scales = (2**-24, 0.0) if page == 0 else (16., 16.) if page == 1 else (
                rng.randrange(1, 2048) / 32768, rng.randrange(1, 2048) / 16384)
            count = (0, 1, 31, 32)[page % 4]
            records.append(codes + struct.pack("<eeB", *scales, count) +
                           bytes(rng.randrange(256) for _ in range(11)))
        scores, coeffs = oracle(q8, g64, records, dim)
        epochs.append((q8, g64, records, scores, coeffs))
    return epochs


def load_fixtures(paths, dim):
    import numpy as np

    fixtures = []
    for path in paths:
        with np.load(path, allow_pickle=False) as fixture:
            if fixture["q8"].shape != (4, dim):
                continue
            q8 = fixture["q8"].astype(np.int8).tolist()
            g64 = fixture["g64"].astype(np.float64).tolist()
            records = [bytes(row) for row in fixture["records"].astype(np.uint8)]
            expected = fixture["scores"].astype(np.int64).tolist()
            observed, coefficients = oracle(q8, g64, records, dim)
            if observed != expected:
                raise AssertionError(f"independent Q24 oracle differs from fixture: {path}")
            if "coeff_a" in fixture and "coeff_b" in fixture:
                supplied = [
                    [y for pair in zip(a, b, strict=True) for y in pair]
                    for a, b in zip(fixture["coeff_a"], fixture["coeff_b"], strict=True)
                ]
                if supplied != coefficients:
                    raise AssertionError(f"independent coefficient oracle differs from fixture: {path}")
            fixtures.append((path, (q8, g64, records, expected, coefficients)))
    return fixtures


def write_hex(path, words, bits):
    path.write_text("".join(f"{word:0{(bits + 3) // 4}x}\n" for word in words))


def write_mem(work, epochs, dim):
    query, gains, bursts, preloads, answers, pages_per_epoch = [], [], [], [], [], []
    for q8, g64, records, scores, coeffs in epochs:
        gains.append(little(floats_to_bits(g64), 64))
        query.extend(little([q for head in q8 for q in head[i:i+16]], 8)
                     for i in range(0, dim, 16))
        stream = b"".join(records)
        stream += b"\xA5" * ((-len(stream)) % 32)
        bursts.extend(int.from_bytes(stream[i:i+32], "little") for i in range(0, len(stream), 32))
        pages_per_epoch.append(len(records))
        for record, score, coefficient in zip(records, scores, coeffs, strict=True):
            count = record[dim+4]
            preload_meta = little(coefficient, 22) | (count << 176) | ((bias(count) & 0xffffffff) << 184)
            preloads.append(int.from_bytes(record[:dim], "little") | (preload_meta << (8*dim)))
            answers.append(little(score, 48) | (count << 192))
    for name, words, bits in (("q8", query, 512), ("g64", gains, 256),
                              ("bursts", bursts, 256), ("preload", preloads, 8*dim+216),
                              ("scores", answers, 200), ("epochs", pages_per_epoch, 32)):
        write_hex(work / f"{name}.hex", words, bits)
    return len(bursts), pages_per_epoch


def summarize(values):
    if not values:
        return None
    return dict(min=min(values), max=max(values), mean=sum(values)/len(values),
                histogram={str(k): v for k, v in sorted(Counter(values).items())})


def gap(values):
    return [b - a for a, b in zip(values, values[1:])]


def between(end, start):
    return [a - b for a, b in zip(end, start, strict=True)]


def repeated_cadence(series):
    """Find an exact periodic cadence, permitting phase rotation by epoch."""
    sample = max(series, key=len)
    for width in range(1, len(sample) // 3 + 1):
        pattern = sample[:width]
        if not all(x == pattern[i % width] for i, x in enumerate(sample)):
            continue
        if all(any(all(x == pattern[(i + phase) % width] for i, x in enumerate(intervals))
                   for phase in range(width))
               for intervals in series):
            return dict(interval_pattern=pattern, effective_ii_cycles=sum(pattern)/width)
    return None


def receipt(stdout, epoch_pages, lanes, preload, chunks):
    trace = []
    for line in stdout.splitlines():
        if line.startswith("EV "):
            _, event, edge = line.split()
            if event == "begin":
                trace.append(defaultdict(list))
            trace[-1][event].append(int(edge))
    assert len(trace) == len(epoch_pages)
    peaks = re.search(r"^PEAK (\d+) (\d+) (\d+)$", stdout, re.MULTILINE)
    assert peaks
    peak_slots, peak_fifo, peak_outstanding = map(int, peaks.groups())
    raw, steady = defaultdict(list), defaultdict(list)
    steady_completions = []
    first_latencies = []
    for data, n in zip(trace, epoch_pages, strict=True):
        assert len(data["dot"]) == n * chunks
        assert len(data["complete"]) == len(data["output"]) == len(data["dot_first"]) == len(data["record_start"]) == n
        assert len(data["carry"]) == (n // 2 if not preload and chunks % 2 == 0 else 0)
        assert between(data["dot_last"], data["dot_first"]) == [chunks - 1] * n
        terms = 8 // lanes
        if not preload:
            assert len(data["coefficient"]) == n and len(data["issue"]) == n * terms
            start = data["issue"][::terms]
            end = data["issue"][terms-1::terms]
            assert all(x > 0 for x in between(start, data["complete"]))
            assert all(x > 0 for x in between(data["dot_first"], data["coefficient"]))
            assert all(x == terms - 1 for x in between(end, start))
            raw["coefficient_first_issue_to_all_ready"].extend(between(data["coefficient"], start))
            raw["coefficient_last_issue_to_all_ready"].extend(between(data["coefficient"], end))
            raw["coefficient_complete_to_first_issue"].extend(between(start, data["complete"]))
            steady["coefficient_record_issue_interval"].extend(gap(start)[2:-1])
        for event, name in (("complete", "record_completion_interval"),
                            ("dot_first", "dot_start_interval"), ("output", "output_interval")):
            intervals = gap(data[event])
            raw[name].extend(intervals)
            steady[name].extend(intervals[2:-1])
            if event == "complete":
                steady_completions.append(intervals[2:-1])
        for start, end in (("complete", "dot_first"), ("complete", "dot_last"),
                           ("complete", "output"), ("dot_last", "output"),
                           ("record_start", "complete"), ("record_start", "output")):
            raw[f"{start}_to_{end}"].extend(between(data[end], data[start]))
        first_latencies.append(dict(
            first_burst_to_complete=data["complete"][0]-data["input"][0],
            first_burst_to_first_output=data["output"][0]-data["input"][0],
            complete_to_first_output=data["output"][0]-data["complete"][0],
            first_query_load_to_first_output=data["output"][0]-data["query"][0],
        ))
    return dict(
        pages=sum(epoch_pages),
        group_heads=4,
        verified_head_scores=4*sum(epoch_pages),
        verified_query_epochs=len(epoch_pages),
        clock_edge_convention="accepted rising-edge cycle difference; all four results share one handshake",
        raw_dot_service_cycles=chunks,
        coefficient_lanes=0 if preload else lanes,
        coefficient_record_issue_cycles=0 if preload else 8 // lanes,
        record_slots=2,
        result_fifo_slots=2,
        observed_peak_record_slots=peak_slots,
        observed_peak_result_fifo_entries=peak_fifo,
        observed_peak_inflight_or_buffered_results=peak_outstanding,
        record_release="dot_last edge; slot can accept bytes on following edge",
        edge_metrics={k: summarize(v) for k, v in raw.items()},
        steady_edge_metrics={k: summarize(v) for k, v in steady.items()},
        exact_steady_record_cadence=repeated_cadence(steady_completions),
        first_record_latencies=first_latencies,
        stalled_input_cycles=sum(len(x["input_block"]) for x in trace),
        stalled_output_cycles=sum(len(x["output_block"]) for x in trace),
        accepted_input_beats=sum(len(x["input"]) for x in trace),
        accepted_boundary_shared_beats=sum(len(x["carry"]) for x in trace),
    )


def simulate(name, work, epochs, stall):
    dim, lanes, patch, preload = PROFILES[name]
    work.mkdir(parents=True, exist_ok=True)
    words, pages = write_mem(work, epochs, dim)
    iv = path_to("iverilog", None)
    vvp = path_to("vvp", None)
    params = dict(CHUNKS=dim//16, COEFF_LANES=lanes, PATCH=patch, PRELOADED=int(preload),
                  NUM_EPOCHS=len(pages), TOTAL_PAGES=sum(pages), TOTAL_WORDS=words)
    replay([iv, "-g2012", "-s", "tb", "-o", "run.vvp",
            *[f"-Ptb.{k}={v}" for k, v in params.items()], *SOURCES, BENCH],
           work, work / "iverilog.log")
    stdout = replay([vvp, "run.vvp", f"+stall={int(stall)}"], work, work / "vvp.log")
    if f"PASS {sum(pages)}\n" not in stdout:
        raise AssertionError("RTL did not finish checking every page")
    return dict(log_sha256=sha256(work / "vvp.log"),
                **receipt(stdout, pages, lanes, preload, dim//16))


def old128(work, fixture_epochs):
    work.mkdir(parents=True, exist_ok=True)
    old = runpy.run_path(str(TOP / "tests/test_score_rtl.py"))
    blocks = []
    for q8, _g64, records, scores, coeffs in fixture_epochs:
        for h, query in enumerate(q8):
            pages = [dict(codes=list(record[:128]), coeff_a=ca_cb[2*h], coeff_b=ca_cb[2*h+1],
                          bias=bias(record[132]), score=score[h])
                     for record, score, ca_cb in zip(records, scores, coeffs, strict=True)]
            blocks.append(dict(query=query, pages=pages))
    # Resolve the old script's command discovery without changing its RTL.
    import os

    os.environ["IVERILOG"] = str(path_to("iverilog", None))
    os.environ["VVP"] = str(path_to("vvp", None))
    stats = old["simulate"]("score_mr_fused", blocks, work)
    return dict(**stats, verified_head_scores=stats["pages"], native_heads=1, raw_dot_service_cycles=8,
                clock_edge_convention="difference from first accepted chunk to newly visible result, 1 cycle before handshake convention",
                ingress="128-bit code chunks plus precomputed coefficient/bias, with no burst assembler",
                output_flow_control="old interface has no backpressure input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, action="append")
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--work", type=Path, default=TOP / ".cache/round12/rtl")
    parser.add_argument("--result", type=Path, default=TOP / "results/round12_frontend/hardware/rtl_validation.json")
    args = parser.parse_args()
    profiles = args.profile or DEFAULT_PROFILES
    fixtures = sorted(args.fixture_dir.glob("*.npz")) if args.fixture_dir else []
    if args.fixture_dir and not fixtures:
        parser.error("no NPZ fixtures found")
    rows = {}
    for name in profiles:
        dim, _lanes, _patch, preload = PROFILES[name]
        epochs = synthetic(dim)
        local = load_fixtures(fixtures, dim) if fixtures else []
        row = dict(record_channels=dim, stride_bytes=dim+16, preloaded_coefficients=preload)
        if name == "old128":
            row["continuous"] = old128(args.work / name / "continuous", epochs)
            if local:
                row["calibration"] = old128(args.work / name / "calibration", [b for _, b in local])
        else:
            row["continuous"] = simulate(name, args.work / name / "continuous", epochs, False)
            row["stalled"] = simulate(name, args.work / name / "stalled", epochs, True)
            if local:
                calibration = [b for _, b in local]
                row["calibration"] = simulate(name, args.work / name / "calibration",
                                              calibration, True)
        row["calibration_fixtures"] = [
            dict(file=path.name, sha256=sha256(path)) for path, _ in local
        ]
        rows[name] = row
        print(name, row["continuous"].get("steady_edge_metrics", {}).get(
            "record_completion_interval", row["continuous"].get("ii_max")), flush=True)
    sources = [*SOURCES, BENCH, RTL / "score_datapaths.sv", Path(__file__)]
    output = dict(stimulus_seed="120012 + DIM deterministic Python PRNG",
                  hardware_storage=dict(
                      summary_ingress="ordered pre-reassembled DRAM burst stream, no addressed reorder FIFO or CDC",
                      burst_bytes=32, bank_write_bytes=32, bank_indexed_read_bytes=16,
                      boundary_beat="one 16-byte write to each bank, no separate duplicated transfer",
                      record_slots=2, record_bank_capacity="channels + 16 bytes per bank",
                      record_bank_ports="one aggregate 32 B beat per cycle; one independent selected 16 B chunk read; trailing fields held directly",
                      coefficient_storage="8 × 22 bits per bank; one/two words written on coefficient result per cycle",
                      query_storage="CHUNKS × 512 BITS raw plus the same for unsigned magnitude; four 64-bit g factors",
                      result_storage="two entries, 192-bit group Q24 score + 32-bit tag + 8-bit count; 1R1W, credits include the dot/score pipeline",
                      length_bias="Q24 ln(count), micro-ROM for counts 1–32. Count 0 is masked; counts 33–255 are out of contract.",
                  ),
                  sources_sha256={str(p.relative_to(TOP)): sha256(p) for p in sources if p.exists()},
                  designs=rows)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
