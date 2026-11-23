"""Compare the scorer RTL against an integer oracle with a streamed page input."""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


RTL = Path(__file__).resolve().parents[1] / "rtl" / "score_datapaths.sv"
TOPS = {
    "score_quest": (True, 8, 9),
    "score_mr_serial": (False, 16, 17),
    "score_mr_fused": (False, 8, 9),
    "score_mr_serial_equal": (False, 8, 17),
}


def record(query, codes, a, b, bias, quest):
    def signed(value, bits):
        return value - (1 << bits) if value >= (1 << (bits - 1)) else value

    if quest:
        x = sum(q * signed(c & 15, 4) for q, c in zip(query, codes) if q < 0)
        y = sum(q * signed(c >> 4, 4) for q, c in zip(query, codes) if q >= 0)
    else:
        x = sum(q * signed(c & 63, 6) for q, c in zip(query, codes))
        y = sum(abs(q) * (c >> 6) for q, c in zip(query, codes))
    return dict(codes=codes, coeff_a=a, coeff_b=b, bias=bias, score=x * a + y * b + bias)


def make_vectors(quest):
    rng = random.Random(512 if quest else 615)
    blocks = []
    for query in ([-128] * 128, [127, -128, 0, 1] * 32,
                  [rng.randint(-128, 127) for _ in range(128)]):
        pages = [record(query, [c] * 128, (1 << 22) - 1, (1 << 22) - 2, -(1 << 31), quest)
                 for c in (0, 0x88, 0xFF, 0x40, 0x80, 0xE0)]
        for _ in range(8):
            codes = [rng.randrange(256) for _ in range(128)]
            pages.append(record(query, codes, rng.randrange(1 << 22), rng.randrange(1 << 22),
                                rng.randrange(-(1 << 31), 1 << 31), quest))
        blocks.append(dict(query=query, pages=pages))
    return blocks


def pack_bytes(values):
    return sum((v & 255) << (8 * i) for i, v in enumerate(values))


def make_testbench(top, blocks, directory, vcd=None):
    """Write input and expected-output memories, and return the SV bench text."""
    expected = []
    commands = []
    for block in blocks:
        query = block["query"]
        assert len(query) == 128 and all(-128 <= v <= 127 for v in query)
        for j in range(8):
            commands.append((1 << 208) | (j << 204) | pack_bytes(query[16*j:16*(j+1)]))
        for page in block["pages"]:
            codes = page["codes"]
            assert len(codes) == 128 and all(0 <= v <= 255 for v in codes)
            a, b, bias = page["coeff_a"], page["coeff_b"], page["bias"]
            assert 0 <= a < (1 << 22) and 0 <= b < (1 << 22) and -(1 << 31) <= bias < (1 << 31)
            expected.append(page["score"] & ((1 << 48) - 1))
            for j in range(8):
                commands.append(((j == 0) << 207) | ((bias & 0xffffffff) << 172) |
                                (b << 150) | (a << 128) | pack_bytes(codes[16*j:16*(j+1)]))
        commands.append((1 << 209) | len(expected))
    assert expected
    (directory / "input.hex").write_text("".join(f"{c:053x}\n" for c in commands))
    (directory / "output.hex").write_text("".join(f"{s:012x}\n" for s in expected))
    wave = f'$dumpfile("{vcd.resolve()}"); $dumpvars(0, dut);' if vcd else ""
    count = len(commands)
    scores = len(expected)
    return f"""
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    reg query_load = 0;
    reg [2:0] query_chunk = 0;
    reg [127:0] query_data = 0;
    reg page_valid = 0;
    wire page_ready;
    reg page_first = 0;
    reg [127:0] page_codes = 0;
    reg [21:0] coeff_a = 0, coeff_b = 0;
    reg signed [31:0] bias = 0;
    wire score_valid;
    wire signed [47:0] score;
    reg [209:0] commands [0:{count-1}];
    reg [47:0] expected [0:{scores-1}];
    integer first_cycles [0:{scores-1}];
    integer i, cycle = 0, got = 0, accepted = 0, interval_min = 1000, interval_max = 0;
    integer latency_min = 1000, latency_max = 0, last_start = 0, gap, latency;
    reg next_block = 1;

    {top} dut (.*);

    always @(posedge clk) begin
        cycle = cycle + 1;
        if (rst_n && page_valid && page_ready && page_first) begin
            first_cycles[accepted] = cycle;
            accepted = accepted + 1;
            if (!next_block) begin
                gap = cycle - last_start;
                if (gap < interval_min) interval_min = gap;
                if (gap > interval_max) interval_max = gap;
            end
            next_block = 0;
            last_start = cycle;
        end
        #1;
        if (rst_n && score_valid) begin
            if (got >= {scores})
                $fatal(1, "unexpected output %h", score);
            if (score !== expected[got])
                $fatal(1, "page %0d expected %h got %h", got, expected[got], score);
            latency = cycle - first_cycles[got];
            if (latency < latency_min) latency_min = latency;
            if (latency > latency_max) latency_max = latency;
            got = got + 1;
        end
        if (cycle > {count*20+100})
            $fatal(1, "watchdog, got %0d pages", got);
    end

    initial begin
        $readmemh("input.hex", commands);
        $readmemh("output.hex", expected);
        {wave}
        repeat (2) @(negedge clk);
        rst_n = 1;
        for (i=0; i<{count}; i=i+1) begin
            @(negedge clk);
            query_load = 0;
            page_valid = 0;
            if (commands[i][209]) begin
                while (got < commands[i][31:0]) @(negedge clk);
                next_block = 1;
            end else if (commands[i][208]) begin
                query_load = 1;
                query_chunk = commands[i][206:204];
                query_data = commands[i][127:0];
            end else begin
                page_valid = 1;
                page_first = commands[i][207];
                page_codes = commands[i][127:0];
                coeff_a = commands[i][149:128];
                coeff_b = commands[i][171:150];
                bias = commands[i][203:172];
                @(posedge clk);
                while (!page_ready) @(posedge clk);
            end
        end
        repeat (2) @(negedge clk);
        if (got != {scores} || accepted != {scores})
            $fatal(1, "wrong number of scores, %0d accepted and %0d produced", accepted, got);
        $display("SCORE_STATS pages=%0d ii_min=%0d ii_max=%0d latency_min=%0d latency_max=%0d",
                 got, interval_min, interval_max, latency_min, latency_max);
        $finish;
    end
endmodule
"""


def simulate(top, blocks, directory, *, vcd=None, rtl=None):
    rtl = (RTL,) if rtl is None else rtl
    for path in rtl:
        assert path.is_file(), f"missing RTL source: {path}"
    iv = os.environ.get("IVERILOG") or shutil.which("iverilog")
    vvp = os.environ.get("VVP") or shutil.which("vvp")
    if not iv or not vvp:
        pytest.skip("Icarus Verilog iverilog/vvp unavailable")
    bench = directory / "tb.sv"
    bench.write_text(make_testbench(top, blocks, directory, vcd))
    build = subprocess.run([iv, "-g2012", "-s", "tb", "-o", "scorer.vvp",
                            *[str(x.resolve()) for x in rtl], str(bench)],
                           cwd=directory, text=True, capture_output=True, check=False)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([vvp, "scorer.vvp"], cwd=directory, text=True,
                         capture_output=True, check=False, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    line = re.search(r"SCORE_STATS (.+)", run.stdout)
    assert line, run.stdout
    return {k: int(v) for k, v in (pair.split("=") for pair in line.group(1).split())}


@pytest.mark.parametrize("top", sorted(TOPS))
def test_streamed_integers_and_units_per_cycle(top, tmp_path):
    quest, ii, latency = TOPS[top]
    rows = make_vectors(quest)
    stats = simulate(top, rows, tmp_path)
    assert stats == dict(pages=sum(len(b["pages"]) for b in rows), ii_min=ii, ii_max=ii,
                         latency_min=latency, latency_max=latency)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quest-vectors", type=Path)
    p.add_argument("--mr-vectors", type=Path)
    p.add_argument("--top", action="append", choices=TOPS)
    p.add_argument("--rtl", action="append", type=Path, help="Optional mapped sources")
    p.add_argument("--vcd-dir", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    files = {True: args.quest_vectors, False: args.mr_vectors}
    if args.vcd_dir:
        args.vcd_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for top in args.top or TOPS:
        quest, _, _ = TOPS[top]
        vectors = files[quest]
        blocks = json.loads(vectors.read_text()) if vectors else make_vectors(quest)
        with tempfile.TemporaryDirectory() as temporary:
            vcd = args.vcd_dir / f"{top}.vcd" if args.vcd_dir else None
            stats = simulate(top, blocks, Path(temporary), rtl=args.rtl, vcd=vcd)
        out.append(dict(top=top, **stats))
    encoded = json.dumps(out, indent=2) + "\n"
    if args.report:
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
