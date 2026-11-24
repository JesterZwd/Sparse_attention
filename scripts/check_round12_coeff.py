"""Run the Round 12 coefficient and page-length RTL against CPU references."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl" / "round12_coeff.sv"
TOOLS = ROOT / ".cache" / "round9" / "tools" / "bin"
Q24 = 1 << 24
COEFFICIENT_MAX = (1 << 22) - 1
PIPELINE_STAGES = 9


def double_bits(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def reference_coefficient(g64, scale, radius):
    gain = struct.unpack(">d", g64.to_bytes(8, "big"))[0]
    factor = struct.unpack(">e", scale.to_bytes(2, "big"))[0]
    if not math.isfinite(gain) or not math.isfinite(factor) or gain <= 0 or factor <= 0:
        return 0
    if radius:
        factor *= .25
    fixed = (gain * factor) * Q24
    return COEFFICIENT_MAX if fixed >= COEFFICIENT_MAX else round(fixed)


def reference_bias(count):
    return round(math.log(count) * Q24) if 1 <= count <= 32 else 0


def directed_vectors():
    # These binary64 products round to exact Q24 halfway values although
    # the unrounded products would produce the opposite integer decision.
    cases = [
        (0x3E5FF801FF801FF9, 0x3C01, 0, 0),
        (0x3E7FF801FF801FF9, 0x3C01, 1, 0),
        (0x3E77FA017FA017FA, 0x3C01, 0, 2),
        (0x3E83F10B37964F45, 0x3C03, 0, 2),
        (0x3FCFD83102BC9447, 0x3C05, 0, 4194302),
        (0x3FCFF800BFD00BFD, 0x3C01, 0, 4194302),
        # The product itself is a binary64 tie before the Q24 tie.
        (0x3E59852F0D8EC100, 0x3D04, 0, 0),
        (0x3EB0618618618618, 0x3D40, 0, 22),
    ]
    for radius in range(2):
        for gain, scale, expected in (
            (0., 0x3C00, 0), (1., 0, 0), (1., 0x8000, 0),
            (-1., 0x3C00, 0), (1., 0xBC00, 0),
            (math.inf, 0x3C00, 0), (math.nan, 0x3C00, 0),
            (1., 0x7C00, 0), (1., 0x7E00, 0),
            (math.ulp(0.), 0x7BFF, 0), (math.ldexp(1., -1022), 0x7BFF, 0),
            (math.ldexp(1., 1023), 0x7BFF, COEFFICIENT_MAX),
            (.5 * 4**radius, 0x0001, 0), (1.5 * 4**radius, 0x0001, 2),
            (2.5 * 4**radius, 0x0001, 2), (3.5 * 4**radius, 0x0001, 4),
            (.5 * 4**radius, 0x03FF, 512), (.5 * 4**radius, 0x0400, 512),
        ):
            cases.append((double_bits(gain), scale, radius, expected))
        for fixed in (.5, 1.5, 2.5, 3.5, COEFFICIENT_MAX - .5, COEFFICIENT_MAX + .5):
            gain = fixed * 4**radius / Q24
            for value in (math.nextafter(gain, 0.), gain, math.nextafter(gain, math.inf)):
                g64 = double_bits(value)
                cases.append((g64, 0x3C00, radius, reference_coefficient(g64, 0x3C00, radius)))
    return cases


def random_vectors(rng, pairs):
    for i in range(pairs):
        scale = rng.randrange(0x7C00)
        if i % 7 == 0:
            scale = rng.randrange(0x0400)
        if i % 4 == 0 or scale == 0:
            g64 = rng.randrange(0x7FF0000000000000)
        else:
            # Concentrate on the unbiased Q24 range across half exponents.
            factor = struct.unpack(">e", scale.to_bytes(2, "big"))[0]
            gain = math.ldexp(1. + rng.random(), rng.randrange(-4, 25)) / (Q24 * factor)
            g64 = double_bits(gain)
        for radius in range(2):
            yield g64, scale, radius, reference_coefficient(g64, scale, radius)


def scorer_vectors(seed):
    """Sample gains and half scales from the existing MR4/MR96 CPU ABI."""
    import torch

    from gqa_probe import compact_summary, scorer_int
    from gqa_probe.vskip_summary import unpack_codes

    rng = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(1, 8, 128, generator=rng)
    query[0, 0] = 0
    q_scale = query.abs().amax(-1, keepdim=True) / 127
    pages = 41
    counts = torch.cat((torch.zeros(1), torch.arange(1, 33), torch.arange(1, 9)))
    counts = counts.to(torch.uint8).reshape(1, pages, 1)
    center = torch.randn(1, pages, 128, generator=rng)
    extent = torch.rand(1, pages, 128, generator=rng) * 2
    channels = torch.arange(96).reshape(1, -1)
    scaler = math.sqrt(.5) / 8
    for method in ("mr4", "mr96_k"):
        record = compact_summary.encode_summary(center * counts, center - extent,
                                                center + extent, counts, method,
                                                channels=channels)
        # Express the compact record using the public width-128 integer scorer.
        # Zero codes preserve the full query gain in the MR96 comparison.
        if method == "mr4":
            fields = unpack_codes(record[0], 6).view(torch.uint8)
            mean = fields & 15
            codes = ((fields & 48) << 2) | mean | ((mean & 8) != 0).to(torch.uint8) * 48
        else:
            codes = torch.nn.functional.pad(record[0], (0, 32))
        oracle = scorer_int.prepare_inputs(query, (codes, *record[1:]), "mr25", scaler)
        compact = compact_summary.scores(query, record, method, scaler, channels=channels)
        torch.testing.assert_close(compact, scorer_int.fixed_scores(oracle), rtol=0, atol=0)

        for head, gain in enumerate((q_scale.double() * scaler)[0, :, 0]):
            g64 = double_bits(float(gain))
            for radius, scales in enumerate(record[1:3]):
                expected = (oracle.coeff_a, oracle.coeff_b)[radius][0, head]
                for bits, value in zip(scales.view(torch.int16).reshape(-1), expected):
                    sample = (g64, int(bits) & 0xFFFF, radius, int(value))
                    assert reference_coefficient(*sample[:3]) == sample[3]
                    yield sample
        for count, bias in zip(counts.reshape(-1), oracle.bias.reshape(-1)):
            assert reference_bias(int(count)) == int(bias)


def tool(name):
    explicit = os.environ.get(name.upper())
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())
    candidate = shutil.which(explicit or name)
    if candidate:
        return candidate
    fallback = TOOLS / name
    return str(fallback) if fallback.is_file() else None


def simulate(vectors, directory):
    assert RTL.is_file(), f"missing Round 12 coefficient RTL: {RTL}"
    iv, vvp = tool("iverilog"), tool("vvp")
    if not iv or not vvp:
        raise RuntimeError("Icarus Verilog iverilog/vvp unavailable")

    idle = (1, 0, 0, 0, 0, 0)
    reset = (0, 0, 0, 0, 0, 0)
    commands = [reset] * 2
    commands.extend((1, 1, *row) for row in vectors[:5])
    commands.append(reset)
    for i, row in enumerate(vectors):
        if i % 29 == 0:
            commands.append(idle)
        commands.append((1, 1, *row))
    commands.extend([idle] * PIPELINE_STAGES)

    delay = [None] * (PIPELINE_STAGES - 1)
    encoded = []
    for rst, valid, gain, scale, radius, expected in commands:
        if not rst:
            delay = [None] * (PIPELINE_STAGES - 1)
        want = delay.pop(0)
        delay.append(expected if valid and rst else None)
        word = (rst << 105) | (valid << 104) | (gain << 40) | (scale << 24) | (radius << 23)
        if want is not None:
            word |= (1 << 22) | want
        encoded.append(f"{word:027x}\n")
    payload = "".join(encoded).encode()
    (directory / "vectors.hex").write_bytes(payload)
    bias_payload = "".join(f"{reference_bias(n):08x}\n" for n in range(256)).encode()
    (directory / "bias.hex").write_bytes(bias_payload)
    bench = directory / "tb.sv"
    bench.write_text(f"""
`timescale 1ns/1ps
module tb;
    reg clk = 0, rst_n = 0, in_valid = 0, radius = 0;
    reg [63:0] g64 = 0;
    reg [15:0] scale = 0;
    wire out_valid;
    wire [21:0] coeff;
    reg [7:0] count;
    wire signed [31:0] bias;
    reg [105:0] commands [0:{len(commands) - 1}];
    reg [31:0] expected_bias [0:255];
    reg expected_valid;
    reg [21:0] expected_coeff;
    integer i, compared = 0;

    always #5 clk = ~clk;
    r12_q24_coeff dut (.*);
    r12_length_bias bias_dut (.*);

    initial begin
        $readmemh("vectors.hex", commands);
        $readmemh("bias.hex", expected_bias);
        for (i=0; i<{len(commands)}; i=i+1) begin
            @(negedge clk);
            {{rst_n, in_valid, g64, scale, radius, expected_valid, expected_coeff}} = commands[i];
            @(posedge clk);
            #1;
            if (out_valid !== expected_valid)
                $fatal(1, "cycle %0d: expected valid %b, got %b", i, expected_valid, out_valid);
            if (expected_valid) begin
                if (coeff !== expected_coeff)
                    $fatal(1, "cycle %0d: expected coeff %0d, got %0d", i, expected_coeff, coeff);
                compared = compared + 1;
            end
        end
        for (i=0; i<256; i=i+1) begin
            count = i;
            #1;
            if (bias !== expected_bias[i])
                $fatal(1, "length %0d: expected bias %0d, got %0d", i, expected_bias[i], bias);
        end
        $display("COEFF_STATS compared=%0d bias_codes=256", compared);
        $finish;
    end
endmodule
""")
    subprocess.run([iv, "-g2012", "-s", "tb", "-o", "coeff.vvp", str(RTL), str(bench)],
                   cwd=directory, check=True)
    process = subprocess.run([vvp, "coeff.vvp"], cwd=directory, capture_output=True, text=True,
                             check=False, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    match = re.search(r"COEFF_STATS compared=(\d+) bias_codes=(\d+)", process.stdout)
    assert match and int(match[1]) == len(vectors), process.stdout
    return {"coefficients_compared": int(match[1]), "bias_codes_compared": int(match[2]),
            "pipeline_stages": PIPELINE_STAGES, "accepted_to_visible_edges": PIPELINE_STAGES - 1,
            "initiation_interval_cycles": 1, "vectors_sha256": sha256(payload).hexdigest(),
            "bias_vectors_sha256": sha256(bias_payload).hexdigest(),
            "rtl_sha256": sha256(RTL.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12024)
    parser.add_argument("--random-pairs", type=int, default=4096)
    parser.add_argument("--output", type=Path, help="Write the verification receipt as JSON")
    args = parser.parse_args()
    if args.random_pairs < 0:
        parser.error("--random-pairs must be nonnegative")

    directed = directed_vectors()
    randomized = list(random_vectors(random.Random(args.seed), args.random_pairs))
    scorer = list(scorer_vectors(args.seed))
    with tempfile.TemporaryDirectory() as temporary:
        report = simulate(directed + randomized + scorer, Path(temporary))
    report.update(seed=args.seed, directed_vectors=len(directed), random_vectors=len(randomized),
                  cpu_scorer_vectors=len(scorer))
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
