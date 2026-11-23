"""Map the four small score datapaths to a common standard-cell library."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import TextIO


# Each top processes one group head. Commands and stimulus format are shared.
DESIGNS = {
    "score_quest": (8, 9, 16),
    "score_mr_serial": (16, 17, 16),
    "score_mr_fused": (8, 9, 16),
    "score_mr_serial_equal": (8, 17, 32),
}
CLOCK_PORT = "clk"
EXCLUDED_INPUTS = {
    CLOCK_PORT, "rst_n", "query_load", "query_chunk", "query_data",
    "page_valid", "page_first", "page_codes", "coeff_a", "coeff_b", "bias",
}
TOP = Path(__file__).resolve().parents[1]
LOCAL_TOOLS = TOP / ".cache/round9/tools"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_area(values: dict, top: str) -> dict:
    module = values["modules"]["\\" + top]
    cells = module["num_cells_by_type"]
    if module["area"] <= 0 or not cells:
        raise ValueError(f"{top} has no mapped area")
    return dict(area_um2=module["area"], cell_count=module["num_cells"],
                flop_bits=sum(n for t, n in cells.items() if t.startswith(("DFF", "SDFF"))),
                latch_bits=sum(n for t, n in cells.items() if t.startswith(("DLH", "DLL"))),
                cells_by_type=cells)


def timing_metrics(period: float, slack: float, ii: int, latency: int) -> dict:
    critical = period - slack
    if period <= 0 or critical <= 0 or ii <= 0:
        raise ValueError("invalid clock or initiation interval")
    meets = slack >= 0
    return dict(target_period_ns=period, setup_wns_ns=slack, critical_period_ns=critical,
                pages_per_cycle=1 / ii, ii_cycles=ii, latency_cycles=latency,
                passes_target=meets, target_rate_mpage_s=1e3 / (period * ii) if meets else 0,
                critical_rate_mpage_s=1e3 / (critical * ii))


def vcd_activity(stream: TextIO, scope: tuple[str, ...] = ("tb", "dut"), *,
                 include_children: bool = True) -> dict:
    """Count only observed 0↔1 bit transitions, including score outputs."""
    path, widths, excluded = [], {}, set()
    score_valid = None
    for line in stream:
        if line.startswith("$enddefinitions") or line.strip().startswith("$enddefinitions"):
            break
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "$scope":
            path.append(fields[2])
        elif fields[0] == "$upscope":
            path.pop()
        elif (fields[0] == "$var" and tuple(path[:len(scope)]) == scope
              and (include_children or len(path) == len(scope))):
            width, symbol, name = int(fields[2]), fields[3], fields[4]
            widths[symbol] = width
            if len(path) == len(scope):
                if name in EXCLUDED_INPUTS:
                    excluded.add(symbol)
                if name == "score_valid":
                    score_valid = symbol
    for symbol in excluded:
        widths.pop(symbol, None)
    if not widths or score_valid not in widths:
        raise ValueError("VCD lacks requested DUT scope or score_valid")
    previous = {}
    pages = toggles = 0
    known = str.maketrans("01xz", "1100")
    binary = str.maketrans("xz", "00")
    for line in stream:
        match line.strip().lower()[:1]:
            case "b":
                bits, symbol = line.strip()[1:].split()
                bits = bits.lower()
            case "0" | "1" | "x" | "z":
                bits, symbol = line.strip()[0].lower(), line.strip()[1:]
            case _:
                continue
        width = widths.get(symbol)
        if width is None:
            continue
        if len(bits) < width:
            bits = bits.rjust(width, bits[0] if bits[0] in "xz" else "0")
        elif len(bits) != width:
            raise ValueError(f"VCD width mismatch: {symbol}")
        mask, data = int(bits.translate(known), 2), int(bits.translate(binary), 2)
        old = previous.get(symbol)
        if old is not None:
            toggles += ((old[1] ^ data) & old[0] & mask).bit_count()
            pages += symbol == score_valid and old == (1, 0) and (mask, data) == (1, 1)
        previous[symbol] = mask, data
    if pages == 0:
        raise ValueError("VCD contains no completed scores")
    return dict(observed_nets=len(widths), observed_bits=sum(widths.values()),
                known_bit_toggles=toggles, scored_pages=pages, toggles_per_page=toggles / pages)


def path_to(tool: str, explicit: Path | None) -> Path:
    candidate = explicit or (LOCAL_TOOLS / "bin" / tool)
    if candidate.is_file():
        return candidate.absolute()
    if not explicit and (path := shutil.which(tool)):
        return Path(path).absolute()
    raise FileNotFoundError(f"missing {tool}: {candidate}")


def replay(cmd: list, work: Path, log: Path) -> str:
    process = subprocess.run([str(x) for x in cmd], cwd=work, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f"{cmd[0]} exited {process.returncode}; see {log}")
    return process.stdout


def quote(path: Path) -> str:
    value = str(path.absolute())
    if any(c in value for c in ('"', "\\", "$", "[", "]", "\n")):
        raise ValueError(f"path cannot be quoted in Yosys/Tcl: {path}")
    return f'"{value}"'


def map_design(top: str, work: Path, rtl: Path, lib: Path, yosys: Path, abc: Path,
               period: float) -> dict:
    # ABC's constraint units are ps and fF for this library. The same
    # constraint, output load and input drive are applied to all four tops.
    constraints = work / "abc.constr"
    constraints.write_text("set_driving_cell BUF_X1\nset_load 5.0\n")
    area = work / "stat.json"
    netlist = work / "mapped.v"
    script = work / "synth.ys"
    script.write_text(f"""read_verilog -sv {quote(rtl)}
synth -top {top} -flatten -noabc
dfflibmap -liberty {quote(lib)}
abc -exe {quote(abc)} -liberty {quote(lib)} -D {period * 1000:.6f} -constr {quote(constraints)}
clean
tee -o stat.json stat -liberty {quote(lib)} -json
write_verilog -noattr -noexpr {quote(netlist)}
""")
    replay([yosys, "-Q", "-T", "-s", script], work, work / "yosys.log")
    mapped = parse_area(json.loads(area.read_text()), top)
    if any(t.startswith("$") for t in mapped["cells_by_type"]):
        raise ValueError(f"{top} retains unmapped primitives")
    ii, latency, lanes = DESIGNS[top]
    return dict(top=top, physical_channel_lanes=lanes, mapped_sha256=sha256(netlist),
                ii_cycles=ii, latency_cycles=latency, pages_per_cycle=1 / ii, **mapped)


def analyze_timing(top: str, work: Path, lib: Path, sta: Path, period: float) -> dict:
    netlist = work / "mapped.v"
    timing = work / "sta.tcl"
    timing.write_text(f"""read_liberty {quote(lib)}
read_verilog {quote(netlist)}
link_design {top}
create_clock -name clock -period {period} [get_ports clk]
set_input_delay -clock clock 0 [all_inputs -no_clocks]
set_output_delay -clock clock 0 [all_outputs]
set_driving_cell -lib_cell BUF_X1 [all_inputs -no_clocks]
set_load 5 [all_outputs]
set_false_path -from [get_ports rst_n]
check_setup
report_checks -path_delay max -group_path_count 3 -digits 9
report_worst_slack -max -digits 9
""")
    result = replay([sta, "-no_init", "-exit", timing], work, work / "sta.log")
    warnings = [ln for ln in result.splitlines() if "Error:" in ln or "Warning:" in ln]
    if warnings:
        raise ValueError(f"unexpected OpenSTA diagnostics: {warnings}")
    match = re.search(r"^worst slack\s+-?max\s+([-+0-9.eE]+)\s*$", result, re.MULTILINE)
    if not match:
        raise ValueError(f"no timing metric in {work / 'sta.log'}")
    ii, latency, _ = DESIGNS[top]
    return timing_metrics(period, float(match[1]), ii, latency)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--rtl", type=Path, default=TOP / "rtl/score_datapaths.sv")
    args.add_argument("--lib", type=Path, default=LOCAL_TOOLS / "NangateOpenCellLibrary_typical.lib")
    args.add_argument("--yosys", type=Path)
    args.add_argument("--abc", type=Path)
    args.add_argument("--sta", type=Path)
    args.add_argument("--work", type=Path, default=TOP / ".cache/round9/synth")
    args.add_argument("--result", type=Path, required=True)
    args.add_argument("--period-ns", type=float, default=1.25)
    args.add_argument("--top", choices=DESIGNS, action="append")
    args.add_argument("--activity-dir", type=Path)
    args.add_argument("--activity-stage", choices=("rtl", "mapped_gate"), default="rtl")
    args.add_argument("--activity-scope", default="tb.dut")
    args.add_argument("--mapping-only", action="store_true")
    args.add_argument("--reuse-mapping", type=Path, help="Reuse a previously verified mapping receipt")
    args.add_argument("--jobs", type=int, default=1)
    parsed = args.parse_args()
    if parsed.period_ns <= 0 or parsed.jobs < 1:
        args.error("--period-ns and --jobs must be positive")
    requested = parsed.top or list(DESIGNS)
    yosys = path_to("yosys", parsed.yosys)
    abc = path_to("yosys-abc", parsed.abc)
    sta = path_to("sta", parsed.sta) if not parsed.mapping_only else None
    work = parsed.work.absolute()
    work.mkdir(parents=True, exist_ok=True)
    commands = [("yosys", [yosys, "-V"]), ("abc", [abc, "-c", "version"])]
    if sta:
        commands.append(("sta", [sta, "-version"]))
    versions = {tool: subprocess.check_output(cmd, text=True).strip() for tool, cmd in commands}
    library_hash, rtl_hash = sha256(parsed.lib), sha256(parsed.rtl)
    source = work / "scorer_rtl.sv"
    if not parsed.reuse_mapping:
        source.write_bytes(parsed.rtl.read_bytes())
        if sha256(source) != rtl_hash:
            raise ValueError("RTL changed while taking the map input snapshot")
    reused = {}
    if parsed.reuse_mapping:
        receipt = json.loads(parsed.reuse_mapping.read_text())
        if (receipt["library_sha256"] != library_hash or receipt["rtl_sha256"] != rtl_hash
                or receipt["constraints"]["period_ns"] != parsed.period_ns
                or any(receipt["tools"][t] != versions[t] for t in ("yosys", "abc"))):
            raise ValueError("reuse receipt source, library, timing target, or synthesis tools differ")
        reused = {r["top"]: r for r in receipt["designs"]}

    def run_top(top):
        primary = work / top
        primary.mkdir(parents=True, exist_ok=True)
        if reused:
            result = reused[top].copy()
            if result["mapped_sha256"] != sha256(primary / "mapped.v"):
                raise ValueError(f"cached map checksum mismatch: {top}")
        else:
            result = map_design(top, primary, source, parsed.lib, yosys, abc, parsed.period_ns)
        if sta:
            result.update(analyze_timing(top, primary, parsed.lib, sta, parsed.period_ns))
        if parsed.activity_dir:
            vcd = parsed.activity_dir / f"{top}.vcd"
            with vcd.open() as f:
                flat = parsed.activity_stage == "mapped_gate"
                result["switching_activity"] = dict(
                    stage=parsed.activity_stage, vcd_sha256=sha256(vcd), flat_scope=flat,
                    **vcd_activity(f, tuple(parsed.activity_scope.split(".")), include_children=not flat))
        timing = f"{result['critical_period_ns']:.6f} ns" if sta else "timing pending"
        print(top, f"{result['area_um2']:.3f} um^2", timing, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=parsed.jobs) as pool:
        designs = list(pool.map(run_top, requested))
    common = max(x["critical_period_ns"] for x in designs) if sta else None
    if common:
        for row in designs:
            row["common_rate_mpage_s"] = 1e3 / (common * row["ii_cycles"])
    output = dict(
        library_sha256=library_hash, rtl_sha256=rtl_hash,
        tools=versions, performance_scope="one group head; synthesis without physical placement",
        constraints=dict(period_ns=parsed.period_ns, input_drive="BUF_X1", output_load_ff=5.0,
                         assumed_io_delay_ns=0, reset_false_path=True),
        common_critical_period_ns=common,
        designs=designs,
    )
    target = parsed.result
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
