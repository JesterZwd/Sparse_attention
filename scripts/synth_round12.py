"""Map the Round 12 scorer variants with one library and 1.25 ns constraints."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

from check_round12_rtl import PROFILES, SOURCES
from synthesize_score import LOCAL_TOOLS, TOP, parse_area, path_to, quote, replay, sha256


PRIMARY_CASES = ("old128", "preloaded96", "full96_dual", "patched96_dual")
CASES = PRIMARY_CASES + ("patched80_dual", "patched64_dual")
CASE_TOPS = {"old128": "score_mr_fused", **{x: "r12_frontend" for x in CASES[1:]}}
def stage_of_net(net):
    if "\\core." in net:
        if re.search(r"\.read_", net):
            return "dot_input"
        if re.search(r"\.product_", net):
            return "dot_products"
        if re.search(r"\.(quad_|tree_)", net):
            return "dot_reduce"
        if re.search(r"\.(accumulated_|sum_)", net):
            return "dot_accumulate"
        if re.search(r"\.(scaled_|scale_|partial_)", net):
            return "score_scale"
        if re.search(r"\.(out_valid|result|compress_)", net):
            return "score_sum"
    if re.search(r"\b(query_cache|abs_cache|g_scales)\b", net):
        return "query_load"
    if re.search(r"\.transform\.|result_word|coefficients?|issue_(pos|slot)", net):
        return "coefficient"
    if re.search(r"pieces\[|external_header|\brecord_tag\b", net):
        return "record_assembly"
    if re.search(r"result_fifo|fifo_(count|read|write)", net):
        return "result_fifo"
    if re.search(r"\b(slot_live|slot_complete|slot_prepared|half_index|write_slot|read_slot|"
                 r"dot_chunk|outstanding|epoch_open|query_loaded|stream_closed|input_tag|query_chunk)\b", net):
        return "flow_control"
    # Original one-head module's stable hierarchy names.
    if "impl.cache." in net:
        return "query_load"
    if "impl.front." in net:
        return "dot_accumulate"
    if "impl.tail." in net:
        return "score_sum" if net.rstrip().startswith(("score ", "score_valid")) else "score_scale"
    return "unclassified_register"


def read_mapped(path):
    """Index the name and signal driving every mapped FF Q/D pin."""
    cells = {}
    text = path.read_text()
    for match in re.finditer(r"^  (\w+) (\w+) \(\n(.*?)^  \);", text, re.MULTILINE | re.DOTALL):
        typ, name, ports = match.groups()
        pin_map = dict(re.findall(r"\.([A-Z0-9]+)\(([^\n]+)\)", ports))
        cells[name] = (typ, pin_map)
    if not cells:
        raise ValueError(f"No standard cells identified in {path}")
    return cells


def flop_endpoints(cells):
    stages = defaultdict(list)
    for name, (typ, pins) in cells.items():
        if typ.startswith(("DFF", "SDFF")):
            stages[stage_of_net(pins["Q"])].append(name + "/D")
    return stages


def area_by_stage(cells, endpoints, liberty, expected):
    """Attribute each mapped cell once by endpoint stage of its combinational cone."""
    liberty_text = liberty.read_text()
    cell_area = {
        cell: float(area) for cell, area in re.findall(
            r'cell\s*\(\s*(\w+)\s*\)\s*\{.*?\barea\s*:\s*([\d.]+)',
            liberty_text, re.DOTALL)
    }
    all_outputs = {"Q", "QN", "Z", "ZN", "CO", "S", "GCK", "IQ", "IQN"}
    drivers = {}
    for cell, (_typ, ports) in cells.items():
        for pin, net in ports.items():
            if pin in all_outputs:
                drivers[net] = cell
    assigned = defaultdict(set)
    for stage, pins in endpoints.items():
        stack = [pin.split("/")[0] for pin in pins]
        local = set(stack)
        while stack:
            cell = stack.pop()
            typ, ports = cells[cell]
            assigned[cell].add(stage)
            for pin, net in ports.items():
                if pin in all_outputs or pin == "CK":
                    continue
                driver = drivers.get(net)
                if driver is not None and driver not in local and not cells[driver][0].startswith(("DFF", "SDFF")):
                    stack.append(driver)
                    local.add(driver)
    area = defaultdict(float)
    count = defaultdict(int)
    for cell, (typ, _ports) in cells.items():
        owners = assigned[cell]
        stage = next(iter(owners)) if len(owners) == 1 else (
            "shared_cones" if owners else "output_or_unattributed"
        )
        area[stage] += cell_area[typ]
        count[stage] += 1
    if abs(sum(area.values()) - expected) > 0.02:
        raise ValueError(f"Area attribution total {sum(area.values())} differs from Yosys {expected}")
    return {stage: dict(area_um2=round(area[stage], 6), cells=count[stage],
                        endpoint_flops=len(endpoints.get(stage, []))) for stage in sorted(area)}


def map_case(case, work, lib, yosys, abc, period):
    top = CASE_TOPS[case]
    constraints = work / "abc.constr"
    constraints.write_text("set_driving_cell BUF_X1\nset_load 5.0\n")
    dim, lanes, patch, preload = PROFILES[case]
    if case == "old128":
        sources = [TOP / "rtl/score_datapaths.sv"]
        select = ""
    else:
        sources = SOURCES
        select = f"chparam -set CHUNKS {dim//16} -set COEFF_LANES {lanes} -set PATCH {patch} -set PRELOADED {int(preload)} {top}\n"
    hashes = {str(p.relative_to(TOP)): sha256(p) for p in sources}
    snapshot = work / "source.sv"
    snapshot.write_bytes(b"\n".join(p.read_bytes() for p in sources))
    if hashes != {str(p.relative_to(TOP)): sha256(p) for p in sources}:
        raise RuntimeError("Source changed while taking synthesis input snapshot")
    mapped = work / "mapped.v"
    script = work / "synth.ys"
    script.write_text(f"""read_verilog -sv {quote(snapshot)}
{select}synth -top {top} -flatten -noabc
rename -top {top}
dfflibmap -liberty {quote(lib)}
abc -exe {quote(abc)} -liberty {quote(lib)} -D {period * 1000:.6f} -constr {quote(constraints)}
clean
tee -o stat.json stat -liberty {quote(lib)} -json
write_verilog -noattr -noexpr {quote(mapped)}
""")
    replay([yosys, "-Q", "-T", "-s", script], work, work / "yosys.log")
    area = parse_area(json.loads((work / "stat.json").read_text()), top)
    if any(t.startswith("$") for t in area["cells_by_type"]):
        raise ValueError(f"{case} retains unmapped cells")
    return dict(top=top, dimension=dim, coefficient_lanes=0 if preload else lanes,
                heads=1 if case == "old128" else 4, mapped_sha256=sha256(mapped),
                source_snapshot_sha256=sha256(snapshot), sources_sha256=hashes, **area)


def parse_paths(text, cells):
    blocks = re.split(r"(?=^Startpoint:)", text, flags=re.MULTILINE)
    rows = []
    for b in blocks:
        start = re.search(r"^Startpoint: (\S+)", b)
        end = re.search(r"^Endpoint: (\S+)", b, re.MULTILINE)
        slack = re.search(r"^\s*([-+0-9.eE]+)\s+slack \(", b, re.MULTILINE)
        if not start or not end or not slack:
            continue
        start, end = start[1], end[1]
        start_q = cells.get(start, (None, {}))[1].get("Q", start)
        end_q = cells.get(end, (None, {}))[1].get("Q", end)
        rows.append(dict(start=start, endpoint=end, start_signal=start_q, endpoint_signal=end_q,
                         start_stage=stage_of_net(start_q), endpoint_stage=stage_of_net(end_q),
                         setup_slack_ns=float(slack[1])))
    return rows


def analyze(case, work, target, cells, stages, lib, sta, period):
    top = CASE_TOPS[case]
    timing = work / "sta.tcl"
    grouped = []
    for stage, pins in stages.items():
        grouped.append(f'puts "STAGE {stage}"\n'
                       f'report_checks -to [get_pins {{{" ".join(pins)}}}] -path_delay max -group_path_count 1 -digits 9')
    timing.write_text(f"""read_liberty {quote(lib)}
read_verilog {quote(work / "mapped.v")}
link_design {top}
create_clock -name clock -period {period} [get_ports clk]
set_input_delay -clock clock 0 [all_inputs -no_clocks]
set_output_delay -clock clock 0 [all_outputs]
set_driving_cell -lib_cell BUF_X1 [all_inputs -no_clocks]
set_load 5 [all_outputs]
set_false_path -from [get_ports rst_n]
check_setup
puts "GLOBAL"
report_checks -path_delay max -group_path_count 3 -digits 9
report_worst_slack -max -digits 9
{chr(10).join(grouped)}
puts "STAGE output_ports"
report_checks -to [all_outputs] -path_delay max -group_path_count 1 -digits 9
""")
    result = replay([sta, "-no_init", "-exit", timing], work, work / "sta.log")
    warnings = [ln for ln in result.splitlines() if "Error:" in ln or "Warning:" in ln]
    if warnings:
        raise ValueError(f"OpenSTA diagnostics: {warnings}")
    parts = re.split(r"^STAGE (\S+)\s*$", result, flags=re.MULTILINE)
    match = re.search(r"^worst slack\s+-?max\s+([-+0-9.eE]+)\s*$", parts[0], re.MULTILINE)
    if not match:
        raise ValueError(f"no timing metric in {work / 'sta.log'}")
    slack = float(match[1])
    critical = period - slack
    paths = parse_paths(parts[0], cells)
    if not paths or abs(paths[0]["setup_slack_ns"] - slack) > 1e-5:
        raise ValueError("OpenSTA path and worst slack disagree")
    groups = {k: parsed[0] for k, v in zip(parts[1::2], parts[2::2], strict=True)
              if (parsed := parse_paths(v, cells))}
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{case}_sta.txt").write_text(result)
    return dict(target_period_ns=period, setup_wns_ns=slack, critical_period_ns=critical,
                estimated_clock_mhz=1e3/critical, passes_target=slack >= 0,
                critical_paths=paths, stage_critical_paths=groups,
                raw_sta_sha256=sha256(target / f"{case}_sta.txt"))


def attach_rates(designs, measured):
    for case, row in designs.items():
        cycles = measured[case]["continuous"]
        if case == "old128":
            head_ii = cycles["ii_max"]
            latency = cycles["latency_max"]
            group_ii = 4 * head_ii
            row["rate_scope"] = "native one-head scorer replayed for four head epochs"
            row["replicated_four_head_area_um2"] = 4 * row["area_um2"]
            row["replicated_four_head_group_ii_cycles"] = head_ii
            row["replicated_four_head_group_mrecords_s_at_sta"] = 1e3 / (row["critical_period_ns"] * head_ii)
        else:
            group_ii = cycles["exact_steady_record_cadence"]["effective_ii_cycles"]
            head_ii = group_ii / 4
            latency = cycles["first_record_latencies"][0]["first_burst_to_first_output"]
            row["rate_scope"] = "one four-head record per output handshake; two record slots"
        row["measured_native_first_input_to_result_cycles"] = latency
        row["measured_group_ii_cycles"] = group_ii
        row["head_mscores_s_at_sta"] = 1e3 / (row["critical_period_ns"] * head_ii)
        row["group_mrecords_s_at_sta"] = 1e3 / (row["critical_period_ns"] * group_ii)
        if row["passes_target"]:
            row["group_mrecords_s_at_target"] = 1e3 / (row["target_period_ns"] * group_ii)
        else:
            row["group_mrecords_s_at_target"] = None


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--case", choices=CASES, action="append")
    args.add_argument("--lib", type=Path, default=LOCAL_TOOLS / "NangateOpenCellLibrary_typical.lib")
    args.add_argument("--work", type=Path, default=TOP / ".cache/round12/synth")
    args.add_argument("--result", type=Path, default=TOP / "results/round12_frontend/hardware/synthesis.json")
    args.add_argument("--period-ns", type=float, default=1.25)
    args.add_argument("--rtl-validation", type=Path)
    args.add_argument("--jobs", type=int, default=1)
    args.add_argument("--reuse-mapping", action="store_true")
    params = args.parse_args()
    params.work = params.work.absolute()
    params.lib = params.lib.absolute()
    if params.period_ns <= 0 or params.jobs < 1:
        args.error("period and job count must be positive")
    requested = params.case or PRIMARY_CASES
    yosys, abc, sta = (path_to(name, None) for name in ("yosys", "yosys-abc", "sta"))
    tools = {
        "yosys": subprocess.check_output([yosys, "-V"], text=True).strip(),
        "abc": subprocess.check_output([abc, "-c", "version"], text=True).strip(),
        "sta": subprocess.check_output([sta, "-version"], text=True).strip(),
    }
    library_hash = sha256(params.lib)

    def run(case):
        work = params.work / case
        work.mkdir(parents=True, exist_ok=True)
        mapping_path = work / "mapping_receipt.json"
        if params.reuse_mapping:
            mapped = json.loads(mapping_path.read_text())
            if (mapped["mapped_sha256"] != sha256(work / "mapped.v") or
                    mapped["library_sha256"] != library_hash or
                    mapped["mapping_period_ns"] != params.period_ns or
                    mapped["mapping_tools"] != {"yosys": tools["yosys"], "abc": tools["abc"]} or
                    mapped["sources_sha256"] != {p: sha256(TOP / p) for p in mapped["sources_sha256"]}):
                raise ValueError(f"Mapping receipt failed reuse validation: {case}")
        else:
            mapped = map_case(case, work, params.lib, yosys, abc, params.period_ns)
            mapped.update(library_sha256=library_hash, mapping_period_ns=params.period_ns,
                          mapping_tools={"yosys": tools["yosys"], "abc": tools["abc"]})
            mapping_path.write_text(json.dumps(mapped, indent=2, sort_keys=True) + "\n")
        cells = read_mapped(work / "mapped.v")
        stages = flop_endpoints(cells)
        timing = analyze(case, work, params.result.parent / "sta", cells, stages, params.lib, sta, params.period_ns)
        area = area_by_stage(cells, stages, params.lib, mapped["area_um2"])
        result = dict(**mapped, **timing, stages_by_area=area)
        print(case, f'{mapped["area_um2"]:.3f} um^2',
              f'{timing["critical_period_ns"]:.6f} ns', timing["critical_paths"][0], flush=True)
        return case, result

    with ThreadPoolExecutor(max_workers=params.jobs) as workers:
        designs = dict(workers.map(run, requested))
    if params.rtl_validation:
        validation = json.loads(params.rtl_validation.read_text())
        if any(sha256(TOP / p) != digest for p, digest in validation["sources_sha256"].items()):
            raise ValueError("RTL validation receipt sources differ from current sources")
        attach_rates(designs, validation["designs"])
    output = dict(
        library_sha256=library_hash, tools=tools, script_sha256=sha256(Path(__file__)),
        constraints=dict(period_ns=params.period_ns, input_drive="BUF_X1", output_load_ff=5.0,
                         io_delays_ns=0, reset_false_path=True),
        performance_scope="Technology-mapped pre-layout STA, without physical place/route or timed SRAM macros",
        area_stage_attribution="Mapped cells grouped by flop Q names and transitive logic cones to their FF D inputs. Shared cones and unassigned output logic remain separate.",
        rtl_validation_sha256=sha256(params.rtl_validation) if params.rtl_validation else None,
        designs=designs,
    )
    params.result.parent.mkdir(parents=True, exist_ok=True)
    params.result.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
