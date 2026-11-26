"""Map and time both full CoMRAD frontends with the same receive FIFO."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import subprocess

from check_round13_rtl import HARDWARE, PROFILES, RX_DEPTH, SOURCES
from synth_round12 import area_by_stage, parse_paths, read_mapped, stage_of_net
from synthesize_score import LOCAL_TOOLS, TOP, parse_area, path_to, quote, replay, sha256


NAME = "r13_frontend"


def inspect_stage(net):
    if re.search(r"\brx_(fifo|count|read|write)\b", net):
        return "receive_fifo"
    return stage_of_net(net)


def map_case(case, work, result, lib, rtl, yosys, abc, sta, period):
    dim, slots, credits = PROFILES[case]
    work.mkdir(parents=True, exist_ok=True)
    source = work/"source.sv"
    source.write_bytes(b"\n".join(p.read_bytes() for p in SOURCES))
    constr = work/"abc.constr"
    constr.write_text("set_driving_cell BUF_X1\nset_load 5.0\n")
    mapped = work/"mapped.v"
    script = work/"synth.ys"
    script.write_text(f"""read_verilog -sv {quote(source)}
chparam -set CHUNKS {dim//16} -set RECORD_SLOTS {slots} -set RESULT_CREDITS {credits} -set RX_DEPTH {RX_DEPTH} {NAME}
synth -top {NAME} -flatten -noabc
rename -top {NAME}
dfflibmap -liberty {quote(lib)}
abc -exe {quote(abc)} -liberty {quote(lib)} -D {period * 1000:.6f} -constr {quote(constr)}
clean
tee -o stat.json stat -liberty {quote(lib)} -json
write_verilog -noattr -noexpr {quote(mapped)}
""")
    replay([yosys, "-Q", "-T", "-s", script], work, work/"yosys.log")
    area = parse_area(json.loads((work/"stat.json").read_text()), NAME)
    if any(t.startswith("$") for t in area["cells_by_type"]):
        raise ValueError(f"{case} has unmapped cells")
    cells = read_mapped(mapped)
    stage_pins = defaultdict(list)
    for name, (typ, pins) in cells.items():
        if typ.startswith(("DFF","SDFF")):
            stage_pins[inspect_stage(pins["Q"])].append(name+"/D")
    stage_cmds = [f'puts "STAGE {stage}"\nreport_checks -to [get_pins {{{" ".join(pins)}}}] '
                  "-path_delay max -group_path_count 1 -digits 9" for stage, pins in stage_pins.items()]
    tcl = work/"sta.tcl"
    tcl.write_text(f"""read_liberty {quote(lib)}
read_verilog {quote(mapped)}
link_design {NAME}
create_clock -name clock -period {period} [get_ports clk]
set_input_delay -clock clock 0 [all_inputs -no_clocks]
set_output_delay -clock clock 0 [all_outputs]
set_driving_cell -lib_cell BUF_X1 [all_inputs -no_clocks]
set_load 5 [all_outputs]
set_false_path -from [get_ports rst_n]
check_setup
report_checks -path_delay max -group_path_count 3 -digits 9
report_worst_slack -max -digits 9
{chr(10).join(stage_cmds)}
puts "STAGE output_ports"
report_checks -to [all_outputs] -path_delay max -group_path_count 1 -digits 9
""")
    raw = replay([sta, "-no_init", "-exit", tcl], work, work/"sta.log")
    if any(x in raw for x in ("Error:","Warning:")):
        raise ValueError(f"Unexpected STA diagnostics for {case}")
    parts = re.split(r"^STAGE (\S+)\s*$", raw, flags=re.MULTILINE)
    slack = float(re.search(r"^worst slack\s+-?max\s+([-+0-9.eE]+)\s*$", parts[0], re.MULTILINE)[1])
    critical = period-slack
    paths = parse_paths(parts[0], cells)
    if not paths or abs(paths[0]["setup_slack_ns"]-slack)>1e-5:
        raise ValueError(f"STA slack mismatch for {case}")
    stage_paths = {key: found[0] for key, block in zip(parts[1::2], parts[2::2], strict=True)
                   if (found := parse_paths(block, cells))}
    for detail in (*paths,*stage_paths.values()):
        detail["start_stage"]=inspect_stage(detail["start_signal"])
        detail["endpoint_stage"]=inspect_stage(detail["endpoint_signal"])
    sta_file = result/"sta"/f"{case}_sta.txt"
    sta_file.write_text(raw)
    capture = rtl["designs"][case]["continuous"]
    ii = capture["steady_output_cadence"]["effective_ii_cycles"]
    ingress_ii = capture["steady_complete_cadence"]["effective_ii_cycles"]
    if ii!=ingress_ii:
        raise ValueError(f"Nonstationary record/output rate for {case}")
    first = capture["external_first_burst_to_result_cycles"][0]
    out = dict(dimension=dim, stride_bytes=dim+16, heads=4, record_slots=slots,
               result_credits=credits, receive_fifo_slots=RX_DEPTH,
               receive_payload_bytes=32*RX_DEPTH, receive_last_flag_bits=RX_DEPTH,
               record_payload_bytes=slots*(dim+16), record_coeff_tag_bits=slots*(176+32),
               score_result_bytes=29*credits, query_bits=4*dim*2*8+4*64,
               target_period_ns=period, critical_period_ns=critical, setup_wns_ns=slack,
               estimated_clock_mhz=1e3/critical, passes_target=slack>=0,
               steady_group_ii_cycles=ii, ns_per_record=ii*critical,
               group_records_million_s=1e3/(ii*critical), first_output_scoring_cycles=first,
               first_output_ns=first*critical, critical_paths=paths, stage_critical_paths=stage_paths,
               stages_by_area=area_by_stage(cells, stage_pins, lib, area["area_um2"]),
               raw_sta_sha256=sha256(sta_file), mapped_sha256=sha256(mapped),
               snapshot_sha256=sha256(source), **area)
    print(case, round(critical,6), out["ns_per_record"], area["area_um2"], paths[0]["endpoint_signal"], flush=True)
    return case,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=PROFILES, action="append")
    p.add_argument("--lib",type=Path,default=LOCAL_TOOLS/"NangateOpenCellLibrary_typical.lib")
    p.add_argument("--work",type=Path,default=TOP/".cache/round13/synth")
    p.add_argument("--output",type=Path,default=HARDWARE)
    p.add_argument("--period-ns",type=float,default=1.25)
    p.add_argument("--jobs",type=int,default=3)
    opt=p.parse_args()
    rtl_path=opt.output/"rtl_validation.json"
    rtl=json.loads(rtl_path.read_text())
    if any(sha256(TOP/name)!=digest for name,digest in rtl["sources_sha256"].items()):
        raise ValueError("Functional receipt source hashes differ")
    opt.lib=opt.lib.absolute()
    (opt.output/"sta").mkdir(parents=True,exist_ok=True)
    yosys,abc,sta=(path_to(x,None) for x in ("yosys","yosys-abc","sta"))
    def run(case):
        return map_case(case,opt.work/case,opt.output,opt.lib,rtl,yosys,abc,sta,opt.period_ns)
    with ThreadPoolExecutor(max_workers=opt.jobs) as workers:
        designs=dict(workers.map(run,opt.case or PROFILES))
    fields=("dimension","stride_bytes","record_slots","result_credits","area_um2","setup_wns_ns",
            "critical_period_ns","steady_group_ii_cycles","ns_per_record","first_output_ns","record_payload_bytes",
            "record_coeff_tag_bits","score_result_bytes")
    with (opt.output/"mapped_frontends.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=["profile",*fields])
        writer.writeheader()
        for name,x in designs.items():
            writer.writerow(dict(profile=name,**{key:x[key] for key in fields}))
    output=dict(library_sha256=sha256(opt.lib), rtl_validation_sha256=sha256(rtl_path),
                source_sha256={str(p.relative_to(TOP)):sha256(p) for p in (*SOURCES,Path(__file__).absolute())},
                tool_versions={k:subprocess.check_output(arg,text=True).strip()
                               for k,arg in (("yosys",[yosys,"-V"]),("abc",[abc,"-c","version"]),("sta",[sta,"-version"]))},
                constraints=dict(period_ns=opt.period_ns,input_drive="BUF_X1",output_load_ff=5,
                                 io_delays_ns=0,reset_false_path=True),
                scope="Nangate45 pre-layout mapped registers and control, including the ordered 16 × (256-bit data + LAST) receive FIFO. Each FIFO has one read and one write port, and each bank has a selected 16-byte read. A straddling 32-byte write can touch two banks on one edge. Clock-domain crossing and compiler SRAM macros are outside this top.",
                designs=designs)
    (opt.output/"synthesis.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
