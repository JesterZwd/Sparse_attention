"""Replace the assumed score-dot wait with verified RTL service cycles."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import mean

from gqa_probe.page_read_budget import page_history
from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DATA
from scripts.run_fused_scorer import ROOT
from scripts.run_ruler_round3 import load_rows


MODEL_CK_NS=1.25
OBSERVED={ "q4": ("score_quest",),
           "mr25": ("score_mr_serial","score_mr_serial_equal","score_mr_fused")}


def update_wait(old_ck,hpages,passes,period,ii,latency,frontend):
    previous=8*hpages*passes
    rtl_window=(8+1+(hpages-1)*ii+latency+frontend*hpages)*period/MODEL_CK_NS
    return old_ck-previous+ceil(rtl_window)


def load_timing(path):
    with gzip.open(path,"rt") as f:
        yield from (json.loads(line) for line in f)


def report(root,data):
    hw=root/"hardware"
    src=hw/"synthesis.json"
    design=json.loads(src.read_text())
    rtl=json.loads((hw/"rtl_validation.json").read_text())
    synth={p["top"]:p for p in design["designs"]}
    checked={p["top"]:p for p in rtl}
    for name,x in synth.items():
        v=checked[name]
        if v["ii_min"]!=v["ii_max"] or v["latency_min"]!=v["latency_max"]:
            raise ValueError("RTL fixture must show stable service cycles")
        if v["ii_min"]!=x["ii_cycles"] or v["latency_min"]!=x["latency_cycles"]:
            raise ValueError("synthesis and observed RTL service disagree")
    q4,mr,single,equal=(synth[x] for x in ("score_quest","score_mr_fused","score_mr_serial","score_mr_serial_equal"))
    common=max(x["critical_period_ns"] for x in (q4,mr,equal))
    common=max(common,MODEL_CK_NS)
    same_ii=(q4["ii_cycles"]==mr["ii_cycles"]==equal["ii_cycles"])
    compare=dict(same_page_ii=same_ii,common_period_ns=common,
                 matched_pages_per_second=1e9/(common*mr["ii_cycles"]),
                 fused_area_delta_vs_serial_equal=mr["area_um2"]/equal["area_um2"]-1,
                 fused_area_delta_vs_q4=mr["area_um2"]/q4["area_um2"]-1,
                 fused_period_delta_vs_q4=mr["critical_period_ns"]/q4["critical_period_ns"]-1,
                 hardware_improvement=same_ii and mr["area_um2"]<equal["area_um2"],
                 base_library_sha256=design["library_sha256"],synthesis_sha256=_sha(src),
                 timing_scope="shared feasible period from pre-layout STA")
    _write_json(hw/"comparison.json",compare)
    values=[]
    for stage in ("validation","independent"):
        folder=root/"report"/stage
        if not (folder/"timings.jsonl.gz").exists():
            continue
        split="test" if stage=="independent" else "calibration"
        _,_,dataset=load_rows(data/split,split)
        prefix={r["sample_id"]:r["input_tokens"] for r in dataset}
        policies={p["name"]:p for p in json.loads((root/stage/"config.json").read_text())["policies"]}
        scores=defaultdict(list)
        for r in load_timing(folder/"timings.jsonl.gz"):
            p=policies[r["policy"]]
            method=p["summary"]
            if method not in OBSERVED:
                continue
            h=page_history(prefix[r["sample_id"]]+r["step"]+1,32)
            old=r["banked_completion_ck"]
            for name in OBSERVED[method]:
                v=checked[name]
                # Equal clock also used for the single serial front. If it
                # exceeds the common constraint, stretch only that row.
                period=max(common,synth[name]["critical_period_ns"])
                for frontend in (0,2,8):
                    ck=update_wait(old,h.pages,2 if method=="mr25" else 1,period,
                                   v["ii_min"],v["latency_min"],frontend)
                    scores[r["policy"],name,frontend].append((old,ck))
        for (policy,name,frontend),parts in scores.items():
            values.append(dict(stage=stage,policy=policy,scorer=name,frontend_cycles_per_page=frontend,
                               states=len(parts),original_mean_ck=mean(a for a,b in parts),
                               updated_mean_ck=mean(b for a,b in parts)))
    write_csv(hw/"attention_timing_projection.csv",values)
    print(json.dumps(compare,ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    report(a.root,a.data)
