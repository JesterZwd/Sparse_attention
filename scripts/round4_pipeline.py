"""Pre-test decision and detached execution of the Round 4 cohorts."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys

from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.vskip_attention import VPolicy
from scripts.report_round4_fixed_k_v import choose_workpoint, summarize, compare_test
from scripts.run_ruler_round3 import load_rows


def call(module,*args):
    subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],check=True)


def collect(root,name):
    call("scripts.run_round4_fixed_k_v",name,"--root",root)
    return summarize(root/name,root/"report"/name)


def freeze(root,point):
    cohort=point["cohort"]
    budget=point["percent"]
    index=json.loads((root/"report"/cohort/"summary.json").read_text())
    quality={r["policy"]:r for r in index["quality"]}
    def item(method,mode="all",threshold=0,**kw):
        return VPolicy(method,percent=budget,mode=mode,threshold=threshold,**kw)
    q8=item("q8")
    static=max((item("q4"),item("q6")), key=lambda p: quality[p.name]["score"])
    picked_sm=max((p for p in quality.values() if p["mode"]=="mass" and p["percent"]==budget),
                  key=lambda row:(row["score"]>=quality[q8.name]["score"]-.01,row["score"] if row["score"]<quality[q8.name]["score"]-.01 else 0,row["threshold"]))
    mass_ep=picked_sm["threshold"]
    def match(mode):
        alts=[p for p in quality.values() if p["mode"]==mode and p["percent"]==budget]
        return min(alts,key=lambda p:(p["score"]<quality[q8.name]["score"]-.01,
                                      abs(p["read_per_decode"]-picked_sm["read_per_decode"]),-p["score"]))["threshold"]
    mass=item("q8","mass",mass_ep)
    other_mass=item(static.scorer,"mass",mass_ep)
    blasst=item("q8","blasst",match("blasst"))
    sparta=item("q8","sparta",match("sparta"))
    refill=item(static.scorer,"mass",mass_ep,refill=True)
    refill8=item("q8","mass",mass_ep,refill=True)
    # The compact radius comparison remains at its original 10% budget.
    radius=max((VPolicy("mr25"),VPolicy("mr50")),key=lambda p:quality[p.name]["score"])
    int4=VPolicy("q4")
    methods=list(dict.fromkeys((q8,static,mass,other_mass,blasst,sparta,refill,refill8,radius,int4)))
    comparisons=dict(q8_fixed_k=(q8.name,mass.name),lowbit_fixed_k=(static.name,other_mass.name),
                     same_k_blasst=(blasst.name,mass.name),same_k_sparta=(sparta.name,mass.name),
                     strongest_static=(static.name,other_mass.name),
                     equal_budget_refill=(static.name,refill.name),
                     equal_budget_q8_refill=(q8.name,refill8.name),
                     mean_radius_vs_int4=(int4.name,radius.name))
    cal_manifest,_,cal=load_rows(root/"data"/cohort,"calibration")
    data=root/"data/test"
    new_cfg={**cal_manifest["configuration"],
             "splits":{"test":{"seed":2026112791,"count_per_task":128}},
             "output_directory":str(data)}
    cfg_path=root/"prepare_test.json"
    _write_json(cfg_path,new_cfg)
    if not (data/"manifest.json").exists():
        call("scripts.prepare_ruler","--config",cfg_path)
    test_manifest,_,rows=load_rows(data,"test")
    old_hashes=[r["source_metadata"]["instance_sha256"] for r in cal]
    if set(test_manifest["splits"]["test"]["seeds"].values()) & set(cal_manifest["splits"]["calibration"]["seeds"].values()):
        raise ValueError("calibration seeds reused")
    protocol=dict(**point,sample_count=len(rows),test_manifest_sha256=_sha(data/"manifest.json"),
                  calibration_instances=old_hashes,policies=[asdict(x) for x in methods],
                  thresholds=dict(official_gain=.01,read_saving=.10,noninferiority=-.01,
                                  paired_bootstrap_confidence=.95,bootstrap_repetitions=10000),
                  comparisons=comparisons,runner_sha256=_sha(Path("scripts/run_round4_fixed_k_v.py")),
                  calibration_receipts={name:_sha(root/name/"complete.json") for name in ("diagnostic",cohort)})
    path=root/"frozen_test.json"
    _write_json(path,protocol)
    return path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results/round4_fixed_k_v"))
    args=p.parse_args()
    root=args.root
    collect(root,"diagnostic")
    collect(root,"32k_mv")
    if not choose_workpoint(root):
        collect(root,"16k_fwe")
    point=choose_workpoint(root)
    if point is None:
        _write_json(root/"finished.json",dict(status="no_non_saturated_calibration_workpoint"))
        return
    frozen=root/"frozen_test.json"
    if not frozen.exists():
        frozen=freeze(root,point)
    call("scripts.run_round4_fixed_k_v","test","--root",root,"--frozen",frozen)
    summarize(root/"test",root/"report/test")
    compare_test(root)
    _write_json(root/"finished.json",dict(status="complete",frozen_sha256=_sha(frozen)))


if __name__=="__main__":
    main()
