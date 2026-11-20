"""Run Round 6, gating optional probes and held-out generation on calibration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from gqa_probe.round6_attention import Policy
from gqa_probe.ruler_experiment import _sha,_write_json
from scripts.report_round6 import (LPDDR,THRESHOLDS,audit_screen,choose_anchor,finalize,
                                   phase_report,validation_plan)
from scripts.round6_subpage_head import ROOT,sources
from scripts.run_ruler_round3 import load_rows


def run(root,phase,policies=None,**extra):
    if policies is not None and phase!="test":
        _write_json(root/f"{phase}_plan.json",dict(policies=[asdict(x) for x in policies],**extra))
    subprocess.run([sys.executable,"-u","-m","scripts.round6_subpage_head",phase,"--root",str(root)],check=True)
    return phase_report(root,phase)


def prepare_protocol(root):
    config=dict(context=32768,percent_cap=10,page=32,head_drop=.02,
                fitting_prompts=8,validation_prompts=8,heldout_prompts=128,
                single_ks=[68,74,78],preserve_overrun_points=True,lpddr=LPDDR,thresholds=THRESHOLDS,
                normalization_cycle_trigger=.10,compact_bytes_share_trigger=.40,
                compact_memory_cycle_share_trigger=.35,source_sha256=sources(),
                data_manifests={p:_sha(root/"data"/p/"manifest.json") for p in ("calibration","test")})
    path=root/"protocol.json"
    if path.is_file() and json.loads(path.read_text())!=config:
        raise ValueError("protocol changed after launch")
    _write_json(path,config)


def freeze_test(root,anchor,policies,maps,candidates):
    _,_,test=load_rows(root/"data/test","test")
    _,_,cal=load_rows(root/"data/calibration","calibration")
    known={r["source_metadata"]["instance_sha256"] for r in cal}
    for phase,path in (("test","results/round4_fixed_k_v/data/test"),("test","results/round5_shared_refill/data/test")):
        _,_,old=load_rows(Path(path),phase)
        known.update(r["source_metadata"]["instance_sha256"] for r in old)
    if known&{r["source_metadata"]["instance_sha256"] for r in test}:
        raise ValueError("candidate holdout overlaps calibration or older evaluation")
    info=dict(anchor_name=anchor.name,candidates=candidates,maps=maps,
              policies=[asdict(p) for p in policies],sources=sources(),
              test_ids=[r["sample_id"] for r in test],test_manifest=_sha(root/"data/test/manifest.json"),
              calibration_phase_hashes={p:_sha(root/p/"complete.json") for p in ("baseline","screen","valid")},
              protocol_sha256=_sha(root/"protocol.json"))
    out=root/"frozen_test.json"
    if out.is_file() and json.loads(out.read_text())!=info:
        raise ValueError("held-out freeze differs")
    _write_json(out,info)
    return info


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    root=a.root
    prepare_protocol(root)
    run(root,"smoke")
    baseline=run(root,"baseline")
    anchor=choose_anchor(baseline)
    run(root,"screen",[Policy("dense")],anchor=asdict(anchor))
    s=audit_screen(root)
    if s["compact_probe"]:
        run(root,"compact",[Policy("compact",method=m) for m in ("q4","mr25","q6")])
    validation=validation_plan(anchor,s)
    eligible=[]
    maps=s["maps"]
    if len(validation)>1:
        observed=run(root,"valid",validation,maps=maps)
        raw={r["policy"]:r for r in observed}
        score=raw[anchor.name]["official_score"]
        for v in validation[1:]:
            row=raw[v.name]
            if v.window=="group":
                continue
            if (row["official_score"]>=score-.01 and not row["over_budget"]):
                eligible.append(v.name)
    if not eligible:
        _write_json(root/"finished.json",dict(status="screened_out",anchor=anchor.name,passed=False,
                                            compact=s["compact_probe"],light_a1=s["light_a1_needed"]))
        return
    controls=[v for v in validation if v.name in eligible or
              v.window=="group" and any(p.name in eligible and p.window=="head" for p in validation)]
    combined=list(dict.fromkeys([Policy("dense"),Policy("static"),anchor,*controls]))
    conf=freeze_test(root,anchor,combined,maps,eligible)
    run(root,"test")
    gate=finalize(root,conf)
    _write_json(root/"finished.json",dict(status="heldout_complete",anchor=anchor.name,
                                       passed=any(v["passed"] for v in gate.values()),
                                       compact=s["compact_probe"],light_a1=s["light_a1_needed"],
                                       test_freeze=_sha(root/"frozen_test.json")))


if __name__=="__main__":
    main()
