"""Durable staged calibration, freeze, and held-out test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from gqa_probe.ruler_experiment import _sha,_write_json
from gqa_probe.shared_refill import RefillPolicy
from scripts.report_round5_shared_refill import (choose_equal,choose_fixed,summarize,test_report,to_policy)
from scripts.round5_shared_refill import ROOT,code_hashes
from scripts.run_ruler_round3 import load_rows


def run(root,phase):
    subprocess.run([sys.executable,"-u","-m","scripts.round5_shared_refill",phase,"--root",str(root)],check=True)


def prep_stage2(root,fixed):
    candidates=[to_policy(r) for mode,r in fixed.items() if mode!="all"]
    rows=[asdict(RefillPolicy(p.mode,p.threshold,b,p.recompute)) for p in candidates for b in (950,900)]
    _write_json(root/"cal2_policies.json",rows)


def freeze(root,cal1,cal2):
    five,tight,primary,fixed=choose_fixed(cal1)
    pairs=choose_equal(cal1+cal2,primary,fixed)
    by={r["policy"]:r for r in cal1+cal2}
    fixed_table=list(dict.fromkeys([fixed["all"]["policy"],five["policy"],tight["policy"],
                                   fixed["blasst"]["policy"],fixed["sparta"]["policy"]]))
    realized={}
    comparators={f"fixed_{m}":r["policy"] for m,r in fixed.items()}
    for mode,r in pairs.items():
        for kind in ("matched","lower","nearest"):
            choice=r[kind]
            if choice and choice["policy"] not in realized.values():
                realized[f"{kind}_{mode}"]=choice["policy"]
    comparators.update(realized)
    eligible=[(k,by[p]) for k,p in comparators.items()
              if abs(by[p]["read_per_decode"]/primary["read_per_decode"]-1)<=.01]
    lows=[(k,by[p]) for k,p in comparators.items() if by[p]["read_per_decode"]<=primary["read_per_decode"]]
    target=max(eligible or lows or [(k,by[p]) for k,p in comparators.items()],
               key=lambda item:(item[1]["official_score"],item[1]["full_count"],-item[1]["read_per_decode"]))[0]
    key_names=list(dict.fromkeys(fixed_table+list(realized.values())))
    policies=[asdict(to_policy(by[name])) for name in key_names]
    shadow=[asdict(to_policy(by[name])) for name in fixed_table if to_policy(by[name]).mode!="all"]
    manifest,_,rows=load_rows(root/"data"/"test","test")
    cal_manifest,_,cal=load_rows(root/"data"/"calibration","calibration")
    if set(manifest["splits"]["test"]["seeds"].values())&set(cal_manifest["splits"]["calibration"]["seeds"].values()):
        raise ValueError("seed reuse")
    cal_sig={r["source_metadata"]["instance_sha256"] for r in cal}
    old_manifest=Path("results/round4_fixed_k_v/data/test")
    _,_,old=load_rows(old_manifest,"test")
    old_sig={r["source_metadata"]["instance_sha256"] for r in old}
    test_sig={r["source_metadata"]["instance_sha256"] for r in rows}
    if test_sig & (cal_sig|old_sig):
        raise ValueError("held-out instances were previously used")
    protocol=dict(policies=policies,primary=primary["policy"],primary_comparison=target,
                  comparators=comparators,fixed_table=fixed_table,realized_table=realized,
                  common_state_shadow=shadow,round4_replication_policy=five["policy"],
                  tight_mass_policy=tight["policy"],calibration_instances=sorted(cal_sig),
                  round4_diagnostic_instances_in_test=0,
                  test_ids=[r["sample_id"] for r in rows],audit_test_ids=[r["sample_id"] for r in rows[:8]],
                  test_manifest_sha256=_sha(root/"data"/"test"/"manifest.json"),
                  calibration_manifest_sha256=_sha(root/"data"/"calibration"/"manifest.json"),
                  sources=code_hashes(),thresholds=dict(match_relative=.01,score_gain=.01,noninferiority=-.01,
                                  per_answer_saving_lcb=.10,ci=.95,bootstrap_repetitions=10000,bootstrap_seed=20261130),
                  stage_receipts={phase:_sha(root/phase/"complete.json") for phase in ("cal1","cal2")},
                  stage_reports={phase:_sha(root/"report"/phase/"summary.json") for phase in ("cal1","cal2")})
    _write_json(root/"frozen_test.json",protocol)
    return protocol


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    root=a.root
    run(root,"smoke")
    run(root,"cal1")
    cal1=summarize(root,"cal1")
    *_,fixed=choose_fixed(cal1)
    prep_stage2(root,fixed)
    run(root,"cal2")
    cal2=summarize(root,"cal2")
    saved=root/"frozen_test.json"
    protocol=json.loads(saved.read_text()) if saved.exists() else freeze(root,cal1,cal2)
    run(root,"test")
    quality=summarize(root,"test")
    advanced=test_report(root,protocol,quality)
    _write_json(root/"finished.json",dict(status="complete",advanced=advanced,frozen_sha256=_sha(saved)))


if __name__=="__main__":
    main()
