"""Freeze selection rules, execute Round 7, and gate the unused 128 prompts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from gqa_probe.round7_attention import Policy
from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.round7_report import checkpoint, pairs, phase_report
from scripts.round7_run import (BASE, CAPPED, COMPACT, DATA, DENSE, DYNAMIC, FIXED, HIST, ROOT,
                                SINGLE, STATIC, run, sources)
from scripts.run_ruler_round3 import load_rows


QUALITY_MARGIN = .01
COST_MARGIN = .05
READ_TOLERANCE = .01


def freeze_json(path,value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen value changed at {path}")
    _write_json(path,value)


def inventory(data,split):
    path = data/split
    manifest,_,rows = load_rows(path,split)
    return dict(manifest_sha256=_sha(path/"manifest.json"),
                rows_sha256=manifest["splits"][split]["sha256"],
                seed=manifest["configuration"]["splits"][split]["seed"],
                ids=[r["sample_id"] for r in rows])


def verify_independence(data):
    seen = set()
    for path,split in ((data/"calibration","calibration"),
                       (Path("results/round5_shared_refill/data/test"),"test"),
                       (Path("results/round4_fixed_k_v/data/test"),"test")):
        if path.exists():
            _,_,rows = load_rows(path,split)
            seen.update(r["source_metadata"]["instance_sha256"] for r in rows)
    _,_,test = load_rows(data/"test","test")
    test_ids = {r["source_metadata"]["instance_sha256"] for r in test}
    if len(test) != 128 or len(test_ids) != 128 or seen & test_ids:
        raise ValueError("independent test contains repeated or prior prompts")


def best_screen(rows, kind):
    table = {x["policy"]:x for x in rows if x["cohort"]=="all"}
    ref = table[HIST.name]
    ps = [p for p in FIXED if p.gate == kind]
    def key(policy):
        x = table[policy.name]
        return (x["reads_bytes"] > ref["reads_bytes"]*(1+READ_TOLERANCE),
                x["head_abs_p99"] > 1.1*ref["head_abs_p99"]+1e-6,
                x["head_abs_p99"],x["total_group_ck"],abs(x["reads_bytes"]-ref["reads_bytes"]))
    return min(ps,key=key)


def gain(base, candidate,key):
    before = base[key]
    after = candidate[key]
    return (before-after)/before if before and after is not None else 0.


def decision(root, data, rows, policies):
    table = {r["policy"]:r for r in rows}
    sparse = [p for p in policies if p.kind != "dense" and not table[p.name]["over_budget"]
              and table[p.name]["read_mb_per_decode"] is not None]
    # Anchor has the strongest score among compliant measured reference points.
    # The exact single-pass row is a numerical reference, with sorter estimates
    # that cannot be used as a weak hardware comparator.
    controls = [p for p in sparse if p.gate != "checked" and
                (p.gate != "exact" or p.kind == "dynamic")]
    baseline = max(controls,key=lambda p:(table[p.name]["official_score"],
                   -table[p.name]["read_mb_per_decode"],-table[p.name]["mean_total_group_ck"]))
    ref = table[baseline.name]
    alternatives = []
    for p in sparse:
        if p.kind!="single" or p.gate not in ("checked","hist64") or p.name==baseline.name:
            continue
        r=table[p.name]
        clock=gain(ref,r,"mean_total_group_ck")
        sram=gain(ref,r,"peak_control_sram_bytes")
        passed=(r["official_score"] >= ref["official_score"]-QUALITY_MARGIN-1e-9 and
                r["read_mb_per_decode"] <= ref["read_mb_per_decode"]*(1+READ_TOLERANCE) and
                max(clock,sram) >= COST_MARGIN)
        alternatives.append(dict(policy=p.name,passes=passed,cycle_reduction=clock,
                                 control_sram_reduction=sram,score_delta=r["official_score"]-ref["official_score"],
                                 read_delta=r["read_mb_per_decode"]/ref["read_mb_per_decode"]-1))
    success = [r for r in alternatives if r["passes"]]
    selected = max(success,key=lambda r:(r["score_delta"],max(r["cycle_reduction"],
                   r["control_sram_reduction"])))["policy"] if success else None
    _write_json(root/"calibration_decision.json",dict(anchor=baseline.name,selected=selected,
                alternatives=alternatives,pass_thresholds=dict(quality_margin=QUALITY_MARGIN,
                read_tolerance=READ_TOLERANCE,cycle_or_control_margin=COST_MARGIN)))
    if selected is None:
        return None
    candidate = next(p for p in sparse if p.name==selected)
    verify_independence(data)
    phases = [p for p in ("baseline","compact","screen","val_gate","compact_gate")
              if (root/p/"complete.json").exists()]
    frozen = dict(**inventory(data,"test"),policies=list(map(asdict,(DENSE,baseline,candidate))),
                  source_sha256=sources(),anchor=baseline.name,candidate=candidate.name,
                  calibration_receipts={p:_sha(root/p/"complete.json") for p in phases},
                  protocol_sha256=_sha(root/"protocol.json"))
    freeze_json(root/"frozen_test.json",frozen)
    return frozen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    root,data=args.root,args.data
    backend=dict(device=args.device,threads=args.threads)
    root.mkdir(parents=True,exist_ok=True)
    base=[DENSE,DYNAMIC,CAPPED,SINGLE,STATIC]
    protocol=dict(base=BASE,context=32768,page_size=32,head_drop=.02,topology="4:1 GQA",
                  framework="affine32 dequantized numerical reference with independent burst replay",
                  primary_cap=10,compact_cap=5,closed_compact_policies=list(map(asdict,COMPACT)),
                  closure_datasets={s:inventory(data,s) for s in ("calibration","test")},
                  fitting_prompts=8,validation_prompts=16,side_prompts=4,test_prompts=128,
                  control_cost_margin=COST_MARGIN,quality_margin=QUALITY_MARGIN,
                  read_tolerance=READ_TOLERANCE,baseline=list(map(asdict,base)),
                  fixed_k_replay=list(map(asdict,FIXED)),score_ci=.95,
                  compact_gate_screen_min_vs_dense=-.01,source_sha256=sources(),**backend)
    freeze_json(root/"protocol.json",protocol)
    run(root,data,"baseline",base,**backend)
    baseline,_,_=phase_report(root,"baseline")
    run(root,data,"screen",[DENSE],count=8,common=FIXED,side_count=4,audits=0,**backend)
    _,screen,_=phase_report(root,"screen")
    screen_a=best_screen(screen,"blasst")
    screen_b=best_screen(screen,"checked")
    run(root,data,"compact",COMPACT,**backend)
    compact,_,compact_answers=phase_report(root,"compact")
    for mode in ("allv","plus4","max"):
        pairs(root/"report"/"compact"/f"paired_{mode}.json",compact_answers,f"q4_5_{mode}",f"mr25_5_{mode}")
    ordered = sorted(compact,key=lambda x:(-x["official_score"],x["read_mb_per_decode"]))
    compact_by_name={x["policy"]:x for x in compact}
    eligible = [r for r in ordered if r["policy"].startswith("mr25") and not r["over_budget"]
                and r["official_score"] >= baseline[0]["official_score"]-.01
                and all(r[k] > compact_by_name[r["policy"].replace("mr25","q4")][k]
                        for k in ("k_evidence_page_recall","v_evidence_page_recall"))]
    active=[HIST,screen_a,screen_b]
    compact_gate_rows,opts=[],[]
    if eligible:
        selected=next(p for p in COMPACT if p.name==eligible[0]["policy"])
        opts=[Policy(f"compact_{g.gate}",method=selected.method,budget=5,bonus=selected.bonus,
                     gate=g.gate,threshold=g.threshold) for g in active]
        run(root,data,"compact_gate",opts,common=opts,**backend)
        compact_gate_rows,_,_=phase_report(root,"compact_gate")
    run(root,data,"val_gate",active,**backend)
    validation,_,_=phase_report(root,"val_gate")
    freeze=decision(root,data,baseline+validation+compact_gate_rows,base+active+opts)
    receipt=dict(status="calibration_complete",heldout=False,passed=False,
                 compact_admitted=bool(eligible),checked_tau=screen_b.threshold,
                 blasst_lambda=screen_a.threshold,group_timing="sampled; modeled LPDDR5 dependencies",
                 source_sha256=sources())
    test=None
    if freeze:
        run(root,data,"test",[Policy(**p) for p in freeze["policies"]],audits=8,**backend)
        test,_,answers=phase_report(root,"test")
        out=root/"report"/"test"
        p=pairs(out/"main_pair.json",answers,freeze["anchor"],freeze["candidate"])
        pairs(out/"dense_pair.json",answers,DENSE.name,freeze["candidate"])
        table={r["policy"]:r for r in test}
        a,b,d=(table[k] for k in (freeze["anchor"],freeze["candidate"],DENSE.name))
        speed=gain(a,b,"mean_total_group_ck")
        sram=gain(a,b,"peak_control_sram_bytes")
        def mac(r):
            return sum(r[k+"_per_decode"] for k in ("qk_mac","pv_mac","scorer_mac"))
        passed=(not a["over_budget"] and not b["over_budget"] and
                p["official_score"]["ci_low"] >= -QUALITY_MARGIN-1e-9 and
                b["read_mb_per_decode"] <= a["read_mb_per_decode"]*(1+READ_TOLERANCE) and
                b["read_mb_per_decode"] < d["read_mb_per_decode"] and mac(b)<mac(d) and
                max(speed,sram)>=COST_MARGIN)
        receipt.update(status="heldout_complete",heldout=True,passed=passed,cycle_reduction=speed,
                       control_sram_reduction=sram,score_ci_lower=p["official_score"]["ci_low"])
    _write_json(root/"finished.json",receipt)
    checkpoint(root,baseline,compact,validation+compact_gate_rows,receipt,test)


if __name__=="__main__":
    main()
