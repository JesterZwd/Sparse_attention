"""Matched-read Q8 scorer calibration and frozen RULER evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

import numpy as np

from gqa_probe.fixed_k_v import byte_quota
from gqa_probe.independent_gate import independent_decision
from gqa_probe.page_read_budget import page_history
from gqa_probe.pagepack_attention import Policy
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.scorer_int import weights as fixed_weights
from gqa_probe.vskip_summary import weights as float_weights
from scripts.report_ruler_round3 import write_csv
from scripts.round7_pipeline import verify_independence
from scripts.round7_run import DATA
from scripts.run_pagepack import pairing, report, run, source_hashes
from scripts.run_ruler_round3 import load_rows


ROOT=Path("results/round9_fused_scorer")
BASE="5a3a27ab84981aa0e1580d43f85140ea21ebfb1d"
CAPS=(5,7)
FIT=4
TOL=.01
RESERVES=(.50,.65,.80)
REF10=Policy("b10_q6_single64","q6","raw","hist64",10,0.,68)


def sources():
    return {**source_hashes(),**{p:_sha(Path(p)) for p in (
        "scripts/run_fused_scorer.py","gqa_probe/scorer_int.py","gqa_probe/lpddr5_banked.py",
        "sim/lpddr5_banked.cpp","gqa_probe/paired_statistics.py")}}


def frozen(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError(f"existing freeze differs at {path}")
    _write_json(path,value)


def policies(reserves=RESERVES):
    return [Policy(f"b{b:02d}_{m}_r{round(1000*r):03d}",m,"raw","hist64",b,r)
            for b in CAPS for m in ("q4","mr25") for r in reserves]


def get_scores(root,row,p,export):
    method=p.summary
    encode=fixed_weights if method in ("q4","mr25") else float_weights
    enabled=(export and p.budget==5 and p.v_reserve==.65 and method in ("q4","mr25")
             and row["sample_id"].rsplit("/",1)[-1] in ("0000","0001"))
    def scorer(q,rec,which,scale,*,step,layer):
        if enabled and layer in (7,21,35) and step in (0,7,15,31,63):
            t=row["input_tokens"]+step+1
            history=page_history(t,32)
            target=root/"local_inputs"/f"{row['sample_id'].rsplit('/',1)[-1]}_{which}_{layer:02d}_{step:03d}.npz"
            target.parent.mkdir(exist_ok=True)
            np.savez(target,query=q[:1].float().cpu().numpy(),scale=scale,method=which,
                     protected=np.array(history.protected),
                     k=byte_quota(t,32,which,5),
                     **{f"r{i}":x[:1].cpu().numpy() for i,x in enumerate(rec)})
        return encode(q,rec,which,scale)
    return scorer


def phase(root,data,name,opts,*,start=0,count=None,freeze=None,export=False):
    run(root,data,name,opts,start=start,count=count,freeze=freeze,dense=False,
        score_factory=lambda r,p:get_scores(root,r,p,export),run_sources=sources(),base_commit=BASE)
    return report(root,name)


def pick(opts,results,excluded=()):
    rows={r["policy"]:r for r in results}
    choices={}
    out=[]
    for cap in CAPS:
        q4=[p for p in opts if p.budget==cap and p.summary=="q4"]
        mr=[p for p in opts if p.budget==cap and p.summary=="mr25"]
        anchor=[r["bytes_per_decode"] for p in q4+mr if p.v_reserve==.65 for r in [rows[p.name]]]
        center=mean(anchor)
        possibilities=[]
        for a in q4:
            for b in mr:
                x,y=(rows[p.name] for p in (a,b))
                distance=y["bytes_per_decode"]/x["bytes_per_decode"]-1
                target=abs(mean((x["bytes_per_decode"],y["bytes_per_decode"]))/center-1)
                label=dict(cap=cap,reference=a.name,candidate=b.name,read_delta=distance,
                           score_delta=y["official_score"]-x["official_score"],center_delta=target,
                           commensurate=abs(distance)<=TOL and not (x["over_budget"] or y["over_budget"]))
                out.append(label)
                if (a.name,b.name) not in excluded and label["commensurate"]:
                    possibilities.append((target,abs(distance),a.name,b.name,a,b))
        if possibilities:
            choices[cap]=min(possibilities)[-2:]
    return choices,out


def compare(root,phase_name,results,pred,ref,new,out_name):
    rows={r["policy"]:r for r in results}
    a,b=(rows[p.name] for p in (ref,new))
    paired=pairing(root/"report"/phase_name/f"{out_name}.json",pred,ref.name,new.name)
    read=paired["read"]
    score=paired["official_score"]
    gates=independent_decision(score_delta=score["point_delta"],score_ci_low=score["ci_low"],
                reference_bytes=a["bytes_per_decode"],candidate_bytes=b["bytes_per_decode"],
                answer_read_relative_delta=read["relative_delta"],
                answer_read_relative_hi=read["relative_ci_high"],
                reference_over=a["over_budget"],candidate_over=b["over_budget"])
    return dict(reference=ref.name,candidate=new.name,score_delta=score["point_delta"],
                score_ci=[score["ci_low"],score["ci_high"]],reference_score=a["official_score"],
                candidate_score=b["official_score"],reference_complete=a["complete"],
                candidate_complete=b["complete"],reference_bytes_per_decode=a["bytes_per_decode"],
                candidate_bytes_per_decode=b["bytes_per_decode"],answer_read_delta=read["relative_delta"],
                answer_read_ci=[read["relative_ci_low"],read["relative_ci_high"]],**gates)


def calibrate(root,data):
    root.mkdir(parents=True,exist_ok=True)
    plan=dict(base=BASE,caps=list(CAPS),fit_count=FIT,reserves=list(RESERVES),read_tolerance=TOL,
              refinement=[.575,.725],test_count=128,q4_and_mr_scorer="Q8 query, 22-bit Q24 coefficient, FP32 A1",
              ref10=asdict(REF10),quality_gain=.01,quality_ni=-.01,min_answer_saving=.10,
              code=sources(),cal_manifest=_sha(data/"calibration"/"manifest.json"))
    frozen(root/"protocol.json",plan)
    opts=policies()
    fit,pred=phase(root,data,"fit",opts,count=FIT,export=True)
    matched,comparisons=pick(opts,fit)
    phase_hashes={"fit":_sha(root/"fit"/"complete.json")}
    if set(matched)!=set(CAPS):
        additional=[p for p in policies((.575,.725)) if p.budget not in matched]
        rows,more=phase(root,data,"refine",additional,count=FIT)
        opts+=additional
        fit+=rows
        pred+=more
        matched,comparisons=pick(opts,fit)
        phase_hashes["refine"]=_sha(root/"refine"/"complete.json")
    _write_json(root/"matching_fit.json",dict(matches={str(k):list(map(asdict,v)) for k,v in matched.items()},
                                               comparisons=comparisons))
    valopts=[p for pair in matched.values() for p in pair]+[REF10]
    results,test=phase(root,data,"validation",valopts,start=FIT)
    phase_hashes["validation"]=_sha(root/"validation"/"complete.json")
    validation={}
    good={}
    for cap,(ref,new) in matched.items():
        row=compare(root,"validation",results,test,ref,new,f"pair_b{cap}")
        validation[str(cap)]=row
        if row["matched_bytes"] and row["zero_overruns"]:
            good[cap]=(ref,new)
    if 7 in matched:
        validation["7_vs_10"]=compare(root,"validation",results,test,REF10,matched[7][1],"pair_7_vs_10")
    selection=dict(eligible_caps=list(good),fits={str(k):list(map(asdict,v)) for k,v in good.items()},
                   validation=validation,calibration_phase_sha256=phase_hashes,
                   protocol_sha256=_sha(root/"protocol.json"))
    _write_json(root/"calibration_decision.json",selection)
    print(json.dumps({k:selection[k] for k in ("eligible_caps","validation")}),flush=True)


def independent(root,data):
    cal=json.loads((root/"calibration_decision.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())
    if cal["protocol_sha256"]!=_sha(root/"protocol.json") or protocol["code"]!=sources():
        raise ValueError("calibration protocol or code changed")
    pairs={int(b):tuple(Policy(**x) for x in pair) for b,pair in cal["fits"].items()}
    if not pairs:
        _write_json(root/"finished.json",dict(status="no_matched_calibration",heldout=False,passes=False))
        return
    numerics=json.loads((root/"hardware"/"scorer_numerics.json").read_text())
    fu=json.loads((root/"hardware"/"rtl_validation.json").read_text())
    if (numerics["input_kind"]!="captured" or numerics["coefficient_bits"]!=22 or
            any(any(x["saturated_coefficients"]) or x["stages"]["q24"]["mean_optional_topk_overlap"]<.95
                for x in numerics["summary"].values()) or
            set(numerics["summary"])!={"q4","mr25"} or len(fu)!=4):
        raise ValueError("numerical and RTL prerequisites for the independent run failed")
    verify_independence(data)
    _,_,rows=load_rows(data/"test","test")
    opts=[p for k,pair in sorted(pairs.items()) for p in pair]
    if 7 in pairs:
        opts.append(REF10)
    freeze=dict(ids=[r["sample_id"] for r in rows],policies=list(map(asdict,opts)),
                sources=sources(),manifest_sha256=_sha(data/"test"/"manifest.json"),
                calibration_sha256=_sha(root/"calibration_decision.json"),predicate="fixed zero-cap, 1pp/1% or NI+10%",
                numerics_sha256=_sha(root/"hardware"/"scorer_numerics.json"),
                rtl_validation_sha256=_sha(root/"hardware"/"rtl_validation.json"))
    frozen(root/"frozen_test.json",freeze)
    table,pred=phase(root,data,"independent",opts,freeze=freeze)
    comparisons={}
    for cap,(ref,new) in pairs.items():
        comparisons[str(cap)]=compare(root,"independent",table,pred,ref,new,f"pair_b{cap}")
    if 7 in pairs:
        comparisons["7_vs_10"]=compare(root,"independent",table,pred,REF10,pairs[7][1],"pair_7_vs_10")
    write_csv(root/"report"/"independent"/"decision.csv",list(comparisons.values()))
    _write_json(root/"finished.json",dict(status="independent_complete",heldout=True,
                passes=any(r["passes"] for r in comparisons.values()),comparisons=comparisons,
                freeze_sha256=_sha(root/"frozen_test.json")))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("calibrate","independent"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    (calibrate if a.stage=="calibrate" else independent)(a.root,a.data)


if __name__=="__main__":
    main()
