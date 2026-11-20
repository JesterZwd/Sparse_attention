"""Aggregate Round 5 receipts and evaluate pre-frozen paired comparisons."""
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.shared_refill import RefillPolicy
from gqa_probe.ruler_experiment import _sha,_write_json
from scripts.report_ruler_round3 import write_csv


ANCHOR=134.5e6
MATCH=.01
SEED=20261130
FIELDS=("summary_payload","summary_padding","index_payload","index_padding","k_codes","k_scale_payload",
        "k_zp_payload","k_padding","v_codes","v_scale_payload","v_zp_payload","v_padding")
HW=("k_bursts_initial","k_bursts_additional","before_first_v_k_bursts","before_first_v_barriers",
    "v_coalesced_runs","transactions","row_hits","activations","qk_mac","pv_mac",
    "exact_gate_exp","pv_renorm_exp","head_max_comparisons","mass_token_to_page_reduce_adds",
    "mass_accumulation_check_ops","mass_sort_comparisons_est","exact_token_sort_comparisons_est",
    "blasst_peak_comparisons","blasst_running_max_comparisons",
    "q6_score_mac","q6_normalize_exp","q6_page_sort_comparisons_est",
    "histogram64_token_inserts","histogram64_two_pass_bin_visits")
SRAM=("fp32_logits","fp16_logits","page_mass_table","accumulator_bytes","token_mask_bytes",
      "histogram64_counter_bytes")


def load_cohort(root,phase):
    root=root/phase
    receipt=json.loads((root/"complete.json").read_text())
    conf=root/"config.json"
    if _sha(conf)!=receipt["config_sha256"]:
        raise ValueError(f"stale config in {phase}")
    ids=json.loads(conf.read_text())["ids"]
    bundles=[]
    for file,sha in receipt["prompts"].items():
        path=root/"bundles"/file
        if _sha(path)!=sha:
            raise ValueError(f"corrupt prompt bundle {file}")
        with gzip.open(path,"rt") as f:
            bundles.append(json.load(f))
    if sorted(ids)!=sorted(b["sample_id"] for b in bundles):
        raise ValueError("misaligned IDs")
    if sum(len(b["predictions"]) for b in bundles)!=receipt["predictions"]:
        raise ValueError("prediction count mismatch")
    return bundles


def official_quality(rows):
    by=defaultdict(list)
    for row in rows:
        by[row["policy"]].append(row)
    result=[]
    for policy,preds in by.items():
        sample=preds[0]
        calls=sum(p["kv_group_calls"] for p in preds)
        n_heads=4
        def per_call(key):
            return sum(p.get(key,0) for p in preds)/max(calls,1)
        ledger_rows={k:sum(p.get(k,0) for p in preds) for k in FIELDS}
        for p in preds:
            if p["read_per_answer"]!=sum(p.get(k,0) for k in FIELDS):
                raise ValueError("known budget components do not sum")
        maximum=max(p["peak_k_pages_per_group"] for p in preds)
        concurrent=max(p["peak_active_pages_across_groups"] for p in preds)
        peak_mass=4*n_heads*maximum if sample["mode"]=="mass" else 0
        peak_supply=concurrent*(4*n_heads*32+4*n_heads*(sample["mode"]=="mass"))
        result.append(dict(policy=policy,**{k:sample[k] for k in ("mode","threshold","budget_bp","recompute")},
                           prompts=len(preds),official_score=mean(p["official_score"] for p in preds),
                           full_count=sum(p["complete_correct"] for p in preds),
                           full_rate=mean(p["complete_correct"] for p in preds),
                           read_per_decode=mean(p["read_per_decode"] for p in preds),
                           read_per_answer=mean(p["read_per_answer"] for p in preds),
                           mean_tokens=mean(p["decode_tokens"] for p in preds),
                           truncated=sum(p["truncated"] for p in preds),
                           initial_k_pages=per_call("initial_k_pages"),added_k_pages=per_call("added_k_pages"),
                           v_pages=per_call("v_pages"),recompute_fraction=per_call("recomputed"),
                           fallback_fraction=per_call("fallback"),peak_k_pages=maximum,
                           peak_fp16_logits_per_group=maximum*n_heads*32*2,
                           peak_fp32_logits_per_group=maximum*n_heads*32*4,
                           peak_page_mass_table_per_group=peak_mass,groups=8,
                           peak_active_pages_8_groups=concurrent,
                           peak_8_groups_double_buffer_floor=2*(peak_supply+8*4*n_heads*(128+2)),
                           **ledger_rows))
    return result


def row_percentiles(a,p):
    return float(np.quantile(a,p)) if a else 0.


def summarize(root,phase):
    bundles=load_cohort(root,phase)
    out=root/"report"/phase
    out.mkdir(parents=True,exist_ok=True)
    preds=[dict(sample_id=b["sample_id"],task=b["task"],**r) for b in bundles for r in b["predictions"]]
    by=defaultdict(list)
    extrema=defaultdict(list)
    n_groups=0
    with gzip.open(out/"per_group.jsonl.gz","wt",compresslevel=4) as stream:
        for b in bundles:
            for row in b["groups"]:
                key=(row["basis"],row["policy"])
                x=dict(sample_id=b["sample_id"],**row)
                by[key].append(x)
                stream.write(json.dumps(x,separators=(",",":"))+"\n")
                n_groups+=1
                if row["policy"].startswith("blasst"):
                    extrema[key].append(x)
    quality=official_quality(preds)
    audit=[]
    for (scope,policy),rs in by.items():
        per_prompt=defaultdict(lambda:[[],[]])
        for r in rs:
            for j,key in enumerate(("first_stage_abs_error","final_abs_error")):
                per_prompt[r["sample_id"]][j].extend(r[key])
        abs_summary={f"{name}_abs_p{q}":mean(row_percentiles(vals[j],q/100) for vals in per_prompt.values())
                     for j,name in enumerate(("initial","final")) for q in (95,99)}
        audit.append(dict(basis=scope,policy=policy,n_groups=len(rs),prompts=len(per_prompt),
                          row_hit_rate=sum(r["row_hits"] for r in rs)/sum(r["transactions"] for r in rs),
                          **{f"mean_{key}":mean(r[key] for r in rs) for key in HW},
                          **{f"peak_{key}":max(r[key] for r in rs) for key in SRAM},
                          **abs_summary))
    hit_index={(r["sample_id"],r["policy"]):r["reference_hits"] for r in preds}
    peaks=[]
    associations=[]
    for (scope,policy),rs in extrema.items():
        for stage in ("initial","final"):
            losses=[v for r in rs for v in r[f"{stage}_discarded_mass"]]
            peaks.append(dict(basis=scope,policy=policy,stage=stage,head_observations=len(losses),
                              worst=max(losses),**{f"p_gt_{int(e*100)}":mean(x>e for x in losses)
                                                   for e in (.05,.1,.2)}))
        grouped=defaultdict(list)
        for row in rs:
            grouped[row["sample_id"]].append(row)
        for sample,x in grouped.items():
            hits=hit_index[sample,policy]
            missed={i for i,v in enumerate(hits) if not v}
            tail=[r for r in x if max(r["final_discarded_mass"])>.2]
            omitted={j for r in tail for j in r["gold_targets_with_selected_k_page_omitted_in_v"]}
            associations.append(dict(policy=policy,basis=scope,sample_id=sample,missed_values=len(missed),
                                     sampled_group_count=len(x),tail_groups=len(tail),tail_omitted_values=len(omitted),
                                     tail_omitted_also_missing=len(missed&omitted)))
    write_csv(out/"quality.csv",quality)
    write_csv(out/"hardware_and_errors.csv",audit)
    write_csv(out/"blasst_head_tails.csv",peaks)
    write_csv(out/"blasst_tail_and_misses.csv",associations)
    _write_json(out/"predictions.json",preds)
    _write_json(out/"summary.json",dict(quality=quality,audited_groups=n_groups,predictions=len(preds)))
    return quality


def stronger(row,dist=ANCHOR):
    return (row["official_score"],row["full_count"],-abs(row["read_per_decode"]-dist))


def pick_mode(rows,mode,dist):
    joint=[r for r in rows if r["mode"]==mode and r["budget_bp"]==1000]
    nearby=[r for r in joint if abs(r["read_per_decode"]-dist)<=.01*dist]
    return max(nearby or joint,key=lambda r:stronger(r,dist))


def to_policy(row):
    return RefillPolicy(**{k:row[k] for k in ("mode","threshold","budget_bp","recompute")})


def choose_fixed(quality):
    m=[r for r in quality if r["mode"]=="mass" and r["budget_bp"]==1000]
    five=next(r for r in m if r["threshold"]==.05)
    tight=max((r for r in m if r["threshold"]<.05),key=stronger)
    main=max((five,tight),key=stronger)
    fixed={name:pick_mode(quality,name,main["read_per_decode"]) for name in ("all","blasst","sparta")}
    return five,tight,main,fixed


def choose_equal(data,primary,fixed):
    use=primary["read_per_decode"]
    result={}
    for mode,reference in fixed.items():
        options=[r for r in data if (r["mode"],r["threshold"],r["recompute"])==
                 (mode,reference["threshold"],reference["recompute"])]
        at=[r for r in options if abs(r["read_per_decode"]-use)<=MATCH*use]
        low=[r for r in options if r["read_per_decode"]<use*(1-MATCH)]
        result[mode]=dict(matched=max(at,key=lambda r:stronger(r,use)) if at else None,
                          lower=max(low,key=lambda r:(r["official_score"],r["full_count"],-r["read_per_decode"])) if low else None,
                          nearest=min(options,key=lambda r:abs(r["read_per_decode"]-use)))
    return result


def test_report(root,frozen,quality):
    report=root/"report/test"
    rows=json.loads((report/"predictions.json").read_text())
    by={r["policy"]:r for r in quality}
    primary=frozen["primary"]
    current=by[primary]
    comparisons={}
    for name,baseline in frozen["comparators"].items():
        b=by[baseline]
        paired={metric:paired_bootstrap(rows,baseline,primary,metric,confidence=.95,seed=SEED,
                                       method_key="policy") for metric in
                ("official_score","complete_correct","read_per_answer","read_per_decode")}
        a={p["sample_id"]:p["official_score"] for p in rows if p["policy"]==primary}
        q={p["sample_id"]:p["official_score"] for p in rows if p["policy"]==baseline}
        d=[a[k]-q[k] for k in a]
        gap=(current["read_per_decode"]-b["read_per_decode"])/b["read_per_decode"]
        score=paired["official_score"]
        answer=paired["read_per_answer"]
        saving=-answer["relative_delta"]
        saving_lb=-answer["relative_ci_high"]
        comparisons[name]=dict(baseline=baseline,primary=primary,read_gap=gap,actual_match=abs(gap)<=MATCH,
                               answer_saving=saving,answer_saving_lcb=saving_lb,
                               win=sum(x>0 for x in d),loss=sum(x<0 for x in d),tie=sum(x==0 for x in d),
                               quality_threshold=(abs(gap)<=MATCH and score["point_delta"]>=.01 and score["ci_low"]>0),
                               savings_threshold=(score["ci_low"]>=-.01 and saving_lb>=.10),paired=paired)
    dominant=sorted({x["baseline"] for x in comparisons.values()
                     if x["read_gap"]>=0 and x["paired"]["official_score"]["point_delta"]<0})
    pivot=frozen["primary_comparison"]
    main=comparisons[pivot]
    advanced=(main["quality_threshold"] or main["savings_threshold"]) and not dominant
    _write_json(report/"paired_and_gate.json",dict(primary=primary,primary_comparison=pivot,
              advanced=advanced,comparison_with_higher_score_and_no_more_reads=dominant,comparisons=comparisons))
    write_csv(report/"fixed_ceiling.csv",[by[k] for k in frozen["fixed_table"]])
    write_csv(report/"realized_read_table.csv",
              [dict(relationship="primary",matched=True,**current),*[
                dict(relationship=label,matched=abs(by[name]["read_per_decode"]/current["read_per_decode"]-1)<=MATCH,
                     **by[name]) for label,name in frozen["realized_table"].items()]])
    return advanced
