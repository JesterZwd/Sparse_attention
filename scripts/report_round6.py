"""Pre-frozen mechanism filters and held-out comparisons for Round 6."""
from collections import defaultdict
from dataclasses import asdict,replace
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.page_read_budget import page_history
from gqa_probe.round6_attention import Policy
from gqa_probe.round6_audit import fixed_k_window
from gqa_probe.ruler_experiment import _sha,_write_json
from scripts.report_ruler_round3 import write_csv


HARDWARE=("reads_bytes","bursts","lpddr_ck","lpddr_reads_and_drains_ck","total_group_ck",
          "summary_index_ck","before_first_v_ck","v_runs","read_runs","row_hits","activations",
          "qk_mac","pv_mac","scorer_mac","scorer_pe_util","qk_pe_util","pv_pe_util","normalize_ck",
          "score_dot_ck","topk_insert_ck","initial_k","added_k","added_k_bursts","initial_qk_ck",
          "extra_qk_ck","initial_gate_ck","extra_gate_ck","fixed_array_ck","ideal_packed_array_ck",
          "group_barriers","expanded_tokens","extra_v_bytes")
CAPACITY=("peak_logits_bytes_fp32","peak_logits_bytes_fp16","page_or_block_probability_bytes",
          "token_mask_bytes","histogram64_bytes","head_drop_state_bytes","pv_accumulator_bytes")
OPERATIONS=("histogram64_inserts","histogram64_bin_visits","subpage_reduce_adds","subpage_bound_checks")
THRESHOLDS=dict(b8_tail_ratio_max=1.10,b8_advantage_min=.03,window_head_rel_p95_max=.05,
                window_mac_reduction_min=.05,validation_quality_loss_max=.01,
                final_cycle_or_control_reduction_min=.05,score_ci_lower_min=-.01)
LPDDR=dict(tck_ps=1250,rate_mt_s=6400,dq=16,burst_bytes=32,banks=16,bank_groups=4,rows_bytes=2048,
           queue=32,controller="ordered read only, open row, model without refresh",
           timing_source="https://github.com/CMU-SAFARI/ramulator2/blob/7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da/python/ramulator/dram/lpddr5.py")


def load(root,phase):
    rp=root/phase
    conf=rp/"config.json"
    c=json.loads((rp/"complete.json").read_text())
    if c["config_sha256"]!=_sha(conf):
        raise ValueError("experiment config checksum mismatch")
    out=[]
    for n,v in c["prompts"].items():
        path=rp/"bundles"/n
        if _sha(path)!=v:
            raise ValueError("prompt receipt mismatch")
        with gzip.open(path,"rt") as f:
            out.append(json.load(f))
    if set(json.loads(conf.read_text())["ids"])!={b["sample_id"] for b in out}:
        raise ValueError("missing experiment prompt")
    if c["predictions"]!=sum(len(x["answers"]) for x in out):
        raise ValueError("generation count mismatch")
    return out


def summary_rows(rows):
    out={}
    for p,r in rows.items():
        def aggregate(k):
            v=[x[k] for x in r if x[k] is not None]
            return mean(v) if v else None
        line={k:aggregate(k) for k in (*HARDWARE,*OPERATIONS)}
        line.update({f"peak_{k}":max(x[k] for x in r) for k in CAPACITY})
        line["n_group_samples"]=len(r)
        out[p]=line
    return out


def phase_report(root,phase):
    bundles=load(root,phase)
    output=root/"report"/phase
    output.mkdir(parents=True,exist_ok=True)
    preds=[]
    timing=defaultdict(list)
    events=[]
    timing_file=output/"timings_per_group.jsonl.gz"
    fields=("added_k_pages","selected_k_pages","remote_heads","kv_groups_with_remote")
    with gzip.open(timing_file,"wt",compresslevel=4) as stream:
        for bundle in bundles:
            for r in bundle["answers"]:
                e=r["overshoot_events"]
                answer={k:v for k,v in r.items() if k!="overshoot_events"}
                assert len(e)==answer.get("over_budget_group_steps",0)
                assert sum(x["extra_bytes"] for x in e)==answer.get("over_budget_bytes",0)
                sample=dict(sample_id=bundle["sample_id"],task=bundle["task"],**answer)
                preds.append(sample)
                events.extend(dict(sample_id=bundle["sample_id"],policy=r["policy"],**v) for v in e)
            for r in bundle["audit"]:
                timing[r["policy"]].append(r)
                stream.write(json.dumps(dict(sample_id=bundle["sample_id"],**r),
                                        separators=(",",":"),allow_nan=False)+"\n")
    lookup=defaultdict(list)
    for r in preds:
        lookup[r["policy"]].append(r)
    audits=summary_rows(timing)
    quality=[]
    for mode,rows in lookup.items():
        rec=dict(policy=mode,**{k:rows[0][k] for k in ("kind","pages","gate","window","map_id","method")})
        g=sum(x["kv_group_calls"] for x in rows)
        nt=sum(x["decode_tokens"] for x in rows)
        toks=max(nt,1)
        rec.update(prompts=len(rows),official_score=mean(r["official_score"] for r in rows),
                   complete_count=sum(r["complete_correct"] for r in rows),
                   truncated=sum(r["truncated"] for r in rows),
                   read_per_decode=mean(r["read_per_decode"] for r in rows),
                   read_per_answer=mean(r["read_per_answer"] for r in rows),
                   over_budget=sum(r.get("over_budget_group_steps",0) for r in rows),
                   total_group_steps=g,
                   over_budget_prompts=sum(r.get("over_budget_group_steps",0)>0 for r in rows),
                   over_budget_bytes=sum(r.get("over_budget_bytes",0) for r in rows),
                   peak_k_pages=max(r["peak_k_pages"] for r in rows),
                   peak_total_k_pages=max(r["peak_total_k_pages"] for r in rows),
                   peak_allocated_logits_fp32_group=max(r["peak_k_pages"] for r in rows)*32*4*4,
                   peak_allocated_logits_fp16_group=max(r["peak_k_pages"] for r in rows)*32*4*2,
                   **{f"{k}_per_group":sum(r.get(k,0) for r in rows)/max(g,1) for k in fields},
                   **{f"{k}_per_decode":sum(r.get(k,0) for r in rows)/toks for k in
                      ("scorer_mac","qk_mac","pv_mac","summary_payload","summary_padding",
                       "index_payload","index_padding","k_codes","k_scale_payload","k_zp_payload","k_padding",
                       "v_codes","v_scale_payload","v_zp_payload","v_padding","k_bytes","v_bytes")})
        if mode=="dense":
            rec["peak_allocated_logits_fp32_group"]=32768*4*4
            rec["peak_allocated_logits_fp16_group"]=32768*4*2
        rec.update({f"audit_{k}":v for k,v in audits.get(mode,{}).items()})
        quality.append(rec)
    write_csv(output/"summary.csv",quality)
    _write_json(output/"summary.json",quality)
    _write_json(output/"predictions.json",preds)
    write_csv(output/"over_budget_steps.csv",events)
    return quality


def choose_anchor(quality):
    dyn=next(x for x in quality if x["policy"]=="dynamic")
    options=[x for x in quality if x["kind"] in ("dynamic","single") and x["over_budget"]==0 and
             abs(x["read_per_decode"]/dyn["read_per_decode"]-1)<=.01 and
             x["official_score"]>=dyn["official_score"]-.01]
    chosen=min(options,key=lambda r:(-r["official_score"],-r["complete_count"],r["audit_total_group_ck"],
                                    r["read_per_decode"]))
    return Policy(chosen["kind"],chosen["pages"])


def prompt_tail(records,key):
    pp=defaultdict(list)
    for r in records:
        pp[r["sample_id"]].extend(r[key])
    return mean(float(np.quantile(vals,.99)) for vals in pp.values())


def audit_screen(root):
    bundles=load(root,"screen")
    output=root/"report"/"screen"
    output.mkdir(parents=True,exist_ok=True)
    training=bundles[:8]; valid=bundles[8:]
    m={}
    records=[dict(sample_id=b["sample_id"],**r) for b in valid for r in b["common"]]
    numerical={"token_abs_p99":prompt_tail(records,"token_abs"),"subpage_abs_p99":prompt_tail(records,"subpage_abs")}
    with gzip.open(output/"common_state.jsonl.gz","wt",compresslevel=4) as stream:
        for b in bundles:
            for r in b["common"]:
                stream.write(json.dumps(dict(sample_id=b["sample_id"],**r),separators=(",",":"))+"\n")
    with gzip.open(output/"locality_dense.jsonl.gz","wt",compresslevel=4) as stream:
        for b in bundles:
            for r in b["locality"]:
                stream.write(json.dumps(dict(sample_id=b["sample_id"],**r),separators=(",",":"))+"\n")
    costs={k:[r[k] for r in records] for k in ("base","expand4","expand8","subpage")}
    a=summary_rows(costs)
    for label in ("expand4","expand8","subpage"):
        s=a[label]
        s["over_budget"]=sum(r[label]["reads_bytes"]>r["cap"] for r in records)
        s["read_change"]=s["reads_bytes"]/a["base"]["reads_bytes"]-1
        s["memory_ck_change"]=s["lpddr_reads_and_drains_ck"]/a["base"]["lpddr_reads_and_drains_ck"]-1
        s["stage_ck_change"]=s["total_group_ck"]/a["base"]["total_group_ck"]-1
    bylayer=[np.array([r["relative"] for b in training for r in b["locality"] if r["layer"]==l]) for l in range(36)]
    model=[np.quantile(x,.95,axis=0) for x in bylayer]
    def remote_map(th):
        return (np.asarray(model)>th).tolist()
    ctrl=[]
    with gzip.open(output/"head_window_fixed_k.jsonl.gz","wt",compresslevel=3) as stream:
        for key,th in (("p95_1pct",.01),("p95_3pct",.03)):
            remote=np.asarray(remote_map(th))
            m[key]=remote.tolist()
            win_errors=[val for b in valid for r in b["locality"]
                        for val in np.asarray(r["relative"])[~remote[r["layer"]]].tolist()]
            p95=float(np.quantile(win_errors,.95)) if win_errors else None
            for kind in ("head","group"):
                row_l=[]
                for r in records:
                    hist=page_history(r["tokens"],32)
                    raw=remote[r["layer"],r["group"]]
                    w=fixed_k_window(hist,r,raw,kind)
                    row_l.append(dict(sample_id=r["sample_id"],layer=r["layer"],step=r["step"],group=r["group"],
                                      kind=kind,map_id=key,**w))
                for x in row_l:
                    stream.write(json.dumps(x,separators=(",",":"))+"\n")
                info=summary_rows({kind:row_l})[kind]
                qk_pv=lambda s: s["qk_mac"]+s["pv_mac"]+s["scorer_mac"]
                ctrl.append(dict(kind=kind,map_id=key,n_local_heads=int((~remote).sum()),
                                 n_all_local_groups=int((~remote.any(-1)).sum()),
                                 fraction_samples_without_remote=mean(r["remote_heads"]==0 for r in row_l),
                                 dense_p95_relative_in_window_heads=p95,
                                 effective_mac_reduction=1-qk_pv(info)/qk_pv(a["base"]),
                                 fixed_array_ck_reduction=1-info["fixed_array_ck"]/a["base"]["fixed_array_ck"],
                                 total_group_ck_reduction=1-info["total_group_ck"]/a["base"]["total_group_ck"],**info))
    b=a["subpage"]
    block_pass=bool(not b["over_budget"] and numerical["subpage_abs_p99"]<=
                    THRESHOLDS["b8_tail_ratio_max"]*numerical["token_abs_p99"] and
                    (b["stage_ck_change"]<=-THRESHOLDS["b8_advantage_min"] or
                     b["peak_page_or_block_probability_bytes"]+b["peak_head_drop_state_bytes"] <
                     .9*a["base"]["peak_histogram64_bytes"]))
    window_options=[r for r in ctrl if r["kind"]=="head" and r["dense_p95_relative_in_window_heads"] is not None and
                    r["dense_p95_relative_in_window_heads"]<=THRESHOLDS["window_head_rel_p95_max"] and
                    r["effective_mac_reduction"]>=THRESHOLDS["window_mac_reduction_min"]]
    window=max(window_options,key=lambda r:r["effective_mac_reduction"])["map_id"] if window_options else None
    base=a["base"]
    s=mean(r["tokens"]//32+(r["tokens"]%32>0) for r in records)*208
    scan_ratio=s/base["reads_bytes"]
    scan_time=base["summary_index_ck"]/base["lpddr_reads_and_drains_ck"]
    norm_ratio=base["normalize_ck"]/base["total_group_ck"]
    result=dict(defined_thresholds=THRESHOLDS,lpddr=LPDDR,ab=a,errors=numerical,windows=ctrl,maps=m,
                pass_B=block_pass,pass_C=window,
                expansion_pass=any(a[k]["memory_ck_change"]<-.01 and not a[k]["over_budget"]
                                   for k in ("expand4","expand8")),
                summary_byte_ratio=scan_ratio,summary_ck_ratio=scan_time,
                normalize_ck_ratio=norm_ratio,
                compact_probe=bool(scan_ratio>=.40 and scan_time>=.35),
                light_a1_needed=bool(norm_ratio>=.10),
                mapping_ids=[b["sample_id"] for b in training],screen_ids=[b["sample_id"] for b in valid],
                common_groups=len(records))
    _write_json(output/"mechanism.json",result)
    write_csv(output/"hardware_ab.csv",[dict(variant=k,**v) for k,v in a.items()])
    write_csv(output/"hardware_window.csv",ctrl)
    return result


def validation_plan(anchor,s):
    policies=[anchor]
    if s["pass_B"]:
        policies.append(replace(anchor,gate="subpage"))
    if s["pass_C"]:
        key=s["pass_C"]
        policies.extend((replace(anchor,window="head",map_id=key),replace(anchor,window="group",map_id=key)))
        if s["pass_B"]:
            policies.append(replace(anchor,gate="subpage",window="head",map_id=key))
    return policies


def finalize(root,config):
    quality=json.loads((root/"report/test/summary.json").read_text())
    pred=json.loads((root/"report/test/predictions.json").read_text())
    q={r["policy"]:r for r in quality}
    anchor=config["anchor_name"]
    dense=q["dense"]
    comparisons={}
    for candidate in config["candidates"]:
        a=q[anchor]; c=q[candidate]
        stats={m:paired_bootstrap(pred,anchor,candidate,m,confidence=.95,seed=20261203,method_key="policy")
               for m in ("official_score","complete_correct","read_per_answer","read_per_decode")}
        work=lambda r:sum(r[f"{key}_per_decode"] for key in ("qk_mac","pv_mac","scorer_mac"))
        cyc=1-c["audit_total_group_ck"]/a["audit_total_group_ck"]
        # Only allocated physical control, with common PV/score buffers.
        ctrl=lambda r:sum(r[f"audit_peak_{k}"] for k in
                          ("histogram64_bytes","head_drop_state_bytes","page_or_block_probability_bytes","token_mask_bytes"))
        ctrl_save=1-ctrl(c)/max(1,ctrl(a))
        nor=(stats["official_score"]["ci_low"]>=THRESHOLDS["score_ci_lower_min"])
        move=c["read_per_decode"]<dense["read_per_decode"]
        mac=work(c)<work(dense)
        advantaged=(cyc>=.05 or (ctrl_save>=.10 and c["audit_total_group_ck"]<=a["audit_total_group_ck"] and
                                c["read_per_decode"]<=a["read_per_decode"]))
        comparisons[candidate]=dict(paired=stats,noninferior=nor,same_candidate_reads_less=move,
                                    same_candidate_macs_less=mac,modeled_cycle_reduction=cyc,
                                    control_reduction=ctrl_save,hardware_gate=advantaged,
                                    passed=bool(nor and move and mac and advantaged and not c["over_budget"]))
    _write_json(root/"report/test/gate.json",comparisons)
    return comparisons
