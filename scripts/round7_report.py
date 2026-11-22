"""Export prompt-paired Round 7 results and fixed-state cost tables."""
from collections import defaultdict
import gzip
import json
from statistics import mean

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.round7_sides import summarize_side_screens
from gqa_probe.ruler_experiment import _write_json
from scripts.report_round6 import load
from scripts.report_ruler_round3 import write_csv


def div(a,b):
    return a/b if b else None


def by(rows,key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def export_gz(path,rows):
    with gzip.open(path,"wt",compresslevel=4) as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")


def mean_fields(rows,fields):
    return {k: mean(vals) if (vals := [r[k] for r in rows if r.get(k) is not None]) else None
            for k in fields}


TIMING = ("reads_bytes","bursts","before_first_v_ck","total_group_ck","lpddr_reads_and_drains_ck",
          "gate_ck","v_runs","row_hits","activations","score_ck","budget_check_ck","qk_mac","pv_mac",
          "scorer_mac","gate_exponentials","gate_comparisons","gate_reduce_adds","histogram_updates",
          "histogram_bin_visits","logits_sram_reads","logits_sram_writes","mask_sram_reads","mask_sram_writes")
PEAKS = ("control_sram_bytes","logits_sram_bytes","histogram_bytes")
TRAFFIC = ("summary_payload","summary_padding","index_payload","index_padding","k_codes","k_scale_payload",
           "k_zp_payload","k_padding","v_codes","v_scale_payload","v_zp_payload","v_padding",
           "summary_build_write_bytes","summary_prefill_build_write_bytes","summary_append_channel_adds",
           "summary_append_extrema_comparisons","k_bytes","v_bytes","qk_mac","pv_mac","scorer_mac")


def quality_rows(answers,timings):
    rows = []
    for policy,preds in by(answers,"policy").items():
        cost = timings.get(policy,[])
        nt = sum(r["decode_tokens"] for r in preds)
        ng = sum(r["kv_group_calls"] for r in preds)
        traffic = {f"{k}_per_decode":div(sum(r.get(k,0) for r in preds),nt) for k in TRAFFIC}
        result = dict(policy=policy,n=len(preds),official_score=mean(r["official_score"] for r in preds),
                      complete=sum(r["complete_correct"] for r in preds),
                      read_mb_per_decode=div(sum(r["read"] for r in preds),nt*1e6),
                      read_gb_per_answer=mean(r["read"] for r in preds)/1e9,
                      mean_decodes=mean(r["decode_tokens"] for r in preds),
                      over_budget=sum(r.get("over_budget_group_steps",0) for r in preds),
                      group_steps=ng,fallback=sum(r.get("fallback_group_steps",0) for r in preds),
                      k_evidence_page_recall=div(sum(r.get("evidence_k_page_hits",0) for r in preds),
                                                 sum(r.get("evidence_k_page_total",0) for r in preds)),
                      v_evidence_page_recall=div(sum(r["gold_page_hits"] for r in preds),
                                                 sum(r["gold_page_total"] for r in preds)),
                      mean_k_pages=div(sum(r.get("k_pages",0) for r in preds),ng),
                      mean_added_pages=div(sum(r.get("added_k_pages",0) for r in preds),ng),
                      peak_k_pages=max(r["peak_k_pages"] for r in preds),sampled_groups=len(cost),
                      lost_all_mean=div(sum(r.get("lost_all_sum",0) for r in preds),ng*4)
                      if "lost_all_sum" in preds[0] else None,
                      lost_prefix_mean=div(sum(r.get("lost_prefix_sum",0) for r in preds),ng*4)
                      if "lost_prefix_sum" in preds[0] else None,
                      **{k:max((r.get(k,0) for r in preds),default=0) for k in ("lost_all_max","lost_prefix_max")},
                      **traffic,
                      **{f"mean_{k}":v for k,v in mean_fields(cost,TIMING).items()},
                      **{f"peak_{k}":max((r.get(k,0) for r in cost),default=0) for k in PEAKS})
        rows.append(result)
    return rows


def mean_prompt_percentile(rows,field,q):
    vals = [[e for r in p for e in r[field] if e is not None]
            for p in by(rows,"sample_id").values()]
    vals = [float(np.quantile(v,q)) for v in vals if v]
    return mean(vals) if vals else None


def diagnostics(rows,key,fields):
    result = []
    for name,method in by(rows,key).items():
        for cohort in ("all","answer_start"):
            subset = method if cohort == "all" else [x for x in method if x["answer_start"]]
            if not subset:
                continue
            cost = mean_fields(subset,TIMING) if key == "policy" else {}
            pairs = ("evidence_v_pages","evidence_total") if key == "policy" else ("evidence_selected","evidence_total")
            loss = {f"{field}_p{int(q*100)}":mean_prompt_percentile(subset,field,q) for field in fields
                    for q in (.95,.99)}
            if key == "policy":
                cost.update(mean_fields(subset,("v_bytes","prefix_pages","k_pages","before_fallback_over_bytes")))
                cost.update(fallback_rate=mean(r["fallback"] for r in subset),
                            beyond_two_percent_all_k=sum(p>.020003 for r in subset for p in r["lost_read_k"]),
                            max_lost_read_k=max(p for r in subset for p in r["lost_read_k"]),
                            max_lost_prefix=max(p for r in subset for p in r["lost_prefix"]))
                cost.update({f"peak_{k}": max(r[k] for r in subset) for k in PEAKS})
            else:
                loss["mean_head_coverage"]=mean(p for r in subset for p in r["head_coverage"])
            result.append({key:name,"cohort":cohort,"states":len(subset),
                           "evidence_recall":div(sum(x[pairs[0]] for x in subset),sum(x[pairs[1]] for x in subset)),
                           **loss,**cost})
    return result


def phase_report(root,phase):
    bundles = load(root,phase)
    output = root/"report"/phase
    output.mkdir(parents=True,exist_ok=True)
    outputs={}
    for source in ("answers","timings","summary_diag","gate_diag","group_steps"):
        outputs[source]=[dict(sample_id=b["sample_id"],task=b["task"],**r)
                         for b in bundles for r in b[source]]
    answers = outputs["answers"]
    _write_json(output/"predictions.json",answers)
    rows = quality_rows(answers,by(outputs["timings"],"policy"))
    write_csv(output/"quality_and_cost.csv",rows)
    for name in ("timings","summary_diag","gate_diag","group_steps"):
        if outputs[name]:
            export_gz(output/f"{name}.jsonl.gz",outputs[name])
    if outputs["summary_diag"]:
        write_csv(output/"fixed_pages.csv",diagnostics(outputs["summary_diag"],"method",
                  ("head_output_absolute_errors","head_output_relative_errors")))
    candidate_rows=[]
    if outputs["gate_diag"]:
        candidate_rows=diagnostics(outputs["gate_diag"],"policy",("head_abs","head_rel","head_dense_abs",
                                   "lost_read_k","lost_prefix"))
        write_csv(output/"fixed_k_gates.csv",candidate_rows)
    sides = [b["sides"] for b in bundles if b["sides"]]
    if sides:
        export_gz(output/"reuse.jsonl.gz",[dict(sample_id=b["sample_id"],**r) for b in bundles
                  if b["sides"] for r in b["sides"]["rows"]])
        export_gz(output/"cold.jsonl.gz",[dict(sample_id=b["sample_id"],**r) for b in bundles
                  if b["sides"] for r in b["sides"]["cold"]])
        _write_json(output/"side_summary.json",summarize_side_screens(sides))
    return rows,candidate_rows,answers


def pairs(output,rows,baseline,candidate):
    r = {k:paired_bootstrap(rows,baseline,candidate,k,confidence=.95,method_key="policy")
         for k in ("official_score","read","read_per_decode")}
    a={p["sample_id"]:p for p in rows if p["policy"]==baseline}
    b={p["sample_id"]:p for p in rows if p["policy"]==candidate}
    r["prompts"]=[dict(sample_id=s,score_delta=b[s]["official_score"]-x["official_score"],
                       complete_base=x["complete_correct"],complete_candidate=b[s]["complete_correct"],
                       read_bytes_base=x["read"],read_bytes_candidate=b[s]["read"])
                  for s,x in a.items()]
    _write_json(output,r)
    return r


def checkpoint(root, baseline, compact, gate, receipt, closure=None):
    def percent(x):
        return f"{100*x:.2f}%" if x is not None else "—"

    def table(rows):
        lines = ["| 策略 | 官方分 | 完整命中 | MB/解码 token | GB/答案 | 回退率 | 超限组步 | 含依赖 CK/组 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            ck=r["mean_total_group_ck"]
            lines.append(f"| {r['policy']} | {percent(r['official_score'])} | {r['complete']}/{r['n']} | "
                         f"{r['read_mb_per_decode']:.3f} | {r['read_gb_per_answer']:.3f} | "
                         f"{percent(div(r['fallback'],r['group_steps']))} | {r['over_budget']:,} | "
                         f"{round(ck) if ck is not None else '—'} |")
        return lines

    lines = ["# Round 7 预算与 V 门控", "",
             "搬运按 affine32 packed burst 计费。周期来自 LPDDR5 和选择器依赖模型。"
             "组步预算检查覆盖全部解码，周期、缓存和门控操作数按共同定义抽样。", "",
             "## 10% 单遍与动态补 K", "", *table(baseline), "",
             "[逐提示与成本账](baseline/quality_and_cost.csv)", "",
             "## 5% 144 B 摘要", "", *table(compact), "",
             "[等页数 Oracle 和摘要诊断](screen/fixed_pages.csv) · [等上限闭环](compact/quality_and_cost.csv)", "",
             "## 固定 K 的门控", "", *table(gate), "",
             "[固定 K 误差与硬件账](screen/fixed_k_gates.csv) · [摘要复用和冷背景](screen/side_summary.json)", "",
             "## 判定", ""]
    decision=json.loads((root/"calibration_decision.json").read_text())
    lines.append(f"校准冻结峰值阈值 λ={receipt['blasst_lambda']:g}，无直方图阈值 τ={receipt['checked_tau']:g}。"
                 f"强基线为 `{decision['anchor']}`。")
    if closure:
        lines += ["", *table(closure), "",
                  f"正式候选为 `{decision['selected']}`。官方分差的 95% 配对区间下界为 "
                  f"{receipt['score_ci_lower']*100:+.3f} 个百分点。"
                  f"依赖周期降幅 {percent(receipt['cycle_reduction'])}，控制 SRAM 降幅 "
                  f"{percent(receipt['control_sram_reduction'])}。"
                  f"推进门槛{'通过' if receipt['passed'] else '未通过'}。", "",
                  "[独立逐提示配对](test/main_pair.json) · [相对 Dense](test/dense_pair.json)"]
    else:
        lines.append("校准阶段没有满足冻结条件的新策略，128 条独立集保持封存。")
    (root/"report"/"checkpoint.md").write_text("\n".join(lines)+"\n")
