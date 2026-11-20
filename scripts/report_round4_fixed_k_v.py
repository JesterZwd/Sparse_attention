"""Reduce fixed-K shadow traces, replay results and paired closed-loop scores."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.vskip_attention import VPolicy
from scripts.report_ruler_round3 import write_csv


def load(cohort):
    receipt=json.loads((cohort/"complete.json").read_text())
    if _sha(cohort/"config.json")!=receipt["config_sha256"]:
        raise ValueError(f"config checksum mismatch in {cohort}")
    expected=json.loads((cohort/"config.json").read_text())["ids"]
    bundles=[]
    for name, digest in receipt["prompts"].items():
        path=cohort/"bundles"/name
        if _sha(path)!=digest:
            raise ValueError(f"prompt checksum mismatch: {path}")
        with gzip.open(path,"rt") as stream:
            bundles.append(json.load(stream))
    if sorted(expected)!=sorted(b["sample_id"] for b in bundles) or len(expected)!=receipt["n"]:
        raise ValueError("missing subject")
    return bundles


def summarize(cohort, output):
    bundles=load(cohort)
    output.mkdir(parents=True,exist_ok=True)
    paths=defaultdict(list)
    curve=defaultdict(list)
    schedules=defaultdict(list)
    residues=[]
    diagnostic=None
    path_rows=[]
    same_state=[]
    for b in bundles:
        for p in b["paths"]:
            rows=dict(sample_id=b["sample_id"],task=b["task"],**p)
            if p["zero_max_error"] not in (None,0.):
                raise ValueError("epsilon=0 output mismatch")
            path_rows.append(rows)
            paths[p["policy"]].append(rows)
        per_prompt=defaultdict(list)
        for r in b["curve"]:
            per_prompt[(r["mode"],r["threshold"])].append(r)
        base=per_prompt.get(("all",0))
        for (name,e), rr in per_prompt.items():
            abs_err=[x for r in rr for x in r["absolute"]]
            rel_err=[x for r in rr for x in r["relative"]]
            assert len(rr)==len(base)
            assert all(x["k_bytes"]==y["k_bytes"] for x,y in zip(rr,base))
            if name in ("mass","sparta") or name.startswith("micro"):
                assert max(r["lost_max"] for r in rr) <= e+3e-6
            record=dict(sample_id=b["sample_id"],mode=name,threshold=e,groups=len(rr),
                        v_saved=1-sum(r["v_bytes"] for r in rr)/sum(r["v_bytes"] for r in base),
                        total_saved=1-sum(r["total_bytes"] for r in rr)/sum(r["total_bytes"] for r in base),
                        pv_saved=1-sum(r["pv_mac"] for r in rr)/sum(r["pv_mac"] for r in base),
                        v_page_fraction_saved=1-sum(r["v_pages"] for r in rr)/sum(r["k_pages"] for r in rr),
                        protected_page_fraction=sum(r["protected_pages"] for r in rr)/sum(r["k_pages"] for r in rr),
                        worst_loss=max(r["lost_max"] for r in rr),
                        absolute_p95=float(np.quantile(abs_err,.95)),absolute_p99=float(np.quantile(abs_err,.99)),
                        relative_p95=float(np.quantile(rel_err,.95)),relative_p99=float(np.quantile(rel_err,.99)))
            curve[name,e].append(record)
            same_state.extend(dict(sample_id=b["sample_id"],**r) for r in rr)
        for r in b["schedule"]:
            schedules[r["method"]].append(dict(sample_id=b["sample_id"],**r))
        residues.extend(dict(sample_id=b["sample_id"],**r) for r in b["progressive"])
        if "diagnostic" in b:
            diagnostic=b["diagnostic"]
    quality=[]
    for key,rs in paths.items():
        example=rs[0]
        quality.append(dict(policy=key,scorer=example["scorer"],page=example["page"],percent=example["percent"],
                            mode=example["mode"],threshold=example["threshold"],equal_pages=example["equal_pages"],
                            prompts=len(rs),score=mean(r["official_score"] for r in rs),
                            complete=mean(r["complete_correct"] for r in rs),
                            truncated=sum(r["truncated"] for r in rs),
                            v_bytes=mean(r["v_bytes"] for r in rs),
                            read_per_answer=mean(r["read_bytes"] for r in rs),
                            read_per_decode=mean(r["read_bytes"]/max(1,r["decode_tokens"]) for r in rs),
                            gold_page_recall=mean(v) if (v:=[r["gold_page_recall"] for r in rs if r["gold_page_recall"] is not None]) else None,
                            pv_mac=mean(r["pv_mac"] for r in rs)))
    curves=[]
    for (name,e),rs in sorted(curve.items()):
        fields=("v_saved","total_saved","pv_saved","v_page_fraction_saved","protected_page_fraction",
                "absolute_p95","absolute_p99","relative_p95","relative_p99")
        curves.append(dict(mode=name,threshold=e,prompts=len(rs),
                           groups=sum(r["groups"] for r in rs),worst_loss=max(r["worst_loss"] for r in rs),
                           **{key:mean(r[key] for r in rs) for key in fields}))
    schedule_summary=[]
    for name,rs in schedules.items():
        payments=sum(r["transactions"] for r in rs)
        schedule_summary.append(dict(mode=name,observations=len(rs),transactions=payments,
                                     row_hit=sum(r["row_hits"] for r in rs)/payments,
                                     activates=sum(r["activations"] for r in rs),
                                     mean_first_v_wait_k_bursts=mean(r["k_bursts_before_first_v"] for r in rs),
                                     mean_k_v_barriers=mean(r["barriers"] for r in rs),
                                     max_page_mass_table=max(r["page_mass_table_bytes"] for r in rs),
                                     max_fp16_logit_storage=max(r["logits_fp16_bytes"] for r in rs),
                                     max_fp32_acc_sram=max(r["accumulator_fp32_bytes"] for r in rs),
                                     fp32_exact_logit_storage=max(r["logits_fp16_bytes"] for r in rs)*2))
    residue_summary={}
    if residues:
        if not all(r["exact"] for r in residues):
            raise ValueError("progressive exact ranking certificate failed")
        residue_summary=dict(observations=len(residues),fallback=mean(r["fallback"] for r in residues),
                             residual_rate=sum(r["residual_pages"] for r in residues)/sum(r["pages"] for r in residues),
                             mean_burst_ratio_to_int8=mean((r["coarse_burst_bytes"]+r["residual_burst_bytes"])/r["int8_burst_bytes"] for r in residues))
        write_csv(output/"progressive.csv",residues)
    write_csv(output/"quality.csv",quality)
    write_csv(output/"fixed_k_curve.csv",curves)
    write_csv(output/"fixed_k_per_prompt.csv",[r for rs in curve.values() for r in rs])
    write_csv(output/"schedule.csv",schedule_summary)
    with gzip.open(output/"per_group.jsonl.gz","wt",compresslevel=4) as stream:
        for row in same_state:
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    _write_json(output/"predictions.json",path_rows)
    _write_json(output/"summary.json",dict(quality=quality,curves=curves,schedule=schedule_summary,
                                          progressive=residue_summary,prompts=len(bundles)))
    if diagnostic:
        _write_json(output/"0006_diagnostic.json",diagnostic)
    return quality,curves


def choose_workpoint(root):
    for cohort in ("32k_mv","16k_fwe"):
        path=root/"report"/cohort/"summary.json"
        if not path.is_file():
            continue
        report=json.loads(path.read_text())
        rows={x["policy"]:x for x in report["quality"]}
        dense=rows[VPolicy("dense").name]["score"]
        for budget in (10,8):
            q8=rows[VPolicy("q8",percent=budget).name]["score"]
            if dense>=.90 and .78 <= q8 < .995 and q8 >= dense-.15:
                return dict(cohort=cohort,percent=budget,dense_score=dense,q8_score=q8)
    return None


def compare_test(root):
    info=json.loads((root/"frozen_test.json").read_text())
    rows=json.loads((root/"report/test/predictions.json").read_text())
    lookup={p["policy"]:p for p in json.loads((root/"report/test/summary.json").read_text())["quality"]}
    comparisons={}
    for label,(base,candidate) in info["comparisons"].items():
        pair={metric:paired_bootstrap(rows,base,candidate,metric,confidence=.95,method_key="policy")
              for metric in ("official_score","complete_correct","read_bytes")}
        b,c=lookup[base],lookup[candidate]
        paired_b={x["sample_id"]:x["official_score"] for x in rows if x["policy"]==base}
        paired_c={x["sample_id"]:x["official_score"] for x in rows if x["policy"]==candidate}
        d=[paired_c[k]-v for k,v in paired_b.items()]
        pair.update(score_wins=sum(x>0 for x in d),score_losses=sum(x<0 for x in d),score_ties=sum(x==0 for x in d),
                    mean_per_decode_reduction=1-c["read_per_decode"]/b["read_per_decode"],
                    mean_per_answer_reduction=1-c["read_per_answer"]/b["read_per_answer"])
        comparisons[label]=pair
    _write_json(root/"report/test/paired.json",comparisons)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("cohort")
    a=p.parse_args()
    summarize(a.root/a.cohort,a.root/"report"/a.cohort)
    if a.cohort=="test":
        compare_test(a.root)


if __name__=="__main__":
    main()
