"""Summarize four-tier fixed-K replay rows emitted by the common-state probe."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.range_gate import LIMITS, TAUS
from gqa_probe.ruler_experiment import _write_json


POLICIES = ("hist64_full", "hist64_hist64", "tiers2_full", "tiers234_full")
MEANS = ("read_bytes", "v_bytes", "banked_cycle", "banked_first_v_command",
         "first_gate_ck", "retry_ck", "fallback_ck", "pv_mac", "qk_mac",
         "mass_scans", "mask_scans", "budget_checks", "gate_exponentials")
PEAKS = ("control_sram_bytes", "logits_sram_bytes", "attention_sram_bytes")
PAIRS = (("hist64_full", "tiers2_full"), ("hist64_full", "tiers234_full"),
         ("tiers2_full", "tiers234_full"), ("hist64_hist64", "tiers234_full"))


def summarize_rows(rows):
    states = defaultdict(dict)
    for row in rows:
        key = (row["sample_id"], row["step"], row["layer"], row["group"])
        routes = states[key]
        if row["policy"] not in POLICIES or row["policy"] in routes:
            raise ValueError(f"unknown or duplicate gate route at {key}")
        routes[row["policy"]] = row
    if not states:
        raise ValueError("no four-tier replay rows")
    for key, routes in states.items():
        if set(routes) != set(POLICIES):
            raise ValueError(f"unpaired gate routes at {key}")
        ref = routes["hist64_full"]
        for r in routes.values():
            if (r["initial_ids"] != ref["initial_ids"] or
                    r["read_bytes"] > r["cap"] or r["read_bytes"] != 32*r["banked_bursts"]):
                raise ValueError(f"candidate K or physical budget mismatch at {key}")

    summary = []
    prompts = []
    for policy in POLICIES:
        group_rows = [routes[policy] for routes in states.values()]
        by_prompt = defaultdict(list)
        for r in group_rows:
            by_prompt[r["sample_id"]].append(r)

        tails = {}
        for sample, observations in by_prompt.items():
            record = dict(policy=policy, sample_id=sample, task="niah_multivalue")
            record.update({metric: mean(r[metric] for r in observations)
                           for metric in (*MEANS, "fallback")})
            for key in ("abs_error", "relative_error", "lost_all_k", "lost_prefix"):
                scores = [x for r in observations for x in r[key]]
                for pct in (95,99):
                    record[f"{key}_p{pct}"] = float(np.quantile(scores,pct/100))
            prompts.append(record)
            for key,value in record.items():
                if key.endswith(("_p95","_p99")):
                    tails.setdefault(key, []).append(value)

        answer_start = [r for r in group_rows if r.get("answer_start")]
        evidence = sum(r["evidence"] for r in answer_start)
        whole_losses = [p for r in group_rows for p in r["lost_all_k"]]
        fallback = sum(r["fallback"] for r in group_rows)
        row = dict(policy=policy, states=len(states), prompts=len(by_prompt),
                   fallback_states=fallback, fallback_rate=fallback/len(states),
                   initial_over_budget_states=sum(r["initial_excess_bytes"]>0 for r in group_rows),
                   rescued_at_3pct=sum(r["accepted_limit"]==.03 for r in group_rows),
                   rescued_at_4pct=sum(r["accepted_limit"]==.04 for r in group_rows),
                   same_cutoff_retry_count=sum(len(r["limits_tried"])-r["mask_scans"] for r in group_rows),
                   full_k_drop_over_2pct_heads=sum(p > .02+3e-6 for p in whole_losses),
                   full_k_drop_over_4pct_heads=sum(p > .04+3e-6 for p in whole_losses),
                   total_heads=len(whole_losses),
                   answer_start_v_evidence_recall=sum(r["v_evidence"] for r in answer_start)/evidence
                   if evidence else None)
        row.update({f"mean_{key}":mean(r[key] for r in group_rows) for key in MEANS})
        row.update({f"peak_{key}":max(r[key] for r in group_rows) for key in PEAKS})
        row.update({key:mean(values) for key,values in tails.items()})
        summary.append(row)

    paired = []
    for base,candidate in PAIRS:
        metrics = ("banked_cycle","read_bytes","pv_mac","abs_error_p99","relative_error_p99")
        paired.append(dict(base=base,candidate=candidate,**{
            metric:paired_bootstrap(prompts,base,candidate,metric,confidence=.95,method_key="policy")
            for metric in metrics}))
    return dict(common_states=len(states), taus=list(TAUS), limits=list(LIMITS),
                address_mode="direct", rows=summary, per_prompt=prompts, paired=paired)


def records(path):
    open_file = gzip.open if path.suffix==".gz" else open
    with open_file(path,"rt") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",nargs="+",type=Path,help="Raw JSONL or JSONL.GZ four-tier rows")
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    summary=summarize_rows(r for path in args.input for r in records(path))
    _write_json(args.output,summary)


if __name__=="__main__":
    main()
