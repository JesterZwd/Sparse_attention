"""Audit Round 9 addressing and independent gates from saved records."""
import argparse
import gzip
import json
from pathlib import Path

from gqa_probe.fixed_k_v import vector_burst
from gqa_probe.page_read_budget import page_history
from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DATA
from scripts.run_fused_scorer import ROOT, sources
from scripts.run_ruler_round3 import load_rows


def inspect(root,data):
    finished=json.loads((root/"finished.json").read_text())
    out={}
    for phase in ("fit","refine","validation","independent"):
        conf=root/phase/"config.json"
        if not conf.is_file():
            continue
        config=json.loads(conf.read_text())
        receipt=json.loads((root/phase/"complete.json").read_text())
        split="test" if phase=="independent" else "calibration"
        assert config["manifest_sha256"]==_sha(data/split/"manifest.json")
        assert config["sources"]==sources() and receipt["config_sha256"]==_sha(conf)
        if phase=="independent":
            freeze=root/"frozen_test.json"
            assert finished["freeze_sha256"]==_sha(freeze)
            pinned=json.loads(freeze.read_text())
            assert all(pinned[k]==config[k] for k in ("ids","sources","policies","manifest_sha256"))
            assert len(pinned["ids"])==128
        _,_,full=load_rows(data/split,split)
        inputs={r["sample_id"]:r["input_tokens"] for r in full}
        budgets={p["name"]:p["budget"] for p in config["policies"]}
        folder=root/"report"/phase
        preds={(r["sample_id"],r["policy"]):r for r in json.loads((folder/"predictions.json").read_text())}
        for name,sha in receipt["files"].items():
            assert sha==_sha(root/phase/"bundles"/name)
        n_rows,pairs=0,set()
        archive=folder/"group_steps.jsonl.gz"
        per_prompt=[]
        with gzip.open(archive,"rt") as f:
            for line in f:
                row=json.loads(line)
                s,p=row["sample_id"],row["policy"]
                assert (s,p) not in pairs
                pairs.add((s,p))
                accepted=byte_total=fallback=max_reads=0
                min_headroom=None
                for step,layer,cap,groups in row["steps"]:
                    h=page_history(inputs[s]+step+1,32)
                    assert cap==2*((h.pages-1)*vector_burst(32)+vector_burst(h.tail_size))*budgets[p]//100
                    for read,k,kr,vr,back,depth,directory,tags in groups:
                        assert read%32==0 and read<=cap and len(h.protected)<=depth<=k
                        assert kr>0 and vr>0 and not directory and not tags
                        accepted+=1
                        byte_total+=read
                        fallback+=back
                        max_reads=max(max_reads,read)
                        gap=cap-read
                        min_headroom=gap if min_headroom is None else min(min_headroom,gap)
                score=preds[s,p]
                assert (accepted,byte_total,fallback)==(score["kv_group_calls"],score["read"],score["fallback_group_steps"])
                per_prompt.append(dict(sample_id=s,policy=p,checked_groups=accepted,read_bytes=byte_total,
                                       fallback_groups=fallback,peak_read_bytes=max_reads,
                                       least_cap_headroom_bytes=min_headroom,over_budget=0))
                n_rows+=accepted
        assert pairs=={k for k,v in preds.items() if v["kv_group_calls"]>0}
        table=folder/"per_prompt_budget.csv"
        write_csv(table,per_prompt)
        out[phase]=dict(prompts=len(config["ids"]),predictions=len(preds),group_steps=n_rows,
                        raw_group_steps_sha256=_sha(archive),per_prompt_budget_sha256=_sha(table))
    if finished["heldout"]:
        assert all(r["zero_overruns"] for r in finished["comparisons"].values())
        assert finished["passes"]==any(r["passes"] for r in finished["comparisons"].values())
    out=dict(phases=out,zero_overruns=True,heldout=finished["heldout"],
             finished_sha256=_sha(root/"finished.json"))
    _write_json(root/"verification.json",out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    print(json.dumps(inspect(a.root,a.data)))
