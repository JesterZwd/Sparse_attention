"""Reconcile saved A2 per-group reads against prompt totals and dense layout."""
import argparse
import gzip
import json
from pathlib import Path

from gqa_probe.fixed_k_v import vector_burst
from gqa_probe.page_read_budget import page_history
from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.probe_pagepack import ROOT
from scripts.run_pagepack import source_hashes
from scripts.round7_run import DATA
from scripts.run_ruler_round3 import load_rows


def verify(root,data):
    result=json.loads((root/"finished.json").read_text())
    counts={}
    for phase in ("cap5","fixed_pages","cap7","independent"):
        if not (root/phase/"complete.json").exists():
            continue
        split="test" if phase=="independent" else "calibration"
        _,_,rows=load_rows(data/split,split)
        known={r["sample_id"]:r["input_tokens"] for r in rows}
        setup_path=root/phase/"config.json"
        setup=json.loads(setup_path.read_text())
        receipt=json.loads((root/phase/"complete.json").read_text())
        if (receipt["config_sha256"]!=_sha(setup_path) or
                setup["sources"]!=source_hashes() or setup["manifest_sha256"]!=_sha(data/split/"manifest.json")):
            raise ValueError("closure hashes changed")
        policies={p["name"]:p["budget"] for p in setup["policies"]}
        if setup["support"]:
            policies[setup["support"]["name"]]=10
        report=root/"report"/phase
        preds={(r["sample_id"],r["policy"]):r for r in json.loads((report/"predictions.json").read_text())}
        checked=0
        pairs=set()
        with gzip.open(report/"group_steps.jsonl.gz","rt") as f:
            for line in f:
                item=json.loads(line)
                ident,policy=item["sample_id"],item["policy"]
                key=(ident,policy)
                if key in pairs:
                    raise ValueError("duplicate saved group ledger")
                pairs.add(key)
                total=n=back=0
                for step,layer,cap,groups in item["steps"]:
                    h=page_history(known[ident]+step+1,32)
                    expected=2*((h.pages-1)*vector_burst(32)+vector_burst(h.tail_size))*policies[policy]//100
                    if cap!=expected:
                        raise ValueError("wrong per-step cap")
                    for read,pages,k,v,f,prefix,directory,tag in groups:
                        if not (read%32==0 and read<=cap and
                                len(h.protected)<=prefix<=pages and k>0 and v>0 and
                                directory%32==0 and tag%32==0):
                            raise ValueError("per-step physical constraints violated")
                        total+=read
                        n+=1
                        back+=f
                r=preds[key]
                if (total,n,back)!=(r["read"],r["kv_group_calls"],r["fallback_group_steps"]):
                    raise ValueError("group and prompt ledgers disagree")
                checked+=n
        sparse={k for k,r in preds.items() if k[1] in policies and r["kv_group_calls"]>0}
        if pairs!=sparse:
            raise ValueError("budget replay missing routes")
        counts[phase]=dict(predictions=len(preds),checked_groups=checked,prompts=len(set(k[0] for k in preds)))
    if result["heldout"]!=("independent" in counts):
        raise ValueError("independent status contradicts receipt")
    output=dict(zero_over_budget=True,counts=counts,heldout=result["heldout"],
                probe_config_sha256=_sha(root/"probe_config.json"),
                finished_sha256=_sha(root/"finished.json"))
    _write_json(root/"verification.json",output)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    print(json.dumps(verify(a.root,a.data)))
