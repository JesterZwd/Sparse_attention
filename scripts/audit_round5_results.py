"""Verify Round 5 receipts and cross-tabulate BLASST tail observations."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from gqa_probe.ruler_experiment import _sha,_write_json
from scripts.report_round5_shared_refill import load_cohort,write_csv
from scripts.round5_shared_refill import ROOT,code_hashes
from scripts.round5_mechanism import CASES


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    args=p.parse_args()
    root=args.root
    f=root/"frozen_test.json"
    info=json.loads(f.read_text())
    done=json.loads((root/"finished.json").read_text())
    assert done["frozen_sha256"]==_sha(f) and info["sources"]==code_hashes()
    assert info["test_manifest_sha256"]==_sha(root/"data/test/manifest.json")
    assert all(sha==_sha(root/phase/"complete.json") for phase,sha in info["stage_receipts"].items())
    assert all(sha==_sha(root/"report"/phase/"summary.json") for phase,sha in info["stage_reports"].items())
    counts={}
    for phase in ("cal1","cal2","test"):
        b=load_cohort(root,phase)
        counts[phase]=dict(prompts=len(b),predictions=sum(len(x["predictions"]) for x in b),
                           group_records=sum(len(x["groups"]) for x in b))
    report=root/"report/test"
    routes=json.loads((report/"paired_and_gate.json").read_text())
    assert done["advanced"]==routes["advanced"]
    rows=json.loads((report/"predictions.json").read_text())
    hit={(r["policy"],r["sample_id"]):r["reference_hits"] for r in rows}
    per=defaultdict(list)
    path=report/"per_group.jsonl.gz"
    with gzip.open(path,"rt") as stream:
        for line in stream:
            row=json.loads(line)
            if row["policy"].startswith("blasst"):
                per[row["basis"],row["policy"],row["sample_id"]].append(row)
    assoc=[]
    for (basis,policy,pk),rs in sorted(per.items()):
        hits=hit[policy,pk]
        failed={i for i,val in enumerate(hits) if not val}
        for cutoff in (.05,.10,.20):
            tail=[r for r in rs if max(r["final_discarded_mass"])>cutoff]
            missing_evidence=set().union(*(r["gold_targets_with_selected_k_page_omitted_in_v"] for r in tail))
            assoc.append(dict(basis=basis,policy=policy,sample_id=pk,threshold=cutoff,
                              sampled_groups=len(rs),tail_groups=len(tail),missing_task_values=len(failed),
                              tail_selected_k_evidence_omitted_from_v=len(missing_evidence),
                              same_values_missed=len(failed&missing_evidence)))
    write_csv(report/"blasst_tail_by_threshold_and_value.csv",assoc)
    cases=[]
    for tag in CASES:
        file=root/"mechanism_verified"/f"{tag}.json"
        data=json.loads(file.read_text())
        onset=next(r for r in data["events"] if "missing_value_onset" in r["events"])
        iv=onset["intervention"]
        assert iv["control"]["target_hit"]
        peak=max(max(x["page_mass_by_head"]) for x in onset["common_qkv_evidence"]
                 if x["policy"]=="refill" and x["stage"]=="added" and x["v_retained"])
        cases.append(dict(case=tag,step=onset["at"]-1,changed_layer_group_pairs=len(iv["swaps"]),
                          peak_added_evidence_mass=peak,control_hit=iv["control"]["target_hit"],
                          replacement_hit=iv["replacement"]["target_hit"],sha256=_sha(file)))
    write_csv(root/"report/mechanism_summary.csv",cases)
    _write_json(root/"report/verification.json",dict(counts=counts,receipt_sha256=_sha(root/"finished.json"),
                frozen_sha256=_sha(f),groups_sha256=_sha(path),sources_verified=True,advanced=routes["advanced"]))
    for basis,p in sorted({(x["basis"],x["policy"]) for x in assoc}):
        for threshold in (.05,.1,.2):
            rr=[x for x in assoc if (x["basis"],x["policy"],x["threshold"])==(basis,p,threshold)]
            print(basis,p,threshold,"tail groups",sum(x["tail_groups"] for x in rr),
                  "missed values with omitted evidence",sum(x["same_values_missed"] for x in rr))
    print(counts)


if __name__=="__main__":
    main()
