"""Check receipts and export comparable baseline, mechanism and buffer ledgers."""
import json
from pathlib import Path

from gqa_probe.fixed_k_v import byte_quota
from gqa_probe.ruler_experiment import _sha,_write_json
from gqa_probe.shared_refill import quota
from scripts.report_round6 import load,write_csv
from scripts.round6_subpage_head import ROOT,sources
from scripts.run_ruler_round3 import load_rows


def allocated(r,t):
    kind=r["kind"]
    if kind=="dense":
        return t
    if kind=="single":
        return r["pages"]*32
    if kind=="compact":
        return byte_quota(t,32,r["method"],5)*32
    h,_,n=quota(t,1000)
    if kind=="dynamic":
        return (n+min(h.pages-n,(n-len(h.protected))//2+1))*32
    return n*32


def main():
    root=ROOT
    protocol=json.loads((root/"protocol.json").read_text())
    assert protocol["source_sha256"]==sources()
    assert all(_sha(root/"data"/p/"manifest.json")==h for p,h in protocol["data_manifests"].items())
    receipt=json.loads((root/"finished.json").read_text())
    screen=json.loads((root/"report/screen/mechanism.json").read_text())
    assert receipt["status"]=="screened_out" and not receipt["passed"] and not (root/"test/complete.json").exists()
    assert not screen["pass_B"] and screen["pass_C"] is None
    counts={}
    for p in ("baseline","screen","compact"):
        bs=load(root,p)
        counts[p]=dict(prompts=len(bs),predictions=sum(len(b["answers"]) for b in bs),
                       groups=sum(len(b["audit"]) for b in bs),
                       dense_common_rows=sum(len(b["common"]) for b in bs))
    _,_,c=load_rows(root/"data/calibration","calibration")
    histories={r["sample_id"]:r["input_tokens"] for r in c}
    sizes=[]
    cost=[]
    out=root/"report"
    for phase in ("baseline","compact"):
        loc=out/phase
        rows=json.loads((loc/"predictions.json").read_text())
        summary=json.loads((loc/"summary.json").read_text())
        for r in summary:
            rs=[a for a in rows if a["policy"]==r["policy"]]
            device_padded=max(allocated(a,t) for a in rs for t in
                              range(histories[a["sample_id"]]+1,histories[a["sample_id"]]+a["decode_tokens"]+1))
            sizes.append(dict(cohort=phase,policy=r["policy"],peak_padded_token_slots_group=device_padded,
                              padded_fp16_logits_bytes_group=device_padded*4*2,
                              padded_fp32_logits_bytes_group=device_padded*4*4,
                              peak_effective_fp32_group=r.get("audit_peak_peak_logits_bytes_fp32"),
                              pm_peak_k_pages=r["peak_k_pages"]))
            cost.append(dict(cohort=phase,**r,device_padded_fp32_logits_group=device_padded*16))
    for policy,r in screen["ab"].items():
        cost.append(dict(cohort="common_state_ab",policy=policy,**r))
    targeted=json.loads((out/"screen/window_targeted.json").read_text())
    for policy,r in targeted["summary"].items():
        cost.append(dict(cohort="common_state_targeted_c",policy=policy,**r))
    all_rows=json.loads((out/"baseline/predictions.json").read_text())
    dense=[p["official_score"] for p in all_rows if p["policy"]=="dense" and p["sample_id"] in screen["screen_ids"]]
    write_csv(out/"peak_logits.csv",sizes)
    write_csv(out/"combined_cost_ledger.csv",cost)
    _write_json(out/"verification.json",dict(protocol_sha256=_sha(root/"protocol.json"),receipt_sha256=_sha(root/"finished.json"),
                code_hashes_match=True,counts=counts,frozen_heldout=False,passed=False,
                samples_5pct_dense_score=sum(dense)/len(dense),
                pointed_window_head_count=len(targeted["locations"])))
    print(json.dumps(counts))


if __name__=="__main__":
    main()
