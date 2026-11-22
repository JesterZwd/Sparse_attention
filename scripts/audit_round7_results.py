"""Check Round 7 receipts and independently reconcile group-step read totals."""
import argparse
import json
from pathlib import Path

from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.report_round6 import load
from scripts.round7_run import ROOT, sources


def read(path):
    return json.loads(path.read_text())


def audit(root):
    protocol_path = root/"protocol.json"
    protocol = read(protocol_path)
    done = read(root/"finished.json")
    code = sources()
    if protocol["source_sha256"]!=code or done["source_sha256"]!=code:
        raise ValueError("completed experiment source hash mismatch")
    phases = ["baseline", "screen", "compact", "val_gate"]
    if done["compact_admitted"]:
        phases.append("compact_gate")
    if done["heldout"]:
        phases.append("test")
        freeze = read(root/"frozen_test.json")
        cfg = read(root/"test/config.json")
        if any(cfg[key]!=freeze[key] for key in ("ids","manifest_sha256","policies","source_sha256")):
            raise ValueError("test differs from frozen protocol")
        if len(cfg["ids"])!=128 or freeze["protocol_sha256"]!=_sha(protocol_path):
            raise ValueError("held-out cohort or protocol mismatch")
    rows = {}
    for phase in phases:
        conf = read(root/phase/"config.json")
        if conf["source_sha256"]!=code:
            raise ValueError(f"source mismatch in {phase}")
        if conf["manifest_sha256"]!=protocol["closure_datasets"][
                "test" if phase=="test" else "calibration"]["manifest_sha256"]:
            raise ValueError(f"manifest mismatch in {phase}")
        count = dict(prompts=0, predictions=0, checked_group_steps=0, sampled_timing=0)
        for bundle in load(root,phase):
            count["prompts"] += 1
            count["predictions"] += len(bundle["answers"])
            count["sampled_timing"] += len(bundle["timings"])
            answers = {x["policy"]:x for x in bundle["answers"]}
            for policy in answers.values():
                if policy.get("over_budget_group_steps",0):
                    raise ValueError(f"runtime budget violation in {phase}")
            for ledger in bundle["group_steps"]:
                n=total=fallback=0
                for step,layer,cap,groups in ledger["steps"]:
                    for read_bytes, kpages, k, v, back, prefix, excess, pv in groups:
                        if not (read_bytes%32==0 and read_bytes<=cap and
                                0<prefix<=kpages and v<=k and pv<=4*v and back==bool(excess)):
                            raise ValueError(f"group violates physical constraints at {phase} {step} {layer}")
                        n+=1
                        total+=read_bytes
                        fallback+=back
                answer=answers[ledger["policy"]]
                if (n,total,fallback)!=(answer["kv_group_calls"],answer["read"],answer.get("fallback_group_steps",0)):
                    raise ValueError(f"packed group records disagree with {phase} prompt totals")
                count["checked_group_steps"]+=n
        rows[phase]=count
    result=dict(source_hashes_match=True, protocol_sha256=_sha(protocol_path),
                finished_sha256=_sha(root/"finished.json"), phases=rows,
                heldout=done["heldout"], passed=done["passed"])
    _write_json(root/"report"/"verification.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    print(json.dumps(audit(p.parse_args().root)))
