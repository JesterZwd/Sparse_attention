"""Consolidate results from the frozen Round 13 evaluation panels."""
import csv
import json
from pathlib import Path

from gqa_probe.ruler_experiment import _sha, _write_json
from scripts.report_ruler_round3 import write_csv

ROOT=Path("results/round13_comrad")


def main():
    selection = ROOT / "task_selection.json"
    task = json.loads(selection.read_text())["chosen"] if selection.is_file() else None
    panels = ("qwen3_64k_multivalue", "qwen25_32k_multivalue", task, "qwen3_longbench_v2")
    evidence={}
    missing = []
    rows=[]
    for panel in panels:
        if panel is None:
            missing.append("qwen3_32k_fwe_or_cwe")
            continue
        path=ROOT/panel/"test"
        if not (path / "complete.json").is_file():
            missing.append(panel)
            continue
        receipt=json.loads((path/"complete.json").read_text())
        config=json.loads((path/"config.json").read_text())
        if receipt["config_sha256"]!=_sha(path/"config.json") or config!=json.loads(
                (path.parent/"frozen_test.json").read_text()):
            raise ValueError(f"{panel} differs from its frozen configuration")
        with (path/"quality_cost.csv").open(newline="") as stream:
            measurements={r["policy"]:r for r in csv.DictReader(stream)}
        if receipt["predictions"]!=sum(int(r["prompts"]) for r in measurements.values()):
            raise ValueError(f"{panel} has incomplete predictions")
        pairs=json.loads((path/"paired.json").read_text())
        for pair in pairs:
            a,b=(measurements[x] for x in (pair["reference"],pair["candidate"]))
            score=pair["intervals"]["official_score"]
            rows.append(dict(panel=panel,prompts=a["prompts"],reference=pair["reference"],candidate=pair["candidate"],
                             baseline_score=a["official_score"],candidate_score=b["official_score"],
                             point_delta=score["point_delta"],ci_low=score["ci_low"],ci_high=score["ci_high"],
                             reference_complete=a["complete"],candidate_complete=b["complete"],
                             reference_read_per_decode=a["read_per_decode"],candidate_read_per_decode=b["read_per_decode"],
                             reference_read_per_answer=a["read_per_answer"],candidate_read_per_answer=b["read_per_answer"],
                             reference_arithmetic_per_decode=a["arithmetic_per_decode"],
                             candidate_arithmetic_per_decode=b["arithmetic_per_decode"],
                             quality_pass=pair["gate"]["quality_pass"] if pair["gate"] else None,
                             saving_pass=pair["gate"]["saving_pass"] if pair["gate"] else None))
        evidence[panel]=dict(frozen_sha256=receipt["config_sha256"],predictions=receipt["predictions"],
                             sparse_overruns=sum(int(r["overrun_groups"]) for r in measurements.values()
                                                 if not r["policy"].startswith("dense")),
                             pair_gates={x["reference"]:x["gate"] for x in pairs if x["gate"]})
    if rows:
        write_csv(ROOT/"cross_panel_quality_cost.csv",rows)
    filename = "panels_partial.json" if missing else "panels_complete.json"
    _write_json(ROOT/filename,dict(panels=evidence,missing_panels=missing,
                replay_sha256=_sha(ROOT/"replay/replay_receipt.json")))


if __name__=="__main__":
    main()
