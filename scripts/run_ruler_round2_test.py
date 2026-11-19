"""Wait for completed calibration, verify the frozen candidate, then test and report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from gqa_probe.ruler_experiment import _sha, freeze_candidate
from scripts.report_ruler_round2 import load_run


def verify_candidate(calibration):
    config, completion, predictions, shadows = load_run(calibration, "calibration", 128)
    file = calibration / "selection.json"
    if completion["files_sha256"].get("selection.json") != _sha(file):
        raise ValueError("calibration receipt does not cover the selected candidate")
    selection = json.loads(file.read_text())
    if (selection["provenance"]["calibration_run_id"] != completion["run_id"]
            or selection["provenance"]["calibration_config_sha256"] != _sha(calibration / "config.json")):
        raise ValueError("selection provenance does not identify this completed calibration")
    identities = {(row["task"], row["sample_id"]) for row in predictions}
    if len(identities) != 128:
        raise ValueError("calibration must contain 128 distinct prompts")
    recomputed = freeze_candidate(shadows, expected_ids=identities, provenance=selection["provenance"])
    if {k: v for k, v in recomputed.items() if k != "created_utc"} != {
            k: v for k, v in selection.items() if k != "created_utc"}:
        raise ValueError("frozen candidate differs from the prespecified calibration rule")
    if datetime.fromisoformat(selection["created_utc"]) > datetime.fromisoformat(completion["completed_utc"]):
        raise ValueError("candidate timestamp falls after calibration completion")
    return config["quantizer"], selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("results/ruler_round2_calibration"))
    parser.add_argument("--test", type=Path, default=Path("results/ruler_round2_test"))
    parser.add_argument("--report", type=Path, default=Path("results/ruler_round2_report"))
    parser.add_argument("--protocol", type=Path, default=Path("results/ruler_round2_protocol.json"))
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume-test", action="store_true")
    args = parser.parse_args()
    while args.wait and not (args.calibration / "complete.json").is_file():
        progress_file = args.calibration / "progress.json"
        if progress_file.exists():
            progress = json.loads(progress_file.read_text())
            if progress["status"] == "interrupted":
                raise RuntimeError(f"calibration requires recovery: {progress.get('error')}")
            updated = datetime.fromisoformat(progress["updated_utc"])
            if (datetime.now(timezone.utc) - updated).total_seconds() > 900:
                raise RuntimeError("calibration progress has been unchanged for 15 minutes")
        time.sleep(15)
    quantizer, selection = verify_candidate(args.calibration)
    protocol = json.loads(args.protocol.read_text())
    for name, digest in protocol["files_sha256"].items():
        if _sha(Path(name)) != digest:
            raise ValueError(f"prespecified analysis protocol file changed: {name}")
    print(f"Verified analysis protocol SHA256 {_sha(args.protocol)}", flush=True)
    print(f"Verified completed calibration and frozen candidate {selection['selected']}", flush=True)
    command = [
        sys.executable, "-u", "-m", "gqa_probe.ruler_experiment", "--phase", "test",
        "--quantizer", quantizer, "--output", str(args.test),
        "--selection", str(args.calibration / "selection.json"),
    ]
    if args.resume_test:
        command.append("--resume")
    subprocess.run(command, check=True)
    subprocess.run([
        sys.executable, "-u", "-m", "scripts.report_ruler_round2",
        "--calibration", str(args.calibration), "--test", str(args.test),
        "--output", str(args.report),
    ], check=True)


if __name__ == "__main__":
    main()
