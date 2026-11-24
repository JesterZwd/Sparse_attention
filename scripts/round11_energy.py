"""Estimate Round 11 decode energy from predictions and ordinary whole-step timing."""

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path

from gqa_probe.round11_energy import SIDE_FIELDS, break_even_rows, estimate_energy


def _read_timing(path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, *, empty_fields=()):
    fields = tuple(rows[0]) if rows else empty_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(predictions_path, timing_path, output, cohort, *, policies=None):
    predictions_payload = predictions_path.read_bytes()
    predictions = json.loads(predictions_payload)
    if policies is not None:
        missing = set(policies).difference(row["policy"] for row in predictions)
        if missing:
            raise ValueError(f"missing closed-loop predictions for {sorted(missing)}")
        predictions = [row for row in predictions if row["policy"] in policies]
    rows, assumptions = estimate_energy(predictions, _read_timing(timing_path), cohort=cohort)
    comparisons = break_even_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "energy.csv", rows)
    _write_csv(output / "energy_break_even.csv", comparisons,
               empty_fields=("cohort", "reference", "candidate", "normalization", "scenario"))
    assumptions["inputs"] = {
        "predictions": {"path": str(predictions_path), "sha256": sha256(predictions_payload).hexdigest()},
        "timing": {"path": str(timing_path), "sha256": sha256(timing_path.read_bytes()).hexdigest()},
    }
    assumptions["coverage"] = {
        "cohort": cohort,
        "energy_rows": len(rows),
        "break_even_rows": len(comparisons),
        "policies": sorted({row["policy"] for row in rows}),
        "primary_policies": sorted({row["policy"] for row in rows if row["primary"]}),
        "optional_fields_available_for": {
            field: sorted({row["policy"] for row in rows if row[field] is not None})
            for field in (*[f"{key}_per_decode_step" for key in SIDE_FIELDS],
                          "descriptor_runs_per_decode_step", "trace_peak_sram_bytes", "trace_fallback_groups")
        },
    }
    (output / "energy_model.json").write_text(json.dumps(assumptions, indent=2) + "\n", encoding="utf-8")
    return assumptions["coverage"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True, help="step_timing.csv with complete regular steps")
    parser.add_argument("--cohort", choices=("32k", "64k"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", help="Compare selected policies from the closed-loop ledger")
    args = parser.parse_args()
    coverage = run(args.predictions, args.timing, args.output, args.cohort, policies=args.policies)
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
