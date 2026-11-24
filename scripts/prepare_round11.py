"""Bound the reused Round 11 datasets and verify all Round 11 split receipts.

Generate the fresh 32K datasets with scripts.prepare_ruler and the two
prepare_32k*.json configs under results/round11_workpoints. Run this module
with --reuse-only to materialize the 64K and fallback splits before generation
finishes, then run it without arguments to check all splits and seal the receipt.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_ruler import sha256, write_json


RESULTS = Path("results")
ROUND10 = RESULTS / "round10_summary_64k"
ROUND11 = RESULTS / "round11_workpoints"
FWE = RESULTS / "round3_summary_bytes/data/fwe"
PINNED_RULER = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
HISTORY = (
    "ruler_round2_data", "round3_summary_bytes/data/16k", "round3_summary_bytes/data/32k",
    "round3_summary_bytes/data/fwe", "round4_fixed_k_v/data/16k_fwe",
    "round4_fixed_k_v/data/32k_mv", "round4_fixed_k_v/data/test",
    "round5_shared_refill/data/calibration", "round5_shared_refill/data/test",
    "round6_subpage_head/data/calibration", "round6_subpage_head/data/test",
    "round10_summary_64k/data/16k", "round10_summary_64k/data/32k", "round10_summary_64k/data/64k",
)
SPLITS = {
    ("32k", "calibration"): (32768, "niah_multivalue", 16, 2026121401),
    ("32k", "test"): (32768, "niah_multivalue", 128, 2026121420),
    ("64k", "calibration"): (65536, "niah_multivalue", 12, 2026121301),
    ("64k", "test"): (65536, "niah_multivalue", 64, 2026121320),
    ("32k_vt", "calibration"): (32768, "vt", 8, 2026121450),
    ("16k_fwe", "calibration"): (16384, "fwe", 8, 2026113001),
}


def read_json(path):
    return json.loads(path.read_text())


def read_split(root, split):
    manifest = read_json(root / "manifest.json")
    block = manifest["splits"][split]
    path = root / block["path"]
    if manifest["status"] != "complete" or manifest["generator"]["commit"] != PINNED_RULER:
        raise ValueError(f"Unexpected generator or dataset status at {root}")
    if sha256(path) != block["sha256"]:
        raise ValueError(f"Dataset checksum mismatch at {path}")
    with path.open("rb") as stream:
        lines = list(stream)
    rows = [json.loads(line) for line in lines]
    if len(rows) != block["count"] or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Dataset count or sample IDs differ at {path}")
    return manifest, rows, lines


def verify_round10_test(source, rows):
    run = ROUND10 / "64k"
    opened = run / "independent"
    paths = [run / "frozen_test.json", run / "decision.json", opened / "config.json",
             opened / "complete.json", opened / "predictions.json"]
    freeze, decision, config, complete, predictions = map(read_json, paths)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 128 or [row["source_metadata"]["official_index"] for row in rows] != list(range(128)):
        raise ValueError("Expected all 128 original Round 10 test rows in official order")
    if (config != freeze["test"] or config["manifest"] != sha256(source / "manifest.json")
            or config["ids"] != ids[:64] or decision["opened_test"] is not True
            or (config["cohort"], config["phase"], config["split"]) != ("64k", "independent", "test")):
        raise ValueError("Round 10 freeze did not open exactly the first 64 test rows")
    policies = (["dense_kv4"] if config["dense"] else []) + [p["name"] for p in config["policies"]]
    observed = [(p["sample_id"], p["policy"]) for p in predictions]
    expected = [(sample, policy) for sample in ids[:64] for policy in policies]
    bundles = {f"{i:04d}.json.gz" for i in range(64)}
    if (observed != expected or complete["config"] != sha256(opened / "config.json")
            or complete["predictions"] != len(expected) or set(complete["bundles"]) != bundles):
        raise ValueError("Round 10 completed predictions differ from the original freeze")
    local_bundles = opened / "bundles"
    if local_bundles.is_dir():
        if {path.name for path in local_bundles.glob("*.json.gz")} != bundles:
            raise ValueError("Round 10 local bundles differ from the completion receipt")
        for i, name in enumerate(sorted(bundles)):
            path = local_bundles / name
            with gzip.open(path, "rt") as stream:
                bundle = json.load(stream)
            start = i * len(policies)
            if sha256(path) != complete["bundles"][name] or bundle != predictions[start:start + len(policies)]:
                raise ValueError(f"Round 10 prediction bundle differs at {path}")
    return {
        "opened_row_range_zero_based": [0, 64],
        "retained_row_range_zero_based": [64, 128],
        "range_end_exclusive": True,
        "prediction_rows": len(predictions),
        "bundle_count_in_receipt": len(bundles),
        "local_bundles_verified": local_bundles.is_dir(),
        "files_sha256": {str(path): sha256(path) for path in paths},
    }


def subset(destination, source, ranges, *, verify_test=False):
    with TemporaryDirectory(dir=destination.parent) as work:
        staged = Path(work) / destination.name
        staged.mkdir()
        blocks = {}
        selections = {}
        for split, (start, stop) in ranges.items():
            manifest, rows, lines = read_split(source, split)
            if verify_test and split == "test":
                round10 = verify_round10_test(source, rows)
            block = manifest["splits"][split]
            if not 0 <= start < stop <= len(rows):
                raise ValueError(f"Invalid selection for {source}/{split}")
            rows = rows[start:stop]
            output = staged / f"{split}.jsonl"
            output.write_bytes(b"".join(lines[start:stop]))
            lengths = [row["input_tokens"] for row in rows]
            count = len(rows)
            blocks[split] = dict(block, path=output.name, count=count, count_per_task=count,
                                 sha256=sha256(output), input_tokens_min=min(lengths), input_tokens_max=max(lengths))
            selections[split] = dict(path=block["path"], sha256=block["sha256"],
                                     row_range_zero_based=[start, stop], range_end_exclusive=True)
        inherited = {key: manifest[key] for key in ("benchmark", "generator", "tokenizer", "chat_template_sha256",
                                                      "corpus", "dependencies", "task_definitions",
                                                      "length_policy", "gold_span_policy")}
        config = dict(manifest["configuration"], output_directory=str(destination), splits={
            split: dict(manifest["configuration"]["splits"][split], count_per_task=block["count"])
            for split, block in blocks.items()})
        provenance = dict(dataset=str(source), manifest_sha256=sha256(source / "manifest.json"),
                          split_selections=selections, raw_metadata_paths_relative_to=str(source))
        if verify_test:
            provenance["round10_test_exposure"] = round10
        write_json(staged / "manifest.json", dict(inherited, schema_version=1, status="complete",
                   configuration=config, splits=blocks, derived_from=provenance,
                   total_rows=sum(block["count"] for block in blocks.values())))
        if destination.exists():
            for file in staged.iterdir():
                if sha256(file) != sha256(destination / file.name):
                    raise ValueError(f"Existing dataset differs from regenerated {destination / file.name}")
        else:
            staged.rename(destination)


def instance_hashes(rows):
    hashes = {row["source_metadata"]["instance_sha256"] for row in rows}
    if len(hashes) != len(rows):
        raise ValueError("Dataset contains duplicate source instances")
    return hashes


def verify_splits():
    historical = {}
    for name in HISTORY:
        root = RESULTS / name
        historical.update({(root, split): instance_hashes(read_split(root, split)[1])
                           for split in read_json(root / "manifest.json")["splits"]})
    seen = set()
    verified = {}
    provenances = {}
    for (cohort, split), (length, task, count, seed) in SPLITS.items():
        root = ROUND11 / "data" / cohort
        manifest, rows, _ = read_split(root, split)
        block = manifest["splits"][split]
        if (manifest["configuration"]["max_seq_length"] != length or len(rows) != count
                or block["seeds"] != {task: seed} or {(r["task"], r["split"], r["source_metadata"]["seed"])
                                                       for r in rows} != {(task, split, seed)}
                or any(len(r["input_ids"]) != r["input_tokens"] or r["input_tokens"] + r["max_new_tokens"] > length
                       for r in rows)):
            raise ValueError(f"Unexpected context, seed, task, or count for {root}/{split}")
        instances = instance_hashes(rows)
        if seen & instances:
            raise ValueError(f"Round 11 split overlaps another Round 11 split: {root}/{split}")
        seen.update(instances)
        source = manifest.get("derived_from")
        known_reuse = (Path(source["dataset"]), split) if source else None
        for prior, hashes in historical.items():
            expected = instances if prior == known_reuse else set()
            if instances & hashes != expected:
                raise ValueError(f"Unexpected overlap between {root}/{split} and {prior}")
        name = f"{cohort}_{split}"
        verified[name] = dict(path=str(root), context=length, task=task, rows=count, seed=seed,
                              dataset_sha256=block["sha256"], manifest_sha256=sha256(root / "manifest.json"),
                              instance_set_sha256=hashlib.sha256("\n".join(sorted(instances)).encode()).hexdigest(),
                              input_tokens_min=block["input_tokens_min"], input_tokens_max=block["input_tokens_max"])
        if source:
            provenances[cohort] = source
    receipt = dict(generator_commit=PINNED_RULER, splits=verified, total_unique_instances=len(seen),
                   checked_historical_splits={f"{root}/{split}": len(hashes)
                                               for (root, split), hashes in historical.items()},
                   independence_key="source_metadata.instance_sha256",
                   historical_overlap="Zero except byte-identical source rows selected for 64K and 16K FWE.",
                   derived_cohorts=provenances)
    path = ROUND11 / "data_receipt.json"
    if path.exists() and read_json(path) != receipt:
        raise ValueError(f"Recomputed receipt differs from {path}")
    write_json(path, receipt)
    print(f"Verified {len(seen)} unique Round 11 instances in {len(verified)} splits against "
          f"{len(historical)} prior splits. {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-only", action="store_true")
    args = parser.parse_args()
    data = ROUND11 / "data"
    data.mkdir(parents=True, exist_ok=True)
    subset(data / "64k", ROUND10 / "data/64k", {"calibration": (0, 12), "test": (64, 128)}, verify_test=True)
    subset(data / "16k_fwe", FWE, {"calibration": (0, 8)})
    print(f"Prepared {data / '64k'} (12 calibration, 64 held out) and {data / '16k_fwe'} (8 screening)", flush=True)
    if not args.reuse_only:
        verify_splits()


if __name__ == "__main__":
    main()
