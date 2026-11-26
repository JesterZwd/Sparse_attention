#!/usr/bin/env python3
"""Prepare source-pinned Round 13 panel receipts without running a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from shutil import copyfileobj
from urllib.request import urlopen

from scripts.prepare_round11 import PINNED_RULER, read_split
from scripts.prepare_ruler import prepare, sha256, write_json

RESULTS = Path("results")
ROUND13 = RESULTS / "round13_comrad"
ROUND12 = RESULTS / "round12_frontend"
RULER64 = ROUND12 / "data/64k"
LONG_BENCH_TASK = "longbench_v2"
LONG_BENCH_CODE = "https://github.com/THUDM/LongBench"
LONG_BENCH_CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_DATA = "https://huggingface.co/datasets/zai-org/LongBench-v2"
LONG_BENCH_DATA_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONG_BENCH_SOURCE_FILES = {
    "data.json": (f"{LONG_BENCH_DATA}/resolve/{LONG_BENCH_DATA_REVISION}/data.json",
                  "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"),
    "dataset_card.md": (f"{LONG_BENCH_DATA}/resolve/{LONG_BENCH_DATA_REVISION}/README.md",
                        "9fdd1a3ebe86507253c124a18e9f78c898ce6341c12990af17ab868b8f600c35"),
    "pred.py": (f"https://raw.githubusercontent.com/THUDM/LongBench/{LONG_BENCH_CODE_REVISION}/pred.py",
                "ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138"),
    "prompts/0shot.txt": (f"https://raw.githubusercontent.com/THUDM/LongBench/{LONG_BENCH_CODE_REVISION}/prompts/0shot.txt",
                         "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"),
    "LICENSE": (f"https://raw.githubusercontent.com/THUDM/LongBench/{LONG_BENCH_CODE_REVISION}/LICENSE",
                "bf7f05f274d65931bdf14f060b6855f1c359b8424eb74fd3091d25de89d700c3"),
}
QWEN3_TOKENIZER = Path("models/Qwen3-4B-Instruct-2507")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
MODEL_REVISION_QWEN25 = "a09a35458c702b33eeacc393d103063234e8bc28"
PANEL_SPLITS = {
    "qwen3_32k_fwe": ("fwe", {"calibration": (8, 2026121701), "test": (64, 2026121720)}),
    "qwen3_32k_cwe": ("cwe", {"calibration": (8, 2026121730), "test": (64, 2026121740)}),
    "qwen25_32k_multivalue": ("niah_multivalue", {"fit": (4, 2026121801), "calibration": (16, 2026121811),
                                               "test": (64, 2026121820)}),
}


def json_file(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def verify_rebuilt_manifest(actual, frozen, expected_sha256):
    """Check a rebuilt cohort against its original manifest and split digests."""
    if sha256(frozen) != expected_sha256:
        raise ValueError(f"published manifest checksum differs: {frozen}")
    local, original = json_file(actual), json_file(frozen)
    for field in ("created_utc", "provenance_files"):
        local.pop(field, None)
        original.pop(field, None)
    if local != original:
        raise ValueError(f"rebuilt RULER manifest differs from frozen input: {actual}")


def historical_split_hashes(*, exclude=()):
    """Index every completed pinned RULER split available before Round 13."""
    for path in sorted(RESULTS.glob("**/manifest.json")):
        if path.parent.is_relative_to(ROUND13) or path.parent in exclude:
            continue
        if path.parent != RESULTS / "ruler_round2_data" and "data" not in path.relative_to(RESULTS).parts:
            continue
        manifest = json_file(path)
        if manifest.get("generator", {}).get("commit") != PINNED_RULER or manifest.get("status") != "complete":
            continue
        for split in sorted(manifest["splits"]):
            _, rows, _ = read_split(path.parent, split)
            yield str(path.parent / split), {r["source_metadata"]["instance_sha256"] for r in rows}


def verify_qwen3_64k():
    """Confirm the Round 12 64K independent cohort and preserve its original IDs."""
    verification_path = ROUND12 / "verification.json"
    verification = json_file(verification_path)
    software = verification["software"]
    absent = [ROUND12 / "64k" / name for name in ("frozen_test.json", "independent")]
    if software["independent_64k_opened"] or software["calibration_64k_eligible"] or any(p.exists() for p in absent):
        raise ValueError("Round 12 64K independent cohort cannot be proven unopened")
    source_receipt = json_file(ROUND12 / "data_receipt.json")["64k"]
    manifest, rows, _ = read_split(RULER64, "test")
    queries = [r["source_metadata"]["instance_sha256"] for r in rows]
    block = manifest["splits"]["test"]
    identities = [r["sample_id"] for r in rows]
    if (source_receipt["manifest_sha256"] != sha256(RULER64 / "manifest.json")
            or source_receipt["dataset_sha256"] != block["sha256"]
            or source_receipt["sample_ids"] != identities
            or source_receipt["instance_sha256"] != queries
            or len(rows) != 64 or len(set(queries)) != 64
            or manifest["configuration"]["max_seq_length"] != 65536
            or block["seeds"] != {"niah_multivalue": 2026121620}):
        raise ValueError("Round 12 64K source or inventory differs from the sealed receipt")
    if any(r["task"] != "niah_multivalue" or r["split"] != "test"
           or len(r["input_ids"]) != r["input_tokens"]
           or r["input_tokens"] + r["max_new_tokens"] > 65536 for r in rows):
        raise ValueError("invalid Round 12 64K source row")
    hashes = set(queries)
    compared = {}
    for key, prior in historical_split_hashes(exclude=(RULER64,)):
        compared[key] = len(prior)
        if hashes & prior:
            raise ValueError(f"Round 12 64K source overlaps historical split {key}")
    if not compared:
        raise ValueError("no historical pinned cohorts")
    receipt = dict(status="verified_unopened_in_round12", source=str(RULER64), split="test",
                   model="Qwen3-4B-Instruct-2507", length=65536, prompts=len(rows),
                   source_round12_verification_sha256=sha256(verification_path),
                   source_round12_data_receipt_sha256=sha256(ROUND12 / "data_receipt.json"),
                   source_manifest_sha256=sha256(RULER64 / "manifest.json"),
                   source_rows_sha256=block["sha256"], seeds=block["seeds"], official_ruler_commit=PINNED_RULER,
                   sample_ids=identities, instance_sha256=queries, compared_historical_splits=compared,
                   zero_historical_instance_overlap=True)
    ROUND13.mkdir(parents=True, exist_ok=True)
    write_json(ROUND13 / "qwen3_64k_seal.json", receipt)
    print(f"Round 12 64K test verified unopened, 64 unique instances against {len(compared)} splits", flush=True)


def extract_longbench_answer(response):
    """Apply the pinned LongBench v2 direct-answer matching rules."""
    cleaned = response.strip().replace("*", "")
    for pattern in (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])"):
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return None


extract_answer = extract_longbench_answer


def longbench_correct(response, answer):
    return extract_longbench_answer(response) == answer


def load_panel(path, split):
    """Load validated panel rows, including the locally adapted LongBench format."""
    path = Path(path)
    manifest = json_file(path / "manifest.json")
    if manifest["generator"]["commit"] == PINNED_RULER:
        manifest, rows, _ = read_split(path, split)
    else:
        block = manifest["splits"][split]
        source = path / block["path"]
        if (manifest["generator"]["commit"] != LONG_BENCH_CODE_REVISION
                or manifest.get("status") != "complete" or sha256(source) != block["sha256"]):
            raise ValueError("LongBench cohort differs from its pinned receipt")
        with source.open() as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != block["count"] or any(r["task"] != LONG_BENCH_TASK for r in rows):
            raise ValueError("invalid adapted LongBench cohort")
    length = manifest["configuration"]["max_seq_length"]
    if len({r["sample_id"] for r in rows}) != len(rows) or any(
        r["split"] != split or len(r["input_ids"]) != r["input_tokens"]
        or r["input_tokens"] + r["max_new_tokens"] > length for r in rows
    ):
        raise ValueError("invalid panel source identities or lengths")
    return manifest, length, rows


def checked_longbench_sources(cache):
    """Fetch missing official bytes and check every pinned source file."""
    provenance = {}
    for name, (url, expected) in LONG_BENCH_SOURCE_FILES.items():
        path = cache / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                copyfileobj(response, stream)
            if sha256(temporary) != expected:
                raise ValueError(f"LongBench download differs from pinned bytes: {name}")
            temporary.replace(path)
        if sha256(path) != expected:
            raise ValueError(f"LongBench cache differs from pinned bytes: {name}")
        provenance[name] = dict(url=url, sha256=expected)
    return provenance


def prepare_longbench():
    """Select the fixed direct-answer subset by post-template token length."""
    from transformers import AutoTokenizer

    cache = Path(".cache/round13/longbench")
    sources = checked_longbench_sources(cache)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER, local_files_only=True)
    template = (cache / "prompts/0shot.txt").read_text()
    raw = json_file(cache / "data.json")
    if len(raw) != 503 or len({r["_id"] for r in raw}) != len(raw):
        raise ValueError("unexpected official LongBench inventory")
    seed = 2026121901
    prompt_min, total_max, reserve = 16384, 32768, 128
    fields = (("DOC", "context"), ("Q", "question"), *((f"C_{c}", f"choice_{c}") for c in "ABCD"))
    eligible = []
    # Hash ordering uses only the official IDs and freeze seed, before labels
    # or inference. Evaluate actual chat IDs and retain the first 64 eligible.
    ordered = sorted(raw, key=lambda item: (json_hash([seed, item["_id"]]), item["_id"]))
    rows = []
    for item in ordered:
        prompt = template
        for placeholder, key in fields:
            prompt = prompt.replace(f"${placeholder}$", item[key].strip())
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        length = len(ids)
        if prompt_min <= length <= total_max - reserve:
            eligible.append({"id": item["_id"], "input_tokens": length, "domain": item["domain"],
                             "difficulty": item["difficulty"]})
            if len(rows) < 64:
                identity = item["_id"]
                answer = item["answer"]
                if answer not in ("A", "B", "C", "D"):
                    raise ValueError(f"unexpected LongBench answer for {identity}")
                metadata = {
                    "official_id": identity, "domain": item["domain"], "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"], "official_length_bucket": item["length"],
                    "instance_sha256": json_hash([LONG_BENCH_TASK, prompt, [answer]]),
                    "final_chat_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "input_ids_sha256": json_hash(ids),
                    "gold_evidence": [], "gold_span_semantics": "unavailable for this task",
                }
                rows.append(dict(sample_id=f"test/{LONG_BENCH_TASK}/{identity}", task=LONG_BENCH_TASK, split="test",
                                 input_ids=ids, references=[answer], gold_spans=[], input_tokens=length,
                                 max_new_tokens=reserve, source_metadata=metadata))
    if len(rows) < 48:
        raise ValueError(f"Only {len(rows)} LongBench rows fit the registered chat lengths")
    if len({r["source_metadata"]["instance_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate adapted LongBench instance")
    output = ROUND13 / "data/qwen3_longbench_v2"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "test.jsonl"
    temporary = source.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = sha256(temporary)
    if source.exists() and sha256(source) != digest:
        temporary.unlink()
        raise ValueError("refusing to change frozen LongBench rows")
    temporary.replace(source)
    details = dict(repository=LONG_BENCH_DATA, revision=LONG_BENCH_DATA_REVISION,
                   official_split="train", official_rows=len(raw), card_license="Apache-2.0",
                   local_data_only=True, sources=sources)
    configuration = dict(max_seq_length=total_max, max_new_tokens=reserve, prompt_min_tokens=prompt_min,
                         prompt_max_tokens=total_max-reserve, model="Qwen3-4B-Instruct-2507",
                         template="official direct-answer zero-shot, one user message; original context; no truncation",
                         selection="first 64 eligible IDs in ascending SHA256(JSON([seed, official_id])) order",
                         seed=seed)
    lengths = [r["input_tokens"] for r in rows]
    manifest = dict(schema_version=1, status="complete", benchmark="LongBench v2",
                    generator=dict(repository=LONG_BENCH_CODE, commit=LONG_BENCH_CODE_REVISION,
                                   license="MIT", script="pred.py", prompt="prompts/0shot.txt"),
                    configuration=configuration, source=details,
                    tokenizer={name: sha256(QWEN3_TOKENIZER / name) for name in TOKENIZER_FILES},
                    chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    gold_span_policy="No official per-question evidence positions",
                    splits=dict(test=dict(path=source.name, sha256=digest, count=len(rows),
                                          input_tokens_min=min(lengths), input_tokens_max=max(lengths))))
    write_json(output / "manifest.json", manifest)
    _, _, checked = load_panel(output, "test")
    audit = dict(official_dataset=details, official_code=manifest["generator"],
                 configuration=configuration, generation="Common deterministic greedy generation across policies",
                 scoring="pred.py extract_answer; exact A–D accuracy; empty/unparsable output is incorrect",
                 eligible_prompts=len(eligible), frozen_prompts=len(checked),
                 eligible_domain_counts=dict(Counter(x["domain"] for x in eligible)),
                 selected_domain_counts=dict(Counter(x["domain"] for x in eligible[:len(checked)])),
                 selected_difficulty_counts=dict(Counter(x["difficulty"] for x in eligible[:len(checked)])),
                 selected_instances=[dict(sample_id=r["sample_id"], input_tokens=r["input_tokens"],
                                          instance_sha256=r["source_metadata"]["instance_sha256"],
                                          final_chat_sha256=r["source_metadata"]["final_chat_sha256"])
                                     for r in checked],
                 input_tokens_min=min(lengths), input_tokens_max=max(lengths),
                 manifest_sha256=sha256(output / "manifest.json"), rows_sha256=digest)
    write_json(ROUND13 / "longbench_receipt.json", audit)
    print(f"LongBench v2: {len(eligible)} eligible, {len(checked)} frozen; {min(lengths)}–{max(lengths)} chat tokens",
          flush=True)


def verify_generated_ruler():
    comparisons = dict(historical_split_hashes())
    history = set().union(*comparisons.values())
    seen = set()
    panels = {}
    for name, (task, splits) in PANEL_SPLITS.items():
        path = ROUND13 / "data" / name
        if name == "qwen3_32k_cwe" and not path.exists():
            continue
        output = {}
        for split, (count, seed) in splits.items():
            manifest, length, rows = load_panel(path, split)
            block = manifest["splits"][split]
            if length != 32768 or len(rows) != count or block["seeds"] != {task: seed} or any(
                r["task"] != task or r["source_metadata"]["seed"] != seed for r in rows
            ):
                raise ValueError(f"unexpected pinned task, seed, or length in {name}/{split}")
            actual = [r["source_metadata"]["instance_sha256"] for r in rows]
            present = set(actual)
            if len(present) != len(rows) or present & (history | seen):
                raise ValueError(f"instance overlap in {name}/{split}")
            seen.update(present)
            output[split] = dict(count=count, seed=seed, rows_sha256=block["sha256"],
                                 input_tokens_min=block["input_tokens_min"], input_tokens_max=block["input_tokens_max"],
                                 ids=[r["sample_id"] for r in rows], instance_sha256=actual)
        panels[name] = dict(manifest_sha256=sha256(path / "manifest.json"),
                            config_sha256=sha256(ROUND13 / ("prepare_" + name + ".json")),
                            tokenizer=manifest["tokenizer"],
                            chat_template_sha256=manifest["chat_template_sha256"], splits=output)
    write_json(ROUND13 / "ruler_data_receipt.json", dict(
        official_ruler_commit=PINNED_RULER, official_code_license="Apache-2.0",
        corpus_note="Paul Graham essays are local evaluation inputs; the generator license does not relicense them.",
        qwen25_tokenizer_repository="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        qwen25_tokenizer_revision=MODEL_REVISION_QWEN25,
        compared_historical_splits={name: len(instances) for name, instances in comparisons.items()},
        unique_new_instances=len(seen), overlap=0, panels=panels))
    print(f"Verified {len(seen)} fresh RULER instances across {len(panels)} panels against {len(comparisons)} historic splits",
          flush=True)


def prepare_workspace(*, verify_only=False):
    """Materialize or check the frozen datasets needed by the GPU queue."""
    r12 = json_file(ROUND12 / "data_receipt.json")
    for length in ("32k", "64k"):
        folder = ROUND12 / "data" / length
        manifest = folder / "manifest.json"
        if not verify_only and not manifest.is_file():
            prepare(ROUND12 / f"prepare_{length}.json")
        verify_rebuilt_manifest(manifest, ROUND13 / "inputs" / f"round12_{length}_manifest.json",
                                r12[length]["manifest_sha256"])
        _, rows, _ = read_split(folder, "test")
        if [row["source_metadata"]["instance_sha256"] for row in rows] != r12[length]["instance_sha256"]:
            raise ValueError(f"Round 12 {length} test instances differ from the original receipt")
        print(f"Round 12 {length}: {len(rows)} frozen rows verified", flush=True)

    saved = json_file(ROUND13 / "ruler_data_receipt.json")["panels"]
    for panel in ("qwen25_32k_multivalue", "qwen3_32k_fwe"):
        folder = ROUND13 / "data" / panel
        config = ROUND13 / f"prepare_{panel}.json"
        if sha256(config) != saved[panel]["config_sha256"]:
            raise ValueError(f"RULER configuration checksum differs for {panel}")
        if not verify_only and not (folder / "manifest.json").is_file():
            prepare(config)
        for split, expected in saved[panel]["splits"].items():
            manifest, _, rows = load_panel(folder, split)
            if (manifest["splits"][split]["sha256"] != expected["rows_sha256"] or
                    manifest["tokenizer"] != saved[panel]["tokenizer"] or
                    manifest["chat_template_sha256"] != saved[panel]["chat_template_sha256"] or
                    [row["source_metadata"]["instance_sha256"] for row in rows] != expected["instance_sha256"]):
                raise ValueError(f"rebuilt rows differ from frozen input: {panel}/{split}")
            print(f"{panel}/{split}: {len(rows)} frozen rows verified", flush=True)

    longbench = ROUND13 / "data/qwen3_longbench_v2"
    expected = json_file(ROUND13 / "longbench_receipt.json")
    if not verify_only and not (longbench / "manifest.json").is_file():
        prepare_longbench()
    manifest, _, rows = load_panel(longbench, "test")
    if (manifest["splits"]["test"]["sha256"] != expected["rows_sha256"] or
            sha256(longbench / "manifest.json") != expected["manifest_sha256"]):
        raise ValueError("LongBench v2 inputs differ from the frozen cohort")
    print(f"LongBench v2: {len(rows)} frozen rows verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("seal", "longbench", "verify-ruler", "workspace", "verify-workspace"))
    args = parser.parse_args()
    if args.stage == "seal":
        verify_qwen3_64k()
    elif args.stage == "longbench":
        prepare_longbench()
    elif args.stage in ("workspace", "verify-workspace"):
        prepare_workspace(verify_only=args.stage == "verify-workspace")
    else:
        verify_generated_ruler()


if __name__ == "__main__":
    main()
