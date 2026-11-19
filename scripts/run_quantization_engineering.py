"""Bounded two-variant quantizer repair on the frozen32 engineering prompts."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.quantization_round2 import QUANTIZATION_SPECS, prefill_for_quantization, quantization_spec
from gqa_probe.ruler_inference import generate_from_prefill
from scripts.run_ruler_engineering import _sha, _write_json, _verified_model_hashes


SELECTION_RULE = dict(expected_prompts_per_variant=32, required_complete_correct=32,
                      eligible_if="32/32 complete answers, matching frozen BF16 engineering",
                      tie_break="lower theoretical KV scale-plus-zero-point bits per element",
                      theoretical_bits_per_element={"affine32": 4.75, "symmetric16": 5.},
                      if_none="stop additional quantization variants; root decides next step",
                      scope="engineering tuning only; no held-out quality claim")


def choose_quantization(records):
    grouped = {name: {} for name in QUANTIZATION_SPECS}
    for record in records:
        name, identity = record["spec"], record["sample_id"]
        if name not in grouped or identity in grouped[name]:
            raise ValueError("unknown variant or duplicate prompt/variant record")
        grouped[name][identity] = record
    counts = {name: len(rows) for name, rows in grouped.items()}
    correct = {name: sum(bool(row["complete_correct"]) for row in rows.values()) for name, rows in grouped.items()}
    if any(count > 32 for count in counts.values()):
        raise ValueError("engineering decision has more than32 prompts per variant")
    if any(count < 32 for count in counts.values()):
        return dict(decision="incomplete", selected_spec=None, counts=counts, complete_correct=correct, eligible=[])
    if set(grouped["affine32"]) != set(grouped["symmetric16"]):
        raise ValueError("quantizer variants must use identical engineering prompts")
    eligible = [name for name in grouped if correct[name] == 32]
    selected = min(eligible, key=lambda name: QUANTIZATION_SPECS[name]["theoretical_bits_per_element"]) if eligible else None
    return dict(decision="selected" if selected else "stop_no_eligible_variant", selected_spec=selected,
                counts=counts, complete_correct=correct, eligible=eligible)


def summarize(records, baselines):
    def aggregate(rows):
        return dict(count=len(rows), complete_correct_count=sum(row["complete_correct"] for row in rows),
                    complete_correct=float(np.mean([row["complete_correct"] for row in rows])),
                    official_score=float(np.mean([row["official_score"] for row in rows])),
                    truncated_count=sum(row.get("truncated", False) for row in rows),
                    prefill_seconds_mean=float(np.mean([row["prefill_seconds"] for row in rows])),
                    decode_seconds_mean=float(np.mean([row["decode_seconds"] for row in rows])))
    per_task = {}
    tasks = sorted({row["task"] for row in baselines.values()})
    for task in tasks:
        task_result = {"BF16_reused": aggregate([row for row in baselines.values() if row["task"] == task])}
        for name in QUANTIZATION_SPECS:
            subset = [row for row in records if row["spec"] == name and row["task"] == task]
            if subset:
                task_result[name] = aggregate(subset)
        per_task[task] = task_result
    overall = {"BF16_reused": aggregate(list(baselines.values()))}
    paired = {}
    for name in QUANTIZATION_SPECS:
        subset = [row for row in records if row["spec"] == name]
        if subset:
            overall[name] = aggregate(subset)
            paired[name] = dict(paired_prompts=len(subset),
                                complete_correct_delta_vs_bf16=float(np.mean([float(row["complete_correct"]) - float(baselines[row["sample_id"]]["complete_correct"]) for row in subset])),
                                official_score_delta_vs_bf16=float(np.mean([row["official_score"] - baselines[row["sample_id"]]["official_score"] for row in subset])))
    return dict(completed_generations=len(records), expected_generations=64, per_task=per_task,
                overall=overall, paired=paired, selection=choose_quantization(records),
                interpretation="Engineering tuning only. Reuses original BF16 scores; numeric-reference timings and theoretical layouts do not establish packed memory or speed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data", type=Path, default=Path("results/ruler_round2_data/engineering.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("results/ruler_round2_engineering"))
    parser.add_argument("--output", type=Path, default=Path("results/ruler_round2_quantization_engineering"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("quantization engineering requires CUDA")
    if args.data.name != "engineering.jsonl":
        raise ValueError("this runner only reads engineering rows")
    manifest_path = args.data.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_hash = _sha(args.data)
    if manifest["status"] != "complete" or data_hash != manifest["splits"]["engineering"]["sha256"]:
        raise ValueError("engineering data manifest/hash mismatch")
    inputs = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(inputs) != 32 or any(row["split"] != "engineering" or row["max_new_tokens"] != 128 or
                               row["input_tokens"] != len(row["input_ids"]) or row["input_tokens"] + 128 > 16384 for row in inputs):
        raise ValueError("expected frozen32 engineering prompts with128 reserved output tokens")
    baseline_config = json.loads((args.baseline / "config.json").read_text())
    baseline_summary = json.loads((args.baseline / "summary.json").read_text())
    baseline_path = args.baseline / "predictions.jsonl.gz"
    if baseline_summary["status"] != "complete" or _sha(baseline_path) != baseline_summary["predictions_sha256"] or baseline_config["data_sha256"] != data_hash:
        raise ValueError("frozen BF16 baseline provenance mismatch")
    with gzip.open(baseline_path, "rt") as stream:
        baseline_records = [json.loads(line) for line in stream]
    baselines = {row["sample_id"]: row for row in baseline_records if row["bitmode"] == "BF16"}
    if len(baselines) != 32 or set(baselines) != {row["sample_id"] for row in inputs} or not all(row["complete_correct"] for row in baselines.values()):
        raise ValueError("baseline must contain all32 correct BF16 engineering predictions")
    for row in inputs:
        baseline = baselines[row["sample_id"]]
        if baseline["input_ids_sha256"] != row["source_metadata"]["input_ids_sha256"] or baseline["references"] != row["references"]:
            raise ValueError("baseline prompt/reference pairing mismatch")
        if score_prediction(baseline["prediction"], baseline["references"]) != {key: baseline[key] for key in ("official_score", "complete_correct")}:
            raise ValueError("baseline score mismatch")
    # Dense backend/helper code must match the original baseline. Torch selector
    # association changes are irrelevant because both actual policies are dense.
    for name in ("capture.py", "oracle_attention.py", "ruler_inference.py", "selection.py", "paired_statistics.py"):
        path = Path("gqa_probe") / name
        if _sha(path) != baseline_config["code_sha256"][str(path)]:
            raise ValueError(f"dense baseline code differs: {name}")
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("resume requires an existing output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])
    logging.info("Verifying local checkpoint and freezing two prespecified repairs")
    source, model_hashes = _verified_model_hashes(args.model)
    if model_hashes != baseline_config["model_file_sha256"]:
        raise ValueError("checkpoint differs from frozen BF16 baseline")
    code_paths = [Path(__file__), Path("scripts/run_ruler_engineering.py"), *(Path("gqa_probe") / name for name in
                  ("__init__.py", "capture.py", "oracle_attention.py", "torch_selection.py", "ruler_inference.py", "selection.py", "paired_statistics.py", "quantization_round2.py"))]
    stable = dict(schema_version=1, split="engineering", data=str(args.data.resolve()), data_sha256=data_hash,
                  data_manifest_sha256=_sha(manifest_path), model=str(args.model.resolve()), model_source=source,
                  model_source_manifest_sha256=_sha(args.model / "source_manifest.json"), model_file_sha256=model_hashes,
                  baseline_config_id=baseline_config["config_id"], baseline_predictions_sha256=_sha(baseline_path),
                  baseline_directory=str(args.baseline.resolve()), baseline_reference_reused=True,
                  methods=QUANTIZATION_SPECS, selection_rule=SELECTION_RULE,
                  code_sha256={str(path): _sha(path) for path in code_paths},
                  model_dtype="bfloat16", attention="grouped_fp32_dense_decode_sdpa_prefill", tf32=False,
                  max_new_tokens=128, eos_policy="generation_config_eos_inclusive_prefill_first_token",
                  torch_threads=8, seed=20261118, python=platform.python_version(),
                  torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name())
    config_id = hashlib.sha256(json.dumps(stable, sort_keys=True, allow_nan=False).encode()).hexdigest()
    config = dict(stable, config_id=config_id, created_utc=datetime.now(timezone.utc).isoformat())
    predictions_path = args.output / "predictions.jsonl.gz"
    records = []
    if args.resume:
        old_config = json.loads((args.output / "config.json").read_text())
        if old_config["config_id"] != config_id:
            raise ValueError("resume code/data/config mismatch")
        config = old_config
        if predictions_path.exists():
            with gzip.open(predictions_path, "rt") as stream:
                records = [json.loads(line) for line in stream]
        choose_quantization(records)
        if any(row["config_id"] != config_id for row in records):
            raise ValueError("resumed prediction config mismatch")
    else:
        _write_json(args.output / "config.json", config)
        _write_json(args.output / "selection_rule.json", SELECTION_RULE)
        with gzip.open(args.output / "bf16_reference.jsonl.gz", "wt") as stream:
            for row in baselines.values():
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        for path in code_paths:
            relative = path.resolve().relative_to(Path.cwd())
            snapshot = args.output / "code_snapshot" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
    _write_json(args.output / "progress.json", dict(status="loading_model", completed_generations=len(records), expected_generations=64, config_id=config_id))
    torch.set_num_threads(8)
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map="cuda", low_cpu_mem_usage=True,
                                                local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    seen = {(record["sample_id"], record["spec"]) for record in records}
    started = time.perf_counter()
    status = "running"
    try:
        for index, row in enumerate(inputs):
            for name in ("affine32", "symmetric16"):
                if (row["sample_id"], name) in seen:
                    continue
                logging.info("START sample=%d/32 id=%s spec=%s", index + 1, row["sample_id"], name)
                state = prefill_for_quantization(model, row["input_ids"], name)
                attention = OracleAttention("dense", gold_spans=row["gold_spans"])
                generated = generate_from_prefill(model, state, 128, eos, attention)
                prediction = tokenizer.decode(generated.token_ids, skip_special_tokens=True)
                eos_reached = generated.token_ids[-1] in eos
                baseline = baselines[row["sample_id"]]
                record = dict(config_id=config_id, sample_id=row["sample_id"], task=row["task"], split="engineering",
                              spec=name, quantization=quantization_spec(name), references=row["references"],
                              prediction=prediction, token_ids=generated.token_ids, generated_tokens=len(generated.token_ids),
                              eos_token_ids=eos, eos_reached=eos_reached, reached_token_limit=len(generated.token_ids) == 128,
                              truncated=len(generated.token_ids) == 128 and not eos_reached,
                              stopping_reason="eos" if eos_reached else "max_new_tokens", max_new_tokens=128,
                              consumed_decode_steps=len(generated.token_ids) - 1, input_tokens=row["input_tokens"],
                              input_ids_sha256=row["source_metadata"]["input_ids_sha256"], source_metadata=row["source_metadata"],
                              prefill_seconds=state.elapsed_seconds, decode_seconds=generated.elapsed_seconds,
                              total_reference_seconds=state.elapsed_seconds + generated.elapsed_seconds,
                              attention_stats=generated.attention_stats, baseline_config_id=baseline["config_id"],
                              baseline_complete_correct=baseline["complete_correct"], baseline_official_score=baseline["official_score"],
                              **score_prediction(prediction, row["references"]))
                with gzip.open(predictions_path, "at", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
                seen.add((row["sample_id"], name))
                summary = summarize(records, baselines)
                summary.update(config_id=config_id, status="running")
                _write_json(args.output / "summary.json", summary)
                _write_json(args.output / "progress.json", dict(status="running", completed_generations=len(records), expected_generations=64,
                                                               config_id=config_id, last_sample_id=row["sample_id"], last_spec=name))
                logging.info("DONE %d/64 spec=%s score=%.3f complete=%s tokens=%d eos=%s prediction=%r", len(records), name,
                             record["official_score"], record["complete_correct"], len(generated.token_ids), eos_reached, prediction)
                del state, attention, generated
        status = "complete"
    except BaseException:
        status = "failed"
        logging.exception("Quantization engineering interrupted")
        raise
    finally:
        summary = summarize(records, baselines)
        summary.update(config_id=config_id, status=status, elapsed_run_seconds=time.perf_counter() - started,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                       predictions_sha256=_sha(predictions_path) if predictions_path.exists() else None)
        _write_json(args.output / "summary.json", summary)
        _write_json(args.output / "selection.json", dict(config_id=config_id, rule=SELECTION_RULE, **summary["selection"]))
        _write_json(args.output / "progress.json", dict(status=status, completed_generations=len(records), expected_generations=64, config_id=config_id))
        logging.info("FINAL status=%s selected=%s", status, summary["selection"])


if __name__ == "__main__":
    main()
