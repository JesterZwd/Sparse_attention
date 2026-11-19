"""Engineering-only BF16 versus dense KV4 reference; never reads held-out rows."""

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

from gqa_probe.capture import QUANTIZATION
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.ruler_inference import prefill_model, generate_from_prefill


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_engineering(records):
    """Descriptive paired engineering results; no formal inference or selection."""
    groups = defaultdict(lambda: defaultdict(list))
    by_prompt = defaultdict(dict)
    seen = set()
    for row in records:
        identity = row["sample_id"], row["bitmode"]
        if identity in seen:
            raise ValueError("duplicate engineering prompt/method record")
        seen.add(identity)
        groups[row["task"]][row["bitmode"]].append(row)
        by_prompt[row["sample_id"]][row["bitmode"]] = row
    def average(rows):
        return dict(count=len(rows), official_score=float(np.mean([r["official_score"] for r in rows])),
                    complete_correct=float(np.mean([r["complete_correct"] for r in rows])),
                    prefill_seconds_mean=float(np.mean([r["prefill_seconds"] for r in rows])),
                    decode_seconds_mean=float(np.mean([r["decode_seconds"] for r in rows])))
    per_task = {task: {mode: average(rows) for mode, rows in sorted(modes.items())}
                for task, modes in sorted(groups.items())}
    overall = {mode: average([r for r in records if r["bitmode"] == mode])
               for mode in ("BF16", "KV4") if any(r["bitmode"] == mode for r in records)}
    pairs = [methods for methods in by_prompt.values() if set(methods) == {"BF16", "KV4"}]
    paired = dict(bf16_correct_kv4_wrong=sum(p["BF16"]["complete_correct"] and not p["KV4"]["complete_correct"] for p in pairs),
                  bf16_wrong_kv4_correct=sum(not p["BF16"]["complete_correct"] and p["KV4"]["complete_correct"] for p in pairs))
    for metric in ("official_score", "complete_correct"):
        paired[metric + "_delta_kv4_minus_bf16"] = (float(np.mean([
            float(p["KV4"][metric]) - float(p["BF16"][metric]) for p in pairs])) if pairs else None)
    return dict(completed_generations=len(records), paired_prompts=len(pairs), per_task=per_task,
                overall=overall, paired=paired,
                interpretation="Engineering descriptive results only; no test labels inspected; runtime is numeric-reference timing, not packed KV4 or sparse-system speed.")


def _verified_model_hashes(model_path):
    source = json.loads((model_path / "source_manifest.json").read_text())
    hashes = {}
    for entry in source["files"]:
        name = entry["Path"]
        path = model_path / name
        actual = _sha(path)
        if actual != entry["Sha256"]:
            raise ValueError(f"local model hash mismatch: {name}")
        hashes[name] = actual
    return source, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data", type=Path, default=Path("results/ruler_round2_data/engineering.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/ruler_round2_engineering"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("engineering requires CUDA")
    if args.data.name != "engineering.jsonl":
        raise ValueError("this runner only reads the engineering split")
    manifest_path = args.data.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_data_hash = manifest["splits"]["engineering"]["sha256"]
    if manifest["status"] != "complete" or _sha(args.data) != expected_data_hash:
        raise ValueError("engineering input manifest/hash is incomplete or invalid")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(rows) != 32 or any(row["split"] != "engineering" or row["max_new_tokens"] != 128 for row in rows):
        raise ValueError("expected exactly32 engineering examples and128 inclusive new-token limits")
    if any(row["input_tokens"] != len(row["input_ids"]) or row["input_tokens"] + 128 > 16384 for row in rows):
        raise ValueError("final input/output token budget invalid")
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("resume requires an existing output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(args.output / "run.log", mode="a"), logging.StreamHandler(sys.stdout)])
    logging.info("Verifying local model source hashes before model load")
    source, model_hashes = _verified_model_hashes(args.model)
    code_paths = [Path(__file__), *(Path("gqa_probe") / name for name in
                  ("capture.py", "oracle_attention.py", "torch_selection.py", "ruler_inference.py", "selection.py", "paired_statistics.py"))]
    stable_config = dict(schema_version=1, split="engineering", model=str(args.model.resolve()),
                         data=str(args.data.resolve()), data_sha256=expected_data_hash,
                         data_manifest_sha256=_sha(manifest_path),
                         model_source=source, model_source_manifest_sha256=_sha(args.model / "source_manifest.json"),
                         model_file_sha256=model_hashes, model_dtype="bfloat16", attention="grouped_fp32_dense_decode_sdpa_prefill",
                         code_sha256={str(p): _sha(p) for p in code_paths}, quantization=QUANTIZATION,
                         methods=[dict(bitmode="BF16", kv_bits=16), dict(bitmode="KV4", kv_bits=4)],
                         max_new_tokens=128, generation="greedy_eos_aware_inclusive_prefill_first_token",
                         torch_threads=8, seed=20261118, tf32=False,
                         python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), gpu_capability=torch.cuda.get_device_capability(),
                         quantization_cache_storage="dequantized_model_dtype_numeric_reference")
    config_id = hashlib.sha256(json.dumps(stable_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    config = dict(config_id=config_id, created_utc=datetime.now(timezone.utc).isoformat(), **stable_config)
    records_path = args.output / "predictions.jsonl.gz"
    records = []
    if args.resume:
        previous = json.loads((args.output / "config.json").read_text())
        if previous["config_id"] != config_id:
            raise ValueError("resume configuration/code/data hash differs")
        config = previous
        if records_path.exists():
            with gzip.open(records_path, "rt") as stream:
                records = [json.loads(line) for line in stream]
        summarize_engineering(records)  # Reject duplicate records before resuming.
        if any(record["config_id"] != config_id for record in records):
            raise ValueError("prediction configuration hash differs")
    else:
        _write_json(args.output / "config.json", config)
    _write_json(args.output / "progress.json", dict(status="loading_model", completed_generations=len(records), expected_generations=64, config_id=config_id))
    torch.set_num_threads(8)
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map="cuda",
                                                low_cpu_mem_usage=True, local_files_only=True).eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))
    seen = {(row["sample_id"], row["bitmode"]) for row in records}
    started_all = time.perf_counter()
    status = "running"
    try:
        for sample_index, row in enumerate(rows):
            for mode, bits in (("BF16", 16), ("KV4", 4)):
                if (row["sample_id"], mode) in seen:
                    continue
                logging.info("START sample=%d/32 id=%s mode=%s tokens=%d", sample_index + 1, row["sample_id"], mode, row["input_tokens"])
                state = prefill_model(model, row["input_ids"], kv_bits=bits)
                attention = OracleAttention("dense", gold_spans=row["gold_spans"])
                result = generate_from_prefill(model, state, 128, eos_ids, attention)
                prediction = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                eos_reached = bool(result.token_ids and result.token_ids[-1] in eos_ids)
                record = dict(config_id=config_id, sample_id=row["sample_id"], task=row["task"], split="engineering",
                              bitmode=mode, kv_bits=bits, references=row["references"], prediction=prediction,
                              token_ids=result.token_ids, generated_tokens=len(result.token_ids),
                              eos_token_ids=eos_ids, eos_reached=eos_reached, reached_token_limit=len(result.token_ids) == 128,
                              truncated=len(result.token_ids) == 128 and not eos_reached,
                              stopping_reason="eos" if eos_reached else "max_new_tokens",
                              max_new_tokens=128, consumed_decode_steps=len(result.token_ids) - 1,
                              input_tokens=row["input_tokens"], input_ids_sha256=row["source_metadata"]["input_ids_sha256"],
                              source_metadata=row["source_metadata"], prefill_seconds=state.elapsed_seconds,
                              decode_seconds=result.elapsed_seconds, total_reference_seconds=state.elapsed_seconds + result.elapsed_seconds,
                              attention_stats=result.attention_stats, **score_prediction(prediction, row["references"]))
                # A complete gzip member per prediction survives process interruption.
                with gzip.open(records_path, "at", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
                seen.add((row["sample_id"], mode))
                summary = summarize_engineering(records)
                summary.update(config_id=config_id, status="running", expected_generations=64,
                               elapsed_run_seconds=time.perf_counter() - started_all)
                _write_json(args.output / "summary.json", summary)
                _write_json(args.output / "progress.json", dict(status="running", completed_generations=len(records), expected_generations=64,
                                                               last_sample_id=row["sample_id"], last_bitmode=mode, config_id=config_id))
                logging.info("DONE %d/64 mode=%s score=%.3f complete=%s tokens=%d eos=%s prefill=%.3fs decode=%.3fs prediction=%r",
                             len(records), mode, record["official_score"], record["complete_correct"], len(result.token_ids), eos_reached,
                             state.elapsed_seconds, result.elapsed_seconds, prediction)
                del state, attention, result
                if mode == "BF16":
                    task_baselines = [r for r in records if r["task"] == row["task"] and r["bitmode"] == "BF16"]
                    if len(task_baselines) >= 4 and all(r["official_score"] == 0 for r in task_baselines[-4:]):
                        status = "stopped_systematic_bf16_failures"
                        raise RuntimeError("Four consecutive zero-score BF16 engineering predictions; diagnose template/answer prefix before continuing")
                if summary["paired_prompts"] >= 4 and summary["paired"]["official_score_delta_kv4_minus_bf16"] <= -.25:
                    logging.warning("KV4_DISTINCTLY_WORSE paired official delta=%.3f on%d prompts", summary["paired"]["official_score_delta_kv4_minus_bf16"], summary["paired_prompts"])
        status = "complete"
    except BaseException:
        if status == "running":
            status = "failed"
        logging.exception("Engineering run interrupted")
        raise
    finally:
        summary = summarize_engineering(records)
        summary.update(config_id=config_id, status=status, expected_generations=64,
                       elapsed_run_seconds=time.perf_counter() - started_all,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                       predictions_sha256=_sha(records_path) if records_path.exists() else None)
        _write_json(args.output / "summary.json", summary)
        _write_json(args.output / "progress.json", dict(status=status, completed_generations=len(records), expected_generations=64, config_id=config_id))
        logging.info("FINAL status=%s completed=%d/64 paired=%s", status, len(records), summary["paired"])


if __name__ == "__main__":
    main()
