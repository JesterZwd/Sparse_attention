"""Run reproducible RULER page-summary calibration, atomically by prompt."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
import transformers

from gqa_probe.paired_statistics import score_prediction
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill, prefill_model
from gqa_probe.page_read_budget import page_history, read_ledger
from gqa_probe.page_summary import PrefixSummary
from gqa_probe.summary_attention import LAYOUT_METHODS, Policy, SummaryAttention


ROOT = Path("results/round3_summary_bytes")
BASE = "15c100bda2ab78f9a1aa9ff56ec34a3a60f519ee"
MODEL = Path("models/Qwen3-4B-Instruct-2507")
CODE_FILES = (
    "scripts/run_ruler_round3.py", "scripts/report_ruler_round3.py", "scripts/prepare_ruler.py",
    "gqa_probe/ruler_data.py", "gqa_probe/summary_attention.py", "gqa_probe/page_summary.py",
    "gqa_probe/page_read_budget.py", "gqa_probe/quantization_round2.py", "gqa_probe/oracle_attention.py",
    "gqa_probe/ruler_inference.py", "gqa_probe/paired_statistics.py", "gqa_probe/ruler_experiment.py",
    "gqa_probe/capture.py", "gqa_probe/selection.py",
)


def load_rows(path, split):
    manifest = json.loads((path / "manifest.json").read_text())
    block = manifest["splits"][split]
    source = path / block["path"]
    if manifest["status"] != "complete" or _sha(source) != block["sha256"]:
        raise ValueError("dataset receipt does not match frozen rows")
    if manifest["generator"]["commit"] != "db43f92ff634bbce1c33d1fb15858aec4bba509a":
        raise ValueError("unexpected RULER commit")
    length = manifest["configuration"]["max_seq_length"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if len(rows) != block["count"] or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("dataset row count or identities do not match receipt")
    if any(len(r["input_ids"]) != r["input_tokens"] or r["input_tokens"] + r["max_new_tokens"] > length for r in rows):
        raise ValueError("data exceeds the configured context length")
    return manifest, length, rows


def calibration_policies(cohort, *, sram_slice):
    ones = [Policy("dense_bf16"), Policy("dense_kv4"), Policy("oracle", quota_percent=5),
            Policy("oracle", quota_percent=10)]
    ones.extend(Policy(method, 32, percent) for percent in (10, 15, 20) for method in LAYOUT_METHODS)
    if cohort in ("16k", "fwe"):
        ones.extend(Policy(method, 64, 15) for method in LAYOUT_METHODS)
        ones.extend(Policy(method, 32, 15, resident_bytes=sram_slice) for method in LAYOUT_METHODS)
    return ones


def feasible_for_row(policy, row, dim):
    if policy.percent is None:
        return True
    start = row["input_tokens"] + 1
    for tokens in range(start, start + row["max_new_tokens"] - 1):
        hist = page_history(tokens, policy.page_size, dim)
        if not read_ledger(hist, policy.method, dim, percent=policy.percent,
                           resident_bytes=policy.resident_bytes)["feasible"]:
            return False
    return True


@torch.inference_mode()
def run_prompt(model, tokenizer, eos_ids, row, policies, context, *, audit):
    ids = row["input_ids"]
    gold = row["gold_spans"]
    dim = model.config.head_dim
    layer_count = model.config.num_hidden_layers
    active = [p for p in policies if feasible_for_row(p, row, dim)]
    sizes = {p.page_size for p in active if p.percent is not None}
    prefix_timing = {}
    prefixes = {}
    state = None
    state_quantizer = None
    predictions = []
    for policy in policies:
        common = dict(sample_id=row["sample_id"], task=row["task"], split=row["split"],
                      context=context, policy=policy.id, kv_quantizer="bf16" if policy.method == "dense_bf16" else "affine32",
                      input_tokens=row["input_tokens"], max_new_tokens=row["max_new_tokens"], **asdict(policy))
        if policy not in active:
            predictions.append(dict(common, status="infeasible"))
            continue
        if state_quantizer != common["kv_quantizer"]:
            state = None
            state_quantizer = common["kv_quantizer"]
            if state_quantizer == "bf16":
                state = prefill_model(model, ids, kv_bits=16)
            else:
                state = prefill_for_quantization(model, ids, "affine32")
                for size in sorted(sizes):
                    start = time.perf_counter()
                    prefixes[size] = tuple(PrefixSummary.build(key, size) for key in state.cache.key_cache)
                    torch.cuda.synchronize()
                    prefix_timing[str(size)] = time.perf_counter() - start
        summary = {policy.page_size: prefixes[policy.page_size]} if policy.percent is not None else {}
        if policy.method == "dense_kv4" and audit:
            summary = prefixes
        attention = SummaryAttention(policy, gold_spans=gold, prefixes=summary, gather_check=audit,
                                     audit_policies=active)
        result = generate_from_prefill(model, state, row["max_new_tokens"],
                                       eos_ids, attention)
        stats = result.attention_stats
        if stats["decode_layer_calls"] != (len(result.token_ids) - 1) * layer_count:
            raise ValueError("not every model layer used the configured decode")
        eos = result.token_ids[-1] in eos_ids
        text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
        text = re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip()
        stats["byte_ledger"].pop("summary_prefill_materialization_bytes", None)
        if policy.percent is not None:
            hist = page_history(row["input_tokens"], policy.page_size, dim)
            accounting = read_ledger(hist, policy.method, dim, percent=policy.percent,
                                     resident_bytes=policy.resident_bytes)
            stats["byte_ledger"]["summary_prefill_materialization_bytes"] = (
                accounting["summary_prefill_materialization_bytes"] * layer_count * model.config.num_key_value_heads)
        predictions.append(dict(common, status="complete", generated_tokens=len(result.token_ids),
                                decode_tokens=len(result.token_ids) - 1, eos=eos, truncated=not eos,
                                generation_seconds=result.elapsed_seconds, **score_prediction(text, row["references"]),
                                **stats))
    return dict(sample_id=row["sample_id"], task=row["task"], input_ids_sha256=row["source_metadata"]["input_ids_sha256"],
                prefix_reference_build_seconds=prefix_timing, predictions=predictions)


def run_cohort(model, tokenizer, eos_ids, args, name, sram_slice, formal=None):
    split = "test" if formal else "calibration"
    manifest, context, rows = load_rows(args.root / "data" / name, split)
    output = args.root / split / name
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        rows = rows[:1]
    policies = [Policy(**params) for params in formal["policies"]] if formal else calibration_policies(
        name, sram_slice=sram_slice)
    signature = dict(base_commit=BASE, context=context, split=split, sample_ids=[r["sample_id"] for r in rows],
                     policies=[asdict(p) for p in policies], model_hashes=args.model_hashes,
                     dependencies={"torch":torch.__version__, "transformers":transformers.__version__},
                     dataset_manifest_sha256=_sha(args.root / "data" / name / "manifest.json"),
                     input_sha256=manifest["splits"][split]["sha256"], source_sha256={f:_sha(Path(f)) for f in CODE_FILES},
                     gpu=torch.cuda.get_device_name(), sram_total_bytes=args.sram_bytes,
                     formal_sha256=_sha(args.formal) if formal else None)
    config_path = output / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != signature:
            raise ValueError(f"output already exists with different source or settings: {output}")
    else:
        _write_json(config_path, signature)
    directory = output / "prompts"
    directory.mkdir(exist_ok=True)
    receipt = []
    for idx, row in enumerate(rows, 1):
        key = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:20]
        path = directory / f"{key}.json"
        if not path.exists():
            bundle = run_prompt(model, tokenizer, eos_ids, row, policies, context, audit=True)
            _write_json(path, bundle)
        bundle = json.loads(path.read_text())
        if (bundle["sample_id"] != row["sample_id"] or
                bundle["input_ids_sha256"] != row["source_metadata"]["input_ids_sha256"] or
                [r["policy"] for r in bundle["predictions"]] != [p.id for p in policies]):
            raise ValueError(f"checkpoint identity/policies mismatch: {path}")
        receipt.append([path.name, _sha(path)])
        print(f"{split}/{name}: {idx}/{len(rows)} {row['task']}", flush=True)
    rows_out = [r for name, _ in receipt for r in json.loads((directory / name).read_text())["predictions"]]
    _write_json(output / "per_prompt.json", rows_out)
    _write_json(output / "complete.json", dict(config_sha256=_sha(config_path),
                per_prompt_sha256=_sha(output / "per_prompt.json"), prompts=receipt,
                num_predictions=len(rows_out)))
    return rows_out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--cohorts", nargs="+", default=["16k", "32k"])
    parser.add_argument("--sram-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", type=Path)
    args = parser.parse_args()
    formal = json.loads(args.formal.read_text()) if args.formal else None
    if formal:
        from scripts.report_ruler_round3 import validate_frozen_formal
        validate_frozen_formal(args.root, formal)
    torch.set_num_threads(8)
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifests = [load_rows(args.root / "data" / name, "test" if formal else "calibration")[0] for name in args.cohorts]
    _, args.model_hashes = _verified_model(args.model, manifests[0])
    if any(m["tokenizer"] != manifests[0]["tokenizer"] for m in manifests):
        raise ValueError("all cohorts must use the verified tokenizer")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        low_cpu_mem_usage=True, local_files_only=True).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    sram_slice = args.sram_bytes // (model.config.num_hidden_layers * model.config.num_key_value_heads)
    all_rows = [r for name in args.cohorts for r in run_cohort(model, tokenizer, eos_ids, args, name, sram_slice, formal)]
    vt = [r for r in all_rows if r["task"] == "vt" and r["method"] == "dense_bf16"]
    vt_by_length = {c:[r["official_score"] for r in vt if r["context"] == c] for c in {r["context"] for r in vt}}
    if not args.smoke and not formal and any(sum(v)/len(v) < .90 for v in vt_by_length.values()):
        fallback = args.root / "data" / "fwe" / "manifest.json"
        if not fallback.exists() or json.loads(fallback.read_text())["status"] != "complete":
            subprocess.run([sys.executable, "-m", "scripts.prepare_ruler", "--config", "configs/ruler_round3_fwe.json",
                            "--output", str(fallback.parent)], check=True, env=os.environ)
        run_cohort(model, tokenizer, eos_ids, args, "fwe", sram_slice)
    if not args.smoke:
        from scripts.report_ruler_round3 import report
        report(args.root, formal=formal)
    print("Round 3 finished", flush=True)


if __name__ == "__main__":
    main()
