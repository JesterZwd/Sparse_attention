"""Measure Qwen3 affine32 prefill parity at 16K/32K and CUDA peaks at 64K."""

import argparse
import gc
import json
from pathlib import Path

import torch
import transformers

from gqa_probe.chunked_prefill import prefill_chunked
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round10_summary_64k")
MODEL = Path("models/Qwen3-4B-Instruct-2507")
CONTEXTS = {"16k": 16384, "32k": 32768, "64k": 65536}


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def peak(start):
    return dict(start_allocated_bytes=start, end_allocated_bytes=torch.cuda.memory_allocated(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def measured_prefill(model, row, *, chunks):
    gc.collect()
    torch.cuda.empty_cache()
    allocated = reset_peak()
    if chunks is None:
        state = prefill_for_quantization(model, row["input_ids"], "affine32")
    else:
        state = prefill_chunked(model, row["input_ids"], chunk_size=chunks,
                                max_new_tokens=row["max_new_tokens"])
    history = state.cache.key_cache + state.cache.value_cache
    if len(history) != 2 * model.config.num_hidden_layers or any(
            t.shape[-2] != row["input_tokens"] for t in history):
        raise ValueError("prefill cache has unexpected visible lengths")
    if not torch.isfinite(state.last_logits).all().item():
        raise ValueError("prefill produced non-finite logits")
    stats = dict(prefill_seconds=state.elapsed_seconds, prefill_memory=peak(allocated),
                 cache_visible_bytes=sum(t.numel() * t.element_size() for t in history),
                 cache_storage_bytes=sum(t.untyped_storage().nbytes() for t in history),
                 first_token=int(state.last_logits.argmax().item()))
    return state, stats


def measured_decode(model, tokenizer, eos, row, state, limit):
    allocated = reset_peak()
    generated = generate_from_prefill(model, state, limit, eos, OracleAttention("dense"))
    memory = peak(allocated)
    ids = generated.token_ids
    accuracy = score_prediction(tokenizer.decode(ids, skip_special_tokens=True), row["references"])
    return ids, dict(**accuracy, generated_tokens=len(ids), decode_limit=limit, ended_with_eos=ids[-1] in eos,
                     decode_seconds=generated.elapsed_seconds, decode_memory=memory)


def check_row(model, tokenizer, eos, row, context, chunks, short_decode):
    limit = min(short_decode, row["max_new_tokens"])
    reference = None
    if context < 65536:
        state, full = measured_prefill(model, row, chunks=None)
        ref_ids, full_decode = measured_decode(model, tokenizer, eos, row, state, limit)
        reference = state.last_logits.float().cpu()
        del state

    state, chunked = measured_prefill(model, row, chunks=chunks)
    actual = state.last_logits.float().cpu()
    ids, decode = measured_decode(model, tokenizer, eos, row, state, limit)
    record = dict(context=context, sample_id=row["sample_id"], input_tokens=row["input_tokens"],
                  reserved_new_tokens=row["max_new_tokens"], chunks=(row["input_tokens"] + chunks - 1) // chunks,
                  chunked=dict(**chunked, **decode))
    if reference is not None:
        error = actual - reference
        rmse = error.square().mean().sqrt()
        record["full"] = dict(**full, **full_decode)
        record["comparison"] = dict(
            logit_max_abs=float(error.abs().max()), logit_mean_abs=float(error.abs().mean()),
            logit_rmse=float(rmse), relative_logit_rmse=float(rmse / reference.square().mean().sqrt()),
            same_first_token=ids[0] == ref_ids[0], same_generated_tokens=ids == ref_ids,
            first_generated_mismatch=next((i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b),
                                          None if len(ids) == len(ref_ids) else min(len(ids), len(ref_ids))),
            same_official_score=decode["official_score"] == full_decode["official_score"],
            same_complete_answer=decode["complete_correct"] == full_decode["complete_correct"],
        )
    else:
        second_ids, second_decode = measured_decode(model, tokenizer, eos, row, state, limit)
        if ids != second_ids or state.cache.get_seq_length() != row["input_tokens"]:
            raise ValueError("reusable affine32 prefix changed across decode branches")
        record["second_branch"] = dict(**second_decode, same_generated_tokens=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "prefill_check.json")
    parser.add_argument("--contexts", nargs="+", choices=tuple(CONTEXTS), default=list(CONTEXTS))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--short-decode", type=int, default=48)
    args = parser.parse_args()
    if min(args.count, args.short_decode, args.chunk_size) < 1:
        parser.error("count, short-decode and chunk-size must be positive")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohorts = {label: load_rows(args.data / label, "calibration") for label in args.contexts}
    _, model_hashes = _verified_model(args.model, next(iter(cohorts.values()))[0])
    for label, (manifest, length, rows) in cohorts.items():
        if length != CONTEXTS[label] or len(rows) < args.count:
            raise ValueError(f"unexpected {label} calibration cohort")
        if any(model_hashes.get(name) != digest for name, digest in manifest["tokenizer"].items()):
            raise ValueError(f"{label} tokenizer provenance mismatch")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        low_cpu_mem_usage=True, local_files_only=True).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = model.generation_config.eos_token_id
    eos = set(tokens if isinstance(tokens, list) else [tokens])
    receipt = dict(quantizer="affine32", dtype="bfloat16", model=model_hashes, chunk_size=args.chunk_size,
                   device=torch.cuda.get_device_name(), cuda_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
                   torch=torch.__version__, transformers=transformers.__version__, short_decode=args.short_decode,
                   data={k: _sha(args.data / k / "manifest.json") for k in cohorts},
                   code={p: _sha(p) for p in ("scripts/check_chunked_prefill.py", "gqa_probe/chunked_prefill.py",
                                              "gqa_probe/quantization_round2.py", "gqa_probe/ruler_inference.py")},
                   records=[])
    for label, (_, context, rows) in cohorts.items():
        for row in rows[:args.count]:
            record = check_row(model, tokenizer, eos, row, context, args.chunk_size, args.short_decode)
            receipt["records"].append(record)
            _write_json(args.output, receipt)
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
