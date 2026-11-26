"""Measure native Dense parity and the target-context memory of a CoMRAD model."""

import argparse
import gc
from pathlib import Path

import torch

from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.round13_models import NativeDenseAttention, load_local_model, model_geometry, prefill_generic
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import _copy_prefix, generate_from_prefill, prefill_model
from scripts.run_ruler_round3 import load_rows

ROOT = Path("results/round13_comrad")
TARGET = Path("models/Qwen2.5-7B-Instruct")
DATA = ROOT / "data/qwen25_32k_multivalue"
OUTPUT = ROOT / "model_adaptation/qwen25"


def _peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return None


def _reset_peak(device):
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cache_geometry(model, state):
    g = model_geometry(model.config)
    tensors = state.cache.key_cache + state.cache.value_cache
    expected_shape = [1, g["num_key_value_heads"], state.input_tokens, g["head_dim"]]
    if len(tensors) != 2 * g["num_hidden_layers"] or any(list(t.shape) != expected_shape for t in tensors):
        raise ValueError("prefill cache differs from native GQA geometry")
    if not torch.isfinite(state.last_logits).all().item():
        raise ValueError("prefill produced non-finite logits")
    return dict(shape=expected_shape, layer_count=g["num_hidden_layers"],
                visible_bytes=sum(t.numel() * t.element_size() for t in tensors),
                allocated_storage_bytes=sum(t.untyped_storage().nbytes() for t in tensors))


def _logit_delta(actual, expected):
    error = actual.float() - expected.float()
    rmse = error.square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt()
    return dict(max_abs=float(error.abs().max()), rmse=float(rmse),
                relative_rmse=float(rmse / baseline.clamp_min(1e-12)),
                first_token_equal=bool((actual.argmax(-1) == expected.argmax(-1)).all()))


@torch.inference_mode()
def short_parity(model, tokenizer, eos):
    short = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with just this integer: 7"}],
                                          tokenize=True, add_generation_prompt=True)
    reference = prefill_model(model, short, kv_bits=16)
    prefix = prefill_generic(model, short, kv_bits=16, max_new_tokens=4, chunk_size=8)
    token = reference.last_logits.argmax(dim=-1).reshape(1, 1)

    def once(attention):
        attention.step = 0
        with attention:
            return model(token, past_key_values=_copy_prefix(reference.cache), use_cache=True,
                         logits_to_keep=1).logits[:, -1, :]

    native = generate_from_prefill(model, reference, 4, eos, NativeDenseAttention("dense"))
    grouped = generate_from_prefill(model, reference, 4, eos, OracleAttention("dense"))
    offset = generate_from_prefill(model, prefix, 4, eos, OracleAttention("dense"))
    comparison = _logit_delta(prefix.last_logits, reference.last_logits)
    comparison.update(native_and_grouped_same_tokens=native.token_ids == grouped.token_ids,
                      unchunked_and_offset_same_tokens=offset.token_ids == grouped.token_ids,
                      decode_logits=_logit_delta(once(OracleAttention("dense")), once(NativeDenseAttention("dense"))))
    return dict(short_prompt_tokens=len(short), short_comparison=comparison)


@torch.inference_mode()
def smoke(model, tokenizer, eos, row, *, chunk_size=2048):
    device = model.get_input_embeddings().weight.device
    short = short_parity(model, tokenizer, eos)
    tiers = {}
    for bits in (16, 4):
        _reset_peak(device)
        state = prefill_generic(model, row["input_ids"], max_new_tokens=row["max_new_tokens"],
                                kv_bits=bits, chunk_size=chunk_size)
        prefill_peak = _peak(device)
        check = _cache_geometry(model, state)
        _reset_peak(device)
        n = min(4, row["max_new_tokens"])
        generated = generate_from_prefill(model, state, n, eos, OracleAttention("dense"))
        tiers[str(bits)] = dict(cache=check, prefill_peak=prefill_peak, decode_peak=_peak(device),
                                prefill_seconds=state.elapsed_seconds, first_token=generated.token_ids[0],
                                generated_tokens=len(generated.token_ids),
                                group_steps=generated.attention_stats["kv_group_calls"])
        del state
    return dict(sample_id=row["sample_id"], instance_sha256=row["source_metadata"]["instance_sha256"],
                prompt_tokens=row["input_tokens"], reserved_tokens=row["max_new_tokens"],
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                device_total_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                **short, chunk_size=chunk_size,
                target_context=tiers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=TARGET)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--stages", choices=("smoke", "short"), default="smoke")
    args = p.parse_args()
    manifest, _, calibration = load_rows(args.data, "calibration")
    model, tokenizer, eos, receipt = load_local_model(args.model, manifest, device=args.device)
    base = dict(model=receipt, dataset_sha256=_sha(args.data / "manifest.json"),
                source_sha256={name: _sha(name) for name in
                               ("gqa_probe/round13_models.py", "scripts/check_round13_model.py",
                                "gqa_probe/quantization_round2.py", "gqa_probe/chunked_prefill.py")})
    if args.stages == "smoke":
        record = smoke(model, tokenizer, eos, calibration[0], chunk_size=args.chunk_size)
        _write_json(args.output / "smoke.json", dict(**base, **record))
    else:
        record = short_parity(model, tokenizer, eos)
        _write_json(args.output / "short_parity.json", dict(**base, **record))
    if not all(record["short_comparison"][k] for k in ("first_token_equal", "native_and_grouped_same_tokens",
                                                         "unchunked_and_offset_same_tokens")):
        raise ValueError("short-context native and adapted greedy paths diverge, see the parity receipt")
    print("Round 13 model parity receipt saved", flush=True)


if __name__ == "__main__":
    main()
