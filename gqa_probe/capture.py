"""Dense KV4 *numerical reference*: dequantized cache, no packed-storage claims."""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


QUANTIZATION = {
    "kind": "symmetric_int4_quantize_dequantize_reference",
    "bits": 4, "integer_range": [-7, 7], "zero_point": 0,
    "axis": "per_token_per_kv_head_channel_groups", "group_size": 32,
    "scale_dtype": "float16", "rounding": "nearest_even",
    "key_position": "after_rope", "cache_storage": "dequantized_bfloat16",
    "residual_unquantized_tokens": 0,
    "decode_attention": "float32_qk_softmax_av_then_cast_to_model_dtype",
    "prefill_attention": "sdpa_with_quantized_dequantized_kv",
}


def quantize_dequantize(tensor, group_size=32):
    if group_size <= 0 or tensor.shape[-1] % group_size:
        raise ValueError("head dimension must be divisible by quantization group size")
    grouped = tensor.float().reshape(*tensor.shape[:-1], -1, group_size)
    # Clamp to FP16's smallest subnormal before encoding the scale.
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7).clamp_min(2 ** -24)
    scale = scale.to(torch.float16).float()
    codes = torch.round(grouped / scale).clamp(-7, 7)
    return (codes * scale).reshape_as(tensor).to(tensor.dtype)


class KV4Cache(DynamicCache):
    def __init__(self, group_size=32):
        super().__init__()
        self.group_size = group_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(quantize_dequantize(key_states, self.group_size),
                              quantize_dequantize(value_states, self.group_size),
                              layer_idx, cache_kwargs)


class TraceRecorder:
    """Intercept the attention interface after q/k norms, RoPE and cache update."""

    def __init__(self, directory, layers, group_size=32, groups=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layers = set(layers)
        self.groups = None if groups is None else set(groups)
        self.group_size = group_size
        self.records = []
        self.sample_id = "unset"
        self.step = -1
        self.original = None

    def __enter__(self):
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.attention
        return self

    def __exit__(self, *_):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.original

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        if query.shape[0] != 1:
            raise ValueError("Trace reference supports batch=1 only")
        if attention_mask is not None and torch.any(attention_mask != 0):
            raise ValueError("decode Trace requires all stored tokens to be valid and visible")
        repeat = query.shape[1] // key.shape[1]
        repeated_key = key.repeat_interleave(repeat, dim=1)
        repeated_value = value.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(query.float(), repeated_key.float().transpose(-1, -2)) * kwargs["scaling"]
        dense = torch.softmax(scores, dim=-1) @ repeated_value.float()
        if self.step >= 0 and module.layer_idx in self.layers:
            for group in range(key.shape[1]):
                if self.groups is not None and group not in self.groups:
                    continue
                filename = f"trace_{len(self.records):06d}.npz"
                begin, end = group * repeat, (group + 1) * repeat
                np.savez_compressed(
                    self.directory / filename,
                    logits=scores[0, begin:end, 0].cpu().numpy(),
                    values=value[0, group].float().cpu().numpy(),
                    dense_output=dense[0, begin:end, 0].cpu().numpy(),
                )
                self.records.append(dict(file=filename, sample_id=self.sample_id,
                                         step=self.step, layer=module.layer_idx,
                                         kv_group=group, valid_tokens=key.shape[-2]))
        return dense.to(query.dtype).transpose(1, 2).contiguous(), None


def retrieval_prompt(tokenizer, length, seed):
    """Custom reproducible smoke/pilot data; this is NOT official RULER."""
    rng = random.Random(seed)
    marker = "EXPERIMENT_PAYLOAD_MARKER"
    template = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
                                             tokenize=False, add_generation_prompt=True)
    before, after = template.split(marker)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = encode(before + "Read the records and find the exact value of the requested key.\n")
    target_key = f"target_{seed:04d}"
    answer = f"{rng.randrange(10**7, 10**8):08d}"
    needle = encode(f"\nKEY {target_key} VALUE {answer}\n")
    suffix = encode(f"\nWhat is the value of KEY {target_key}? Output only its eight digits." + after)
    remaining = length - len(prefix) - len(needle) - len(suffix)
    if remaining < 1:
        raise ValueError("prompt length too small")
    records = []
    while len(records) < remaining:
        text = "".join(f"KEY item_{rng.randrange(10**6):06d} VALUE {rng.randrange(10**8):08d}\n"
                       for _ in range(256))
        records.extend(encode(text))
    records = records[:remaining]
    position = int(len(records) * (.25 if seed % 2 else .75))
    ids = prefix + records[:position] + needle + records[position:] + suffix
    assert len(ids) == length
    return ids, {"seed": seed, "expected_answer": answer, "target_key": target_key,
                 "needle_token_offset": len(prefix) + position,
                 "input_tokens": length, "source": "custom_key_value_retrieval_pilot"}


def capture_sample(model, input_ids, recorder, max_decode_steps, eos_token_ids=(), group_size=32):
    cache = KV4Cache(group_size=group_size)
    recorder.step = -1
    result = model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    generated = [int(result.logits[0, -1].argmax())]
    for step in range(max_decode_steps):
        if generated[-1] in eos_token_ids:
            break
        recorder.step = step
        token = torch.tensor([[generated[-1]]], device=input_ids.device)
        result = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        generated.append(int(result.logits[0, -1].argmax()))
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--layers", default="0,7,14,21,28,35")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.context < 1 or args.samples < 1 or args.decode_steps < 1:
        parser.error("context, samples, and decode steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not visible; run on the workstation outside the restricted sandbox")
    args.output.mkdir(parents=True, exist_ok=False)
    traces = args.output / "traces"
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              cache_dir="models/huggingface")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir="models/huggingface",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", low_cpu_mem_usage=True,
    ).eval()
    if model.config.model_type not in {"qwen3", "qwen2"}:
        raise ValueError("this reference has only been validated for full-attention Qwen2/Qwen3")
    if args.context + args.decode_steps > model.config.max_position_embeddings:
        raise ValueError("input plus output exceeds the model context configuration")
    layers = [int(x) for x in args.layers.split(",")]
    if any(x < 0 or x >= model.config.num_hidden_layers for x in layers):
        raise ValueError("requested layer does not exist")
    groups = None if args.groups == "all" else [int(x) for x in args.groups.split(",")]
    if groups is not None and any(x < 0 or x >= model.config.num_key_value_heads for x in groups):
        raise ValueError("requested KV group does not exist")
    recorder = TraceRecorder(traces, layers=layers, groups=groups)
    manifest = dict(source_kind="real_model_custom_retrieval_pilot",
                    quantization=QUANTIZATION, model=args.model,
                    requested_revision=args.revision,
                    resolved_revision=getattr(model.config, "_commit_hash", None),
                    command=sys.argv, python=platform.python_version(),
                    torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    config=model.config.to_dict(), samples=[], records=[])
    source_manifest = Path(args.model) / "source_manifest.json"
    if source_manifest.exists():
        manifest["checkpoint_source"] = json.loads(source_manifest.read_text())
    manifest["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path("gqa_probe").glob("*.py"))}
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
    with torch.inference_mode(), recorder:
        for sample in range(args.samples):
            ids, metadata = retrieval_prompt(tokenizer, args.context, args.seed + sample)
            recorder.sample_id, recorder.step = str(sample), -1
            started = time.perf_counter()
            inputs = torch.tensor([ids], device="cuda")
            np.save(traces / f"sample_{sample}_input_ids.npy", np.asarray(ids, dtype=np.int64))
            generated = capture_sample(model, inputs, recorder, args.decode_steps, eos)
            metadata["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)
            metadata["generated_token_ids"] = generated
            metadata["elapsed_seconds_including_trace_io"] = time.perf_counter() - started
            metadata["sample_id"] = str(sample)
            metadata["input_ids_file"] = f"sample_{sample}_input_ids.npy"
            metadata["processed_decode_steps"] = len(generated) - 1
            manifest["samples"].append(metadata)
            manifest["records"] = recorder.records
            (traces / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
            print(f"sample {sample + 1}/{args.samples}: {len(recorder.records)} traces, "
                  f"answer={metadata['generated_text']!r}", flush=True)
            torch.cuda.empty_cache()
    manifest["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (traces / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"wrote {len(recorder.records)} traces to {traces}", flush=True)


if __name__ == "__main__":
    main()
