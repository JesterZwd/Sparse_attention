"""Model-aware full-context reference runtime for CoMRAD experiments."""

from contextlib import nullcontext
import json
from numbers import Integral
from pathlib import Path
import time

import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache

from .chunked_prefill import PreallocatedAffineKV4Cache, _chunk_attention_mask
from .oracle_attention import OracleAttention
from .ruler_experiment import _sha
from .ruler_inference import PrefillState, _input_tensor, _synchronize


def _effective_sliding_window(config):
    window = getattr(config, "sliding_window", None)
    # Qwen2 keeps a raw window field even when its optional window is off.
    # Its causal-mask path may still read the raw value, so also check the cap.
    if (window is not None and getattr(config, "use_sliding_window", True) is False and
            window >= config.max_position_embeddings):
        return None
    return window


def model_geometry(config):
    """Read GQA, RoPE and context geometry from the model configuration."""
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if not heads or config.hidden_size % heads:
            raise ValueError("hidden size is incompatible with the query heads")
        head_dim = config.hidden_size // heads
    if heads < 1 or kv_heads < 1 or heads % kv_heads or head_dim < 1:
        raise ValueError("invalid grouped-query geometry")
    return dict(
        model_type=config.model_type,
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        query_heads_per_kv_head=heads // kv_heads,
        head_dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=getattr(config, "rope_theta", None),
        rope_scaling=getattr(config, "rope_scaling", None),
        rms_norm_eps=getattr(config, "rms_norm_eps", None),
        configured_sliding_window=getattr(config, "sliding_window", None),
        sliding_window=_effective_sliding_window(config),
    )


def verified_assets(model_path, manifest=None):
    """Check local weights and tokenizer files against their source receipt."""
    path = Path(model_path)
    source_path = path / "source_manifest.json"
    source = json.loads(source_path.read_text())
    if not source.get("verified"):
        raise ValueError("local model source manifest is not verified")
    hashes = {}
    for entry in source["files"]:
        name = entry["Path"]
        if '/' in name or name in (".", "..") or name in hashes:
            raise ValueError(f"invalid file entry in the model receipt: {name}")
        digest = _sha(path / name)
        if digest != entry["Sha256"]:
            raise ValueError(f"local model hash mismatch: {name}")
        hashes[name] = digest
    if manifest is not None and any(hashes.get(name) != digest for name, digest in manifest["tokenizer"].items()):
        raise ValueError("dataset tokenizer and local model hashes differ")
    if "model.safetensors.index.json" in hashes:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        if not set(index["weight_map"].values()).issubset(hashes):
            raise ValueError("weight shard missing from source receipt")
    elif "model.safetensors" not in hashes:
        raise ValueError("no safetensors weight files in source receipt")
    return source, hashes


def load_local_model(model_path, manifest=None, *, device="cuda"):
    """Load a verified BF16 decoder, tokenizer, EOS IDs, and geometry receipt."""
    path = Path(model_path)
    source, hashes = verified_assets(path, manifest)
    config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
    geometry = model_geometry(config)
    if geometry["head_dim"] != 128 or geometry["sliding_window"] is not None:
        raise ValueError("CoMRAD requires 128-channel full-context attention")
    if manifest is not None:
        context = manifest.get("configuration", {}).get("max_seq_length", manifest.get("context_tokens"))
        if context is not None and context > geometry["max_position_embeddings"]:
            raise ValueError("dataset context limit exceeds the native model context")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device,
        low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    modules = [layer.self_attn for layer in model.model.layers]
    geometry.update(q_norm_layers=sum(hasattr(attn, "q_norm") for attn in modules),
                    k_norm_layers=sum(hasattr(attn, "k_norm") for attn in modules),
                    attention_scaling=float(modules[0].scaling))
    receipt = dict(model_path=str(path), source_manifest_sha256=_sha(path / "source_manifest.json"),
                   source_revision=source.get("revision", source.get("snapshot_revision")), files_sha256=hashes,
                   geometry=geometry, weights_dtype="bfloat16", attention="sdpa",
                   transformers_version=transformers.__version__, torch_version=str(torch.__version__))
    return model, tokenizer, eos, receipt


class PreallocatedNativeCache(DynamicCache):
    """Store native K/V in one fixed buffer per layer with a shared tail."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self._key_storage = []
        self._value_storage = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start = self.get_seq_length(layer_idx)
        stop = start + key_states.shape[-2]
        if stop > self.capacity:
            raise ValueError("native KV cache capacity exceeded")
        if layer_idx == 0:
            self._seen_tokens += stop - start
        if layer_idx == len(self.key_cache):
            for storage, visible, states in ((self._key_storage, self.key_cache, key_states),
                                             (self._value_storage, self.value_cache, value_states)):
                buffer = states.new_empty(*states.shape[:-2], self.capacity, states.shape[-1])
                storage.append(buffer)
                visible.append(buffer[..., :0, :])
        for storage, visible, states in ((self._key_storage, self.key_cache, key_states),
                                         (self._value_storage, self.value_cache, value_states)):
            buffer = storage[layer_idx]
            buffer[..., start:stop, :].copy_(states)
            visible[layer_idx] = buffer[..., :stop, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


@torch.inference_mode()
def prefill_generic(model, input_ids, *, max_new_tokens=128, kv_bits=4, chunk_size=2048):
    """Keep an explicit native context cap and a reusable full-context prefix."""
    for name, value in (("max_new_tokens", max_new_tokens), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(kv_bits, bool) or not isinstance(kv_bits, Integral) or kv_bits not in (4, 16):
        raise ValueError("kv_bits must be 4 or 16")
    if model.config._attn_implementation != "sdpa" or _effective_sliding_window(model.config) is not None:
        raise ValueError("CoMRAD prefill requires full-context SDPA")
    ids = _input_tensor(model, input_ids)
    count = ids.shape[1]
    limit = model.config.max_position_embeddings
    if count + max_new_tokens > limit:
        raise ValueError(f"prompt plus generation reservation exceeds native context limit {limit}")
    device = ids.device
    dtype = model.get_input_embeddings().weight.dtype
    cache_cls = PreallocatedAffineKV4Cache if kv_bits == 4 else PreallocatedNativeCache
    cache = cache_cls(count + max_new_tokens - 1)
    kernels = (sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
               if device.type == "cuda" else nullcontext())
    _synchronize(device)
    started = time.perf_counter()
    with kernels:
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            positions = torch.arange(start, stop, device=device)
            mask = _chunk_attention_mask(positions, stop, dtype) if start and stop - start > 1 else None
            logits = model(ids[:, start:stop], past_key_values=cache, use_cache=True,
                           position_ids=positions.unsqueeze(0), cache_position=positions,
                           attention_mask=mask, logits_to_keep=1).logits
    last_logits = logits[:, -1, :].clone()
    _synchronize(device)
    return PrefillState(cache=cache, last_logits=last_logits, input_tokens=count,
                        elapsed_seconds=time.perf_counter()-started)


class NativeDenseAttention(OracleAttention):
    """Use the Transformers SDPA dispatch for every generation step."""

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        return self.original(module, query, key, value, attention_mask, **kwargs)
