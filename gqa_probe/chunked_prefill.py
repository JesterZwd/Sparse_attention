"""Bounded-memory affine32 prefill for a single unpadded Qwen3 sequence."""

from contextlib import nullcontext
from numbers import Integral
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .oracle_attention import OracleAttention
from .quantization_round2 import AffineKV4Cache, affine_quantize_dequantize
from .ruler_inference import PrefillState, _input_tensor, _synchronize


class PreallocatedAffineKV4Cache(AffineKV4Cache):
    """Expose visible prefix views backed by a single buffer per layer and tensor.

    ``generate_from_prefill`` copies the object and visible-cache lists while
    sharing the backing storage. Sequential decode branches overwrite the
    reserved generation tail. The visible prefix in the original state is
    immutable. Run branches one at a time because their tails share storage.
    """

    def __init__(self, capacity):
        super().__init__(group_size=32)
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
            raise ValueError("KV capacity must be a positive integer")
        self.capacity = int(capacity)
        self._key_storage = []
        self._value_storage = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start = self.get_seq_length(layer_idx)
        stop = start + key_states.shape[-2]
        if stop > self.capacity:
            raise ValueError(f"KV cache needs {stop} tokens, exceeding capacity {self.capacity}")

        # Match AffineKV4Cache's post-RoPE, per-token affine32 operation order.
        keys = affine_quantize_dequantize(key_states, self.group_size)
        values = affine_quantize_dequantize(value_states, self.group_size)
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
            self._key_storage.append(None)
            self._value_storage.append(None)
        if self._key_storage[layer_idx] is None:
            self._key_storage[layer_idx] = keys.new_empty(*keys.shape[:-2], self.capacity, keys.shape[-1])
            self._value_storage[layer_idx] = values.new_empty(*values.shape[:-2], self.capacity, values.shape[-1])

        key_store = self._key_storage[layer_idx]
        value_store = self._value_storage[layer_idx]
        key_store[..., start:stop, :].copy_(keys)
        value_store[..., start:stop, :].copy_(values)
        self.key_cache[layer_idx] = key_store[..., :stop, :]
        self.value_cache[layer_idx] = value_store[..., :stop, :]
        if layer_idx == 0:
            self._seen_tokens += stop - start
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _chunk_attention_mask(positions, stop, dtype):
    # Qwen slices this 4D bias to visible KV length in SDPA. Keep its backing
    # row stride aligned even when an odd-length tail has fewer valid columns.
    columns = torch.arange((stop + 7) // 8 * 8, device=positions.device)
    bias = torch.zeros((1, 1, positions.numel(), columns.numel()), dtype=dtype, device=positions.device)
    return bias.masked_fill_(columns > positions[:, None], torch.finfo(dtype).min)


@torch.inference_mode()
def prefill_chunked(model, input_ids, *, chunk_size=2048, max_new_tokens=128):
    """Return the usual PrefillState with a reusable generation tail.

    Chunked dense SDPA sees all prior tokens and the current chunk up to each
    query position. The cache keeps affine32-dequantized values in model dtype.
    On CUDA, require a fused SDPA kernel to bound peak attention memory.
    """
    for name, value in (("chunk_size", chunk_size), ("max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if model.config._attn_implementation != "sdpa" or model.config.sliding_window is not None:
        raise ValueError("chunked affine32 prefill requires full-context Qwen3 SDPA")

    ids = _input_tensor(model, input_ids)
    count = ids.shape[1]
    dtype = model.get_input_embeddings().weight.dtype
    cache = PreallocatedAffineKV4Cache(count + max_new_tokens - 1)
    kernels = (sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
               if ids.device.type == "cuda" else nullcontext())
    _synchronize(ids.device)
    started = time.perf_counter()
    with OracleAttention("dense"), kernels:
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            positions = torch.arange(start, stop, device=ids.device)
            mask = _chunk_attention_mask(positions, stop, dtype) if start and stop - start > 1 else None
            logits = model(ids[:, start:stop], past_key_values=cache, use_cache=True,
                           position_ids=positions.unsqueeze(0), cache_position=positions,
                           attention_mask=mask, logits_to_keep=1).logits
    last_logits = logits[:, -1, :].clone()
    _synchronize(last_logits.device)
    return PrefillState(cache=cache, last_logits=last_logits, input_tokens=count,
                        elapsed_seconds=time.perf_counter() - started)
