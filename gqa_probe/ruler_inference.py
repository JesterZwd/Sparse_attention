"""Reusable dense prefill and EOS-aware greedy inference for Round 2.

The prefill's first argmax is the first generated token. A request for N new
tokens therefore consumes at most N-1 one-token decode forwards. The original
prefill cache is immutable by convention: each generation shallow-copies its
object and separately copies both cache lists. DynamicCache.update concatenates
into new tensors, so prefix tensor storage can safely remain shared.
"""

from copy import copy
from dataclasses import dataclass
from numbers import Integral
import time

import torch
from transformers import DynamicCache

from .capture import KV4Cache
from .oracle_attention import OracleAttention


@dataclass(frozen=True)
class PrefillState:
    cache: DynamicCache
    last_logits: torch.Tensor  # [1,vocab], on the model's output device.
    input_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    attention_stats: dict
    elapsed_seconds: float


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _input_tensor(model, input_ids):
    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(input_ids, device=device)
    if ids.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("input_ids must contain integer final-token IDs")
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("prefill supports one nonempty token sequence")
    return ids.to(dtype=torch.long)


@torch.inference_mode()
def prefill_model(model, input_ids, kv_bits=4):
    """Return an immutable reusable prefix with dense SDPA prefill.

    ``kv_bits=4`` uses the established post-RoPE KV4Cache numeric reference,
    with 32-channel groups and dequantized model-dtype storage. ``kv_bits=16``
    uses DynamicCache with the model's native dtype (BF16 in formal runs).
    Final chat-template IDs may be a list or integer tensor [T]/[1,T].
    Input placement is outside the measured model-prefill interval.
    """
    if isinstance(kv_bits, bool) or not isinstance(kv_bits, Integral) or kv_bits not in {4, 16}:
        raise ValueError("kv_bits must be 4 or 16")
    ids = _input_tensor(model, input_ids)
    cache = KV4Cache() if kv_bits == 4 else DynamicCache()
    _synchronize(ids.device)
    started = time.perf_counter()
    # All prefills delegate original SDPA, including a one-token prompt.
    with OracleAttention("dense"):
        result = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    last_logits = result.logits[:, -1, :].clone()
    _synchronize(last_logits.device)
    elapsed = time.perf_counter() - started
    return PrefillState(cache=cache, last_logits=last_logits,
                        input_tokens=ids.shape[1], elapsed_seconds=elapsed)


def _copy_prefix(cache):
    if not isinstance(cache, (DynamicCache, KV4Cache)):
        raise ValueError("prefill cache must be DynamicCache or KV4Cache")
    result = copy(cache)
    result.key_cache = list(cache.key_cache)
    result.value_cache = list(cache.value_cache)
    result._seen_tokens = int(cache._seen_tokens)
    return result


@torch.inference_mode()
def generate_from_prefill(model, state, max_new_tokens, eos_token_ids, attention):
    """Greedily generate up to max_new_tokens, stopping before consuming EOS.

    Timing includes cache-object/list copies, greedy argmax, attention callbacks
    and all model forwards, with CUDA synchronization at its boundaries. It
    excludes final aggregate-stat materialization. For shadow runs, reported
    ``attention_stats['shadow_audit_seconds']`` identifies the audit component.
    Returned token IDs include EOS if generated; no text decoding is performed.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if eos_token_ids is None:
        eos = set()
    elif isinstance(eos_token_ids, Integral):
        eos = {int(eos_token_ids)}
    else:
        eos = {int(token) for token in eos_token_ids}
    device = state.last_logits.device
    if state.last_logits.ndim != 2 or state.last_logits.shape[0] != 1:
        raise ValueError("prefill last_logits must have shape [1,vocab]")
    _synchronize(device)
    started = time.perf_counter()
    cache = _copy_prefix(state.cache)
    generated = [int(state.last_logits.argmax(-1).item())]
    attention.step = -1
    with attention:
        while len(generated) < max_new_tokens and generated[-1] not in eos:
            attention.step = len(generated) - 1
            token = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            result = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            generated.append(int(result.logits[0, -1].argmax().item()))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return GenerationResult(token_ids=generated, attention_stats=attention.stats(),
                            elapsed_seconds=elapsed)
