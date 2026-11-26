"""Isolate quantization-induced Qwen2.5 NaNs/loss on frozen calibration rows."""
import json
from pathlib import Path

import torch

from gqa_probe import round13_models as generic
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.quantization_round2 import affine_quantize_dequantize
from gqa_probe.ruler_inference import generate_from_prefill
from scripts import run_round13 as runner

panel = 'qwen25_32k_multivalue'
_, manifest, _, rows = runner.panel_data(runner.ROOT, panel, 'calibration')
model, tokenizer, eos, _ = runner.initialize(panel, manifest)
native_type = generic.PreallocatedNativeCache
affine_type = generic.PreallocatedAffineKV4Cache

class SelectiveCache(native_type):
    """Use the same preallocation and quantizer while isolating K and V."""
    def __init__(self, capacity, quant_k, quant_v):
        super().__init__(capacity)
        self.quant_k = quant_k
        self.quant_v = quant_v

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        if self.quant_k:
            keys = affine_quantize_dequantize(keys, 32)
        if self.quant_v:
            values = affine_quantize_dequantize(values, 32)
        return super().update(keys, values, layer_idx, cache_kwargs)

def make(row, mode):
    if mode in {'BF16', 'KV4'}:
        return runner.make_prefill(panel, model, row, 16 if mode=='BF16' else 4)
    if mode == 'BF16-prefill-then-KV4':
        state = generic.prefill_generic(model, row['input_ids'], max_new_tokens=row['max_new_tokens'], kv_bits=16)
        cache = SelectiveCache(state.cache.capacity, True, True)
        for field in ('_key_storage', '_value_storage', 'key_cache', 'value_cache'):
            setattr(cache, field, getattr(state.cache, field))
        cache._seen_tokens = state.cache._seen_tokens
        for prefix in (cache.key_cache,cache.value_cache):
            for x in prefix:
                x.copy_(affine_quantize_dequantize(x, 32))
        return generic.PrefillState(cache=cache, last_logits=state.last_logits, input_tokens=state.input_tokens, elapsed_seconds=state.elapsed_seconds)
    generic.PreallocatedAffineKV4Cache = lambda capacity: SelectiveCache(capacity, mode=='K4-only', mode=='V4-only')
    try:
        return runner.make_prefill(panel, model, row, 4)
    finally:
        generic.PreallocatedAffineKV4Cache = affine_type

log = Path('.cache/round13-cloud/qwen25-acceptance-diagnostic.json')
results = []
with torch.inference_mode():
    for row in rows[:2]:
        reference = None
        for mode in ('BF16', 'KV4', 'K4-only', 'V4-only', 'BF16-prefill-then-KV4'):
            state = make(row, mode)
            current = state.last_logits.float()
            if reference is None:
                reference = current.clone()
            output = generate_from_prefill(model, state, row['max_new_tokens'], eos, OracleAttention('dense'))
            text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
            omega = runner.score_prediction(text.strip(), row['references'])
            record = dict(sample=row['sample_id'], mode=mode, score=omega['official_score'],
                          prefill_first=int(current.argmax()), generated=len(output.token_ids),
                          finite_logits=bool(torch.isfinite(current).all()),
                          logits_relative_rmse=float((current-reference).square().mean().sqrt()/reference.square().mean().sqrt()),
                          ended_with_eos=output.token_ids[-1] in eos)
            if mode == 'KV4':
                native = generate_from_prefill(model, state, 12, eos, generic.NativeDenseAttention('dense'))
                record['native_prefix_equal']=native.token_ids == output.token_ids[:12]
            print(json.dumps(record),flush=True)
            results.append(record)
            log.write_text(json.dumps(results, indent=2)+'\n')
            del state
