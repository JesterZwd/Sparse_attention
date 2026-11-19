"""Two prespecified engineering-only KV4 repairs, with explicit layouts.

Both caches store dequantized model-dtype K/V tensors and make no packed-memory
or speed claim. Theoretical bits assume tightly packed 4-bit codes, a separate
FP16 scale per group, and (affine only) a uint8 integer zero point per group,
with no alignment, padding, allocator, or cache-object overhead.
"""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
import time

import torch
from transformers import DynamicCache

from .capture import KV4Cache
from .oracle_attention import OracleAttention
from .ruler_inference import PrefillState, _input_tensor, _synchronize


_COMMON = dict(bits=4, axis="per_token_per_kv_head_channel_groups", scale_dtype="float16",
               scale_bits_per_group=16, positive_minimum_scale=2 ** -24,
               rounding="nearest_even", key_position="after_rope", values_quantized=True,
               residual_unquantized_tokens=0, actual_cache_storage="dequantized_model_dtype",
               weights_dtype="bfloat16", decode_attention="grouped_float32_qk_softmax_av_then_model_dtype",
               prefill_attention="dense_sdpa_with_quantized_dequantized_kv",
               theoretical_layout="packed_uint4_codes_plus_separate_fp16_scales_and_optional_uint8_zero_points_no_padding_or_object_overhead",
               packed_storage_implemented=False)
QUANTIZATION_SPECS = {
    "affine32": dict(_COMMON, name="affine32", kind="affine_uint4_quantize_dequantize_reference",
                     group_size=32, integer_range=[0, 15], zero_point_storage="uint8",
                     zero_point_bits_per_group=8, minmax_include_zero=True,
                     zero_point_formula="clamp(round_nearest_even(-minimum/encoded_fp16_scale),0,15)",
                     code_formula="clamp(round_nearest_even(x/encoded_fp16_scale)+integer_zero_point,0,15)",
                     dequantization_formula="(code-integer_zero_point)*encoded_fp16_scale",
                     theoretical_bits_per_element=4.75),
    "symmetric16": dict(_COMMON, name="symmetric16", kind="symmetric_int4_quantize_dequantize_reference",
                        group_size=16, integer_range=[-7, 7], zero_point_storage="implicit_zero",
                        zero_point_bits_per_group=0, minmax_include_zero=False,
                        zero_point_formula="0", code_formula="clamp(round_nearest_even(x/encoded_fp16_scale),-7,7)",
                        dequantization_formula="code*encoded_fp16_scale", theoretical_bits_per_element=5.),
}


def quantization_spec(spec):
    """Resolve an ID or a canonical metadata dict; reject undeclared overrides."""
    name = spec if isinstance(spec, str) else spec.get("name") if isinstance(spec, Mapping) else None
    if name not in QUANTIZATION_SPECS:
        raise ValueError("quantization spec must be affine32 or symmetric16")
    canonical = QUANTIZATION_SPECS[name]
    if isinstance(spec, Mapping) and any(key not in canonical or value != canonical[key] for key, value in spec.items()):
        raise ValueError("quantization metadata cannot override a prespecified spec")
    return deepcopy(canonical)


def affine_quantize_dequantize(tensor, group_size=32):
    """Affine uint4 with encoded FP16 scale and nearest-even integer arithmetic.

    Min/max include zero. Codes round x/scale *before* adding the integer zero
    point; this fixes the parity convention for half-step ties explicitly.
    The scale is clamped positive before FP16 encoding, including zero groups.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, Integral) or group_size < 1 or tensor.shape[-1] % group_size:
        raise ValueError("head dimension must be divisible by a positive integer group size")
    if not tensor.is_floating_point():
        raise ValueError("quantization requires floating input")
    grouped = tensor.float().reshape(*tensor.shape[:-1], -1, group_size)
    minimum = grouped.amin(-1, keepdim=True).clamp_max(0)
    maximum = grouped.amax(-1, keepdim=True).clamp_min(0)
    scale = ((maximum - minimum) / 15).clamp_min(2 ** -24).to(torch.float16).float()
    zero_point = torch.round(-minimum / scale).clamp(0, 15).to(torch.uint8)
    codes = (torch.round(grouped / scale) + zero_point.float()).clamp(0, 15)
    return ((codes - zero_point.float()) * scale).reshape_as(tensor).to(tensor.dtype)


class AffineKV4Cache(DynamicCache):
    def __init__(self, group_size=32):
        super().__init__()
        self.group_size = group_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(affine_quantize_dequantize(key_states, self.group_size),
                              affine_quantize_dequantize(value_states, self.group_size),
                              layer_idx, cache_kwargs)


def make_cache(spec):
    metadata = quantization_spec(spec)
    if metadata["name"] == "affine32":
        return AffineKV4Cache(group_size=32)
    return KV4Cache(group_size=16)


@torch.inference_mode()
def prefill_for_quantization(model, input_ids, spec):
    """Use exactly the existing dense-prefill timing/interface for either repair."""
    ids = _input_tensor(model, input_ids)
    cache = make_cache(spec)
    _synchronize(ids.device)
    started = time.perf_counter()
    with OracleAttention("dense"):
        result = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    last_logits = result.logits[:, -1, :].clone()
    _synchronize(last_logits.device)
    return PrefillState(cache=cache, last_logits=last_logits, input_tokens=ids.shape[1],
                        elapsed_seconds=time.perf_counter() - started)
