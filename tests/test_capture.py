import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gqa_probe.attention import attention_output
from gqa_probe.capture import KV4Cache, TraceRecorder, capture_sample, quantize_dequantize


def test_quantization_has_explicit_rounding_and_zero_group_behavior():
    x = torch.tensor([[-7., -2.4, 2.6, 7.], [0., 0., 0., 0.]])
    expected = torch.tensor([[-7., -2., 3., 7.], [0., 0., 0., 0.]])
    torch.testing.assert_close(quantize_dequantize(x, 4), expected)


def test_cache_quantizes_new_kv_and_preserves_previous_tokens():
    cache = KV4Cache(group_size=4)
    x = torch.tensor([[[[-7., -2.4, 2.6, 7.]]]])
    k, v = cache.update(x, x, 0)
    torch.testing.assert_close(k, torch.tensor([[[[-7., -2., 3., 7.]]]]))
    k2, v2 = cache.update(torch.zeros_like(x), torch.zeros_like(x), 0)
    assert k2.shape[-2] == 2
    torch.testing.assert_close(k2[..., :1, :], k)
    torch.testing.assert_close(v2[..., :1, :], v)


def test_tiny_model_trace_matches_attention_used_by_dense_path(tmp_path):
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    cache = KV4Cache(group_size=4)
    recorder = TraceRecorder(tmp_path, layers=[0, 1], group_size=4)
    with torch.inference_mode(), recorder:
        model(torch.tensor([[1, 2, 3, 4]]), past_key_values=cache, use_cache=True)
        recorder.sample_id = "tiny"
        recorder.step = 0
        model(torch.tensor([[5]]), past_key_values=cache, use_cache=True)
    assert len(recorder.records) == 4
    for metadata in recorder.records:
        with np.load(tmp_path / metadata["file"], allow_pickle=False) as trace:
            assert trace["logits"].shape == (2, 5)
            assert trace["values"].shape == (5, 8)
            dense = attention_output(trace["logits"], trace["values"])
            np.testing.assert_allclose(dense, trace["dense_output"], atol=1e-6, rtol=1e-5)
            group = metadata["kv_group"]
            np.testing.assert_array_equal(trace["values"], cache.value_cache[
                metadata["layer"]][0, group].float().numpy())


def test_capture_does_not_decode_beyond_eos(tmp_path):
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()  # All logits tie; deterministic argmax is token 0.
    recorder = TraceRecorder(tmp_path, layers=[0], group_size=4)
    with torch.inference_mode(), recorder:
        generated = capture_sample(model, torch.tensor([[1, 2, 3]]), recorder,
                                   max_decode_steps=4, eos_token_ids=[0], group_size=4)
    assert generated == [0]
    assert recorder.records == []
