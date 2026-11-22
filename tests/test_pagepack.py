import torch

from gqa_probe.pagepack import (
    pack_pages, packed_logits, packed_output, physical_v_mask, plan_segments,
    segment_statistics, select_plan, summary_pages,
)
from gqa_probe.quantization_round2 import affine_quantize_dequantize
from gqa_probe.vskip_summary import encode


def test_fixed_compaction_and_unsealed_protection():
    k = torch.arange(32, dtype=torch.float32).view(1, 4, 8, 1).expand(-1, -1, -1, 32)
    v = k + 2
    packed = pack_pages(k, v, [8, 8, 3, 0], "kv4", forced=[False, True, False, False])
    assert packed.key_lengths.tolist() == [[
        [4, 4, 0, 0, 0, 0, 0, 0], [1] * 8, [1, 1, 1, 0, 0, 0, 0, 0], [0] * 8,
    ]]
    assert packed.segment_starts[0, 0, :2].tolist() == [0, 4]
    torch.testing.assert_close(packed.key[0, 0, :2, 0], torch.tensor([1.5, 5.5]))
    torch.testing.assert_close(packed.value[0, 1], v[0, 1])
    torch.testing.assert_close(packed.key[0, 2, :3], k[0, 2, :3])
    assert not packed.key[0, 3].any()


def test_joint_logits_include_cardinality_and_v_only_uses_token_weights():
    k = torch.zeros(1, 1, 4, 32)
    v = torch.arange(1, 8, 2, dtype=torch.float32).view(1, 1, 4, 1).expand_as(k)
    q = torch.zeros(1, 2, 32)
    joint = pack_pages(k, v, [4], "kv2", with_summary=False)
    joint_z = packed_logits(q, joint, 1.)
    torch.testing.assert_close(joint_z[..., :2], torch.full((1, 2, 1, 2), 2.).log())
    assert torch.isneginf(joint_z[..., 2:]).all()
    torch.testing.assert_close(packed_output(joint_z, joint), torch.full((1, 2, 32), 4.))

    k[0, 0, 1, 0] = 2.
    q[0, 0, 0] = 1.
    only = pack_pages(k, v, [4], "v2", with_summary=False)
    z = packed_logits(q, only, 1.)
    assert z[0, 0, 0].tolist() == [0., 2., 0., 0.]
    gate = torch.tensor([[[[False, True, False, True]], [[True, True, False, False]]]])
    want = (torch.tensor(2.).exp() * 2 + 6) / (torch.tensor(2.).exp() + 1)
    out = packed_output(z, only, gate)
    torch.testing.assert_close(out[0, 0], want.expand(32))
    torch.testing.assert_close(out[0, 1], torch.full((32,), 2.))
    assert physical_v_mask(gate, only).tolist() == [[[[True, True, False, False]],
                                                      [[True, False, False, False]]]]


def test_mixed_partition_uses_k_and_optional_v_guard_and_singletons_are_exact():
    k = torch.ones(1, 1, 8, 32)
    k[..., -1, :] = 3.
    v = torch.ones_like(k)
    v[..., :4:2, :] = 0.
    vp = pack_pages(k, v, 8, "mixed", k_threshold=.5, with_summary=False)
    kvp = pack_pages(k, v, 8, "mixed", k_threshold=.5, v_threshold=.5, with_summary=False)
    assert vp.value_lengths[0, 0, :4].tolist() == [4, 2, 1, 1]
    assert kvp.value_lengths[0, 0, :7].tolist() == [1, 1, 1, 1, 2, 1, 1]
    torch.testing.assert_close(kvp.key[0, 0, 6], k[0, 0, 7])


def test_page_mr_record_comes_from_original_hierarchy_including_tail():
    k = torch.linspace(-3., 4., 8*32).view(1, 1, 8, 32)
    partial = torch.cat((k, 2*k), dim=1)
    partial[0, 1, 3:] = float("nan")
    lengths = [8, 3]
    stats = segment_statistics(partial, lengths)
    plan = plan_segments(partial, partial, lengths, "mixed", key_stats=stats)
    pack = pack_pages(partial, partial, lengths, "mixed", plan=plan, key_stats=stats)
    actual = summary_pages(partial, lengths, stats=stats)
    exp_tail = encode((2*k[:, :, :3]).sum(2), (2*k[:, :, :3]).amin(2), (2*k[:, :, :3]).amax(2),
                      torch.tensor([[[3]]]), "mr25")
    for a,b,c in zip(actual, exp_tail, pack.summary):
        torch.testing.assert_close(a[:, 1:], b)
        torch.testing.assert_close(a, c)
    assert torch.isfinite(pack.key).all()


def test_affine_representative_real_codes_and_selected_inactive_slots():
    k = torch.linspace(-4., 3., 2 * 4 * 32).view(1, 2, 4, 32)
    v = k.sqrt().nan_to_num(0.)
    plan = plan_segments(k, v, [4, 4], "kv2")
    ids = torch.tensor([[1, 0]])
    selected = select_plan(plan, ids, active=torch.tensor([[True, False]]))
    pack = pack_pages(k[:, [1, 0]], v[:, [1, 0]], [4, 0], "kv2", plan=selected,
                      quantized=True, with_codes=True, with_summary=False)
    expert = affine_quantize_dequantize(k[:, 1].unflatten(1, (2, 2)).mean(2))
    torch.testing.assert_close(pack.key[:, 0, :2], expert)
    assert not pack.key[:, 1].any()
    assert pack.key_codes.code.dtype == torch.uint8
    assert pack.key_codes.scale.dtype == torch.float16
    assert pack.key_codes.zero_point.dtype == torch.uint8
    assert pack.key_codes.code.shape[-1] == 16
    torch.testing.assert_close(pack.key_codes.decode()[:, 0, :2], pack.key[:, 0, :2])


def test_quantized_mixed_with_no_merges_has_no_affine_representatives():
    k = torch.arange(128, dtype=torch.float32).view(1, 1, 4, 32)
    for with_codes in (False, True):
        packed = pack_pages(k, k, [4], "mixed", k_threshold=0., v_threshold=0.,
                            quantized=True, with_codes=with_codes, with_summary=False)
        torch.testing.assert_close(packed.key, k)
        assert packed.key_lengths.tolist() == [[[1, 1, 1, 1]]]
        if with_codes:
            assert packed.key_codes.code.shape == (1, 1, 4, 16)
            assert not packed.key_codes.code.any()
