import torch

from gqa_probe import scorer_int
from gqa_probe.compact_summary import (
    ChannelCalibration,
    CompactOnline,
    CompactPrefix,
    encode_summary,
    fit_channel_statistics,
    records_as_bytes,
    scores,
)
from gqa_probe.pagepack import summary_pages
from gqa_probe.vskip_summary import decoded, unpack_codes


def test_full_width_mean4_radius2_record_layout():
    mean = torch.tensor([-7., 7., 0., -1.] * 32).reshape(1, 1, 128)
    radius = torch.tensor([0., 1., 2., 3.] * 32).reshape_as(mean)
    lengths = torch.full((1, 1, 1), 32)
    record = encode_summary(mean * lengths, mean - radius, mean + radius, lengths, "mr4")
    payload, mean_scale, radius_scale, count = record
    assert payload.shape == (1, 1, 96)
    codes = unpack_codes(payload, 6).int() & 63
    torch.testing.assert_close(codes, ((mean.int() & 15) | (radius.int() << 4)))
    assert mean_scale.item() == radius_scale.item() == 1
    assert count.item() == 32
    physical = records_as_bytes(record, "mr4")
    assert physical.shape == (1, 1, 112)
    assert physical[0, 0, 96:].tolist() == [0, 60, 0, 60, 32, *([0] * 11)]


def test_selected_channels_preserve_full_query_scaling():
    generator = torch.Generator().manual_seed(72)
    key = torch.randn(1, 2, 64, 128, generator=generator)
    q = torch.randn(2, 4, 128, generator=generator)
    channels = torch.stack((torch.arange(96), torch.arange(32, 128)))
    q[0, :, 127] = q[1, :, 0] = 12
    record = CompactPrefix.build(key, 32, ("mr96",), {"mr96": channels}).records["mr96"]
    packed, sa, sb, counts = record
    reference = torch.zeros(2, 2, 128, dtype=torch.uint8)
    reference.scatter_(-1, channels[:, None, :].expand(-1, 2, -1), packed)
    expected = scorer_int.scores(q, (reference, sa, sb, counts), "mr25", 128**-.5)
    actual = scores(q, record, "mr96", 128**-.5, channels=channels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert records_as_bytes(record, "mr96").shape == (2, 2, 112)


def test_original_method_bytes_and_scores_are_unchanged():
    generator = torch.Generator().manual_seed(33)
    key = torch.randn(1, 2, 64, 128, generator=generator)
    q = torch.randn(2, 4, 128, generator=generator)
    prefix = CompactPrefix.build(key, 32, ("mr25", "q4"))
    for method in ("mr25", "q4"):
        reference = summary_pages(key[0].unflatten(1, (2, 32)), torch.full((2, 2), 32), method=method)
        record = prefix.records[method]
        for actual, expected in zip(record, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scores(q, record, method, 128**-.5),
                                   scorer_int.scores(q, reference, method, 128**-.5), rtol=0, atol=0)
        assert records_as_bytes(record, method).shape == (2, 2, 144)


def test_online_append_crosses_p64_boundary_and_does_not_mutate_prefix():
    key = torch.arange(128).view(1, 1, 1, 128).expand(1, 1, 65, 128).half() / 8
    prefix = CompactPrefix.build(key[:, :, :62], 64, ("mr4", "mr25"))
    online = CompactOnline(prefix)
    for count, expected in ((63, [63]), (64, [64]), (65, [64, 1])):
        record = online.append(key[0, :, count - 1], count)
        for method in prefix.records:
            assert record[method][-1].flatten().tolist() == expected
            assert prefix.records[method][0].shape[1] == 0
    again = CompactOnline(prefix).append(key[0, :, 62], 63)
    assert again["mr4"][-1].item() == 63
    q = torch.zeros(1, 4, 128)
    actual = scores(q, record["mr4"], "mr4", 128**-.5, temperature=1.443)
    expected = torch.tensor([64., 1.]).log().view(1, 1, 2).expand(1, 4, 2)
    torch.testing.assert_close(actual, expected)


def test_channel_calibration_measures_magnitude_and_scoring_separately():
    key = torch.zeros(1, 2, 64, 128)
    key[0, :, :, 1] = 100
    key[0, :, :, 2] = 2
    key[0, :, ::2, 2] = -4
    key[0, 1, :, 127] = 200
    q = torch.zeros(2, 4, 128)
    q[:, :, 2] = 11
    record = CompactPrefix.build(key, 32, ("mr25",)).records["mr25"]
    local = fit_channel_statistics(q, key, record)
    center, radius, _ = decoded(record, "mr25")
    expected = (q.abs()[:, :, None] * (center.abs() + .25 * radius)[:, None]).mean((1, 2))
    torch.testing.assert_close(local.score_abs.float(), expected)
    torch.testing.assert_close(local.key_abs.float(), key[0].abs().mean(1))
    calibration = ChannelCalibration()
    calibration.update(3, q, key, record)
    calibration.update(3, q * 2, key, record)
    assert calibration.statistics[3].observations == 2
    frozen = calibration.freeze(2)
    assert frozen["k"][3].tolist() == [[1, 2], [1, 127]]
    assert frozen["score"][3].tolist() == [[0, 2], [0, 2]]
