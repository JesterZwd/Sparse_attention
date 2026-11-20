import pytest
import torch

from gqa_probe.oracle_attention import grouped_logits, grouped_output
from gqa_probe.page_read_budget import kv_page, page_history, read_ledger, summary_record
from gqa_probe.page_summary import OnlineSummary, PrefixSummary, decode_vector, gathered_output, select_ranked, summary_weights


def test_full_pages_and_tiny_tail_budget():
    assert kv_page(32).total == 4864
    assert kv_page(64).total == 9728
    assert kv_page(1).total == 192
    assert summary_record("quest_max").total == 272
    assert summary_record("centroid8").total == 144
    for size in (32, 64):
        h = page_history(16321, size)
        assert h.protected[0] == 0 and h.protected[-1] == h.pages-1
        assert h.protected_kv.total == h.tail.total + (len(h.protected)-1) * h.full.total
        assert not read_ledger(h, "quest_softmax", percent=1)["feasible"]
        quest = read_ledger(h, "quest_softmax", percent=10)
        mean = read_ledger(h, "centroid8", percent=10)
        for row in (quest, mean):
            assert 0 <= row["unused_budget_bytes"] < h.full.total + 4
            assert row["valid_tail_tokens"] == 1
        assert mean["selected_pages"] > quest["selected_pages"]
        warm = read_ledger(h, "quest_max", percent=10, resident_bytes=2000)
        assert warm["summary_records_scanned"] == h.pages - 2000 // 272
    short = page_history(100, 64)
    assert short.protected == (0, 1) and short.expanded_tokens == 0


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))])
def test_incremental_int8_enclosure_ties_and_gather(size, device):
    gen = torch.Generator().manual_seed(23)
    key = torch.randn(1, 2, 4 * size + 5, 128, generator=gen).bfloat16().to(device)
    value = torch.randn(key.shape, generator=gen).bfloat16().to(device)
    query = torch.randn(1, 8, 1, 128, generator=gen).bfloat16().to(device)
    original = PrefixSummary.build(key[:, :, :size - 2], size)
    tracker = OnlineSummary(original)
    for length in range(size - 1, key.shape[-2] + 1):
        summary = tracker.append(key[0, :, length - 1], length)
    assert int(summary.counts[-1]) == 5 and int(original.tail_count) == size-2
    for encoded, minimum in ((summary.min_key, True), (summary.max_key, False)):
        decoded = decode_vector(encoded).repeat_interleave(size, dim=1)[:, :key.shape[-2]]
        assert bool((decoded <= key[0]).all() if minimum else (decoded >= key[0]).all())
    fresh = PrefixSummary.build(key[:, :, :-1], size)
    regenerated = OnlineSummary(fresh).append(key[0, :, -1], key.shape[-2])
    torch.testing.assert_close(summary.fp_centroid, regenerated.fp_centroid)
    h = page_history(key.shape[-2], size)
    weights = torch.zeros((2, h.pages), device=device)
    count = min(h.pages, len(h.protected) + 1)
    mask = select_ranked(weights, h.protected, count)
    assert mask[0].nonzero().flatten().tolist() == sorted(set(h.protected) | (
        {next(i for i in range(h.pages) if i not in h.protected)} if count > len(h.protected) else set()))
    scaling = 128**-.5
    for method in ("quest_max", "quest_softmax", "centroid8", "centroid_fp32"):
        scores = summary_weights(query, summary, method, scaling)
        assert scores.shape == (2, h.pages) and bool(torch.isfinite(scores).all())
    expected = grouped_output(grouped_logits(query, key, scaling), value, mask, size)
    actual = gathered_output(query, key, value, mask, h, count, scaling)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
