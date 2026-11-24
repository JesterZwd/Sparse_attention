import pytest
import torch

from gqa_probe.compact_summary import CompactOnline, CompactPrefix, records_as_bytes, score_inputs, scores
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.round11_workpoints import M, allocation
from gqa_probe.round12_narrow import CANDIDATES, same_k


@pytest.mark.parametrize("tokens", [32641, 32768, 65393])
def test_narrow_m_bills_its_own_summary_with_inherited_k(tokens):
    h = page_history(tokens, 32)
    weights = torch.arange(h.pages, dtype=torch.float).repeat(2, 1)
    base = allocation(M, ranked_pages(weights, h), h)
    for p in CANDIDATES:
        other = same_k(p, ranked_pages(-weights, h), h)
        assert other.cap == base.cap and other.summary_bytes < base.summary_bytes
        torch.testing.assert_close(other.k_counts, base.k_counts)
        assert not torch.equal(other.ids, base.ids)


@pytest.mark.parametrize("p,width,stride", zip(CANDIDATES, (80, 64), (96, 80)))
def test_compact_record_and_full_query_scale(p, width, stride):
    rng = torch.Generator().manual_seed(8)
    keys = torch.randn(1, 2, 35, 128, generator=rng)
    q = torch.randn(2, 4, 128, generator=rng)
    q[..., 127] = 50
    ids = torch.arange(width).expand(2, -1)
    prefix = CompactPrefix.build(keys[:, :, :-1], 32, (p.method,), {p.method: ids})
    record = CompactOnline(prefix).append(keys[0, :, -1], 35)[p.method]
    packed = records_as_bytes(record, p.method)
    assert packed.shape == (2, 2, stride) and (packed[:, -1, width+4] == 3).all()
    data = score_inputs(q, record, p.method, .125, channels=ids)
    expected = (q[..., :width] / (50/127)).round().to(torch.int8)
    torch.testing.assert_close(data.query, expected)
    assert torch.isfinite(scores(q, record, p.method, .125, channels=ids)).all()
