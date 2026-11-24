import pytest
import torch

from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.round11_workpoints import A, B, C, M, allocation, verdict


@pytest.mark.parametrize("tokens", [32640, 32641, 32671, 32672, 32768, 65536])
def test_limit_k_preserves_reference_work_with_own_summary_budget(tokens):
    history = page_history(tokens, 32)
    order = torch.arange(history.pages, dtype=torch.float).expand(2, -1)
    a = allocation(A, ranked_pages(order, history), history)
    b = allocation(B, ranked_pages(-order, history), history)
    c = allocation(C, ranked_pages(-order, history), history)
    middle = allocation(M, ranked_pages(-order, history), history)
    assert a.ids.shape == b.ids.shape
    assert not torch.equal(a.ids, b.ids)
    torch.testing.assert_close(a.k_counts, b.k_counts)
    assert a.cap == b.cap == c.cap
    assert b.summary_bytes < a.summary_bytes
    assert torch.all(b.safe_count == b.count)
    assert c.count[0] >= a.count[0]
    assert middle.count[0] == min(a.count[0] + 4, c.count[0])


def cost(read=100, qk=100, arithmetic=230, over=0):
    return dict(read_per_decode=read, qk_per_decode=qk, arithmetic_per_decode=arithmetic, overrun_groups=over)


def test_gates_cover_same_k_noninferiority_and_quality_roles():
    assert verdict(B, cost(), cost(95, 101, 229), -.003, -.01)["passes"]
    assert not verdict(B, cost(), cost(95, 101, 230), -.003, -.01)["passes"]
    assert not verdict(B, cost(), cost(95, 101, 229, 1), -.003, -.01)["passes"]
    assert not verdict(B, cost(), cost(95, 101, 229), -.003, -.0101)["passes"]
    assert not verdict(B, cost(), cost(95, 101.1, 229), -.003, -.01)["passes"]
    assert verdict(C, cost(), cost(101, 136, 250), .01, .001)["passes"]
    assert not verdict(C, cost(), cost(101.01, 136, 250), .01, .001)["passes"]
    assert not verdict(C, cost(), cost(101, 136, 250), .01, 0.)["passes"]
