import torch

from gqa_probe.cold_page_compensation import hybrid_output, reserve_hot


def test_mean_mass_recovers_uniform_token_attention():
    hot_logits = torch.zeros((1, 2, 2))
    hot_values = torch.tensor([[[1., 2.], [2., 4.]]])
    cold_log_mass = torch.tensor([[[2.], [2.]]]).log()
    cold_values = torch.tensor([[[3., 6.]]])
    y = hybrid_output(hot_logits, hot_values, cold_log_mass, cold_values)
    torch.testing.assert_close(y, torch.tensor([[[2.25, 4.5], [2.25, 4.5]]]))


def test_extra_mean_planes_reduce_hot_count_and_fit_cap():
    from gqa_probe.fixed_k_v import byte_quota

    h, cap, count, traffic = reserve_hot(32737)
    assert count < byte_quota(h.tokens, 32, "q6", 10)
    assert 32*traffic["mean_bursts"] >= 144*h.pages
    assert traffic["total"] <= cap
