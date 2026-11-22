import torch

from gqa_probe.page_read_budget import page_history
from gqa_probe.round7_sides import SideScreen, compensation_outputs, summary_bursts, value_start_pages
from gqa_probe.vskip_summary import StaticPrefix, weights


def test_summary_request_rounding_deduplicates_shared_bursts():
    assert summary_bursts([1, 0, 1, 3], 208) == [(0, 416), (608, 224)]


def test_exact_mass_compensation_with_tail_keeps_hot_contribution():
    h = page_history(5, 2)
    logits = torch.log(torch.tensor([[[.20, .10, .15, .40, .15]]]))
    value = torch.tensor([[[[3.], [3.], [1.], [5.], [9.]]]])
    hot = torch.tensor([[True, False, False]])
    dense, hot_only, corrected = compensation_outputs(logits, value, hot, h)
    torch.testing.assert_close(hot_only, torch.tensor([[[3.]]]))
    torch.testing.assert_close(corrected, torch.tensor([[[.3 * 3 + .55 * 3 + .15 * 9]]]))
    assert abs((dense - corrected).item()) < abs((dense - hot_only).item())


def test_value_start_is_the_query_before_the_first_answer_token():
    class Tokenizer:
        def decode(self, tokens, *, skip_special_tokens):
            return "".join(tokens)

    evidence = [{"value": "abc", "association_span": [31, 33]},
                {"value": "zoo", "association_span": [200, 210]}]
    assert value_start_pages(Tokenizer(), ["[", " a", "bc", "]"], evidence) == {0: {0, 1}}


def test_reuse_reference_moves_only_on_full_refresh_and_reranks_same_subset():
    h = page_history(32 * 12, 32)
    g = torch.Generator().manual_seed(3)
    key = torch.randn((1, 1, h.tokens, 128), generator=g)
    value = torch.randn(key.shape, generator=g)
    record = StaticPrefix.build(key, 32, ("q6",)).records["q6"]
    q1 = torch.randn((1, 1, 128), generator=g)
    alt = torch.randn((1, 1, 128), generator=g)
    alt = alt - q1 * (alt * q1).sum() / q1.square().sum()
    alt = alt / alt.norm() * q1.norm()
    screen = SideScreen(layers=(0,), page_count=len(h.protected) + 1,
                        top_count=4, interval=2, cosine=.95, cold_steps=())
    for step, q in enumerate((q1, q1 + .25 * alt, q1 + .41 * alt)):
        logits = q @ key[0].transpose(-1, -2)
        scores = weights(q, record, "q6", 128**-.5)
        screen.observe(step, 0, h, q, key, value, logits, None, scores, record, 128**-.5, (4,),
                       new_value_head=step == 1)
    rows = {(row["step"], row["strategy"]): row for row in screen.finalize()["rows"]}
    assert rows[1, "cosine"]["reference_step"] == 0
    assert rows[1, "rerank128"]["summary_records"] == 4 + len(h.protected)
    assert rows[1, "rerank128"]["new_value_head"]
    assert rows[1, "cosine"]["summary_records"] == 0
    assert rows[2, "cosine"]["reference_step"] == 2
    assert rows[2, "rerank128"]["full_scan"]
    assert rows[2, "interval"]["full_scan"]
    for step in range(3):
        assert all(set(h.protected) <= set(rows[step, route]["selected_pages"])
                   for route in ("full", "interval", "cosine", "rerank128"))
    marked = screen.finalize(value_starts={2: {4, 5}})
    last = next(row for row in marked["rows"] if row["step"] == 2 and row["strategy"] == "full")
    assert last["new_value_head"] and last["gold_pages"] == 1 and last["value_start_pages"] == 2


def test_long_history_cold_oracle_preserves_mass_at_large_dynamic_range():
    h = page_history(32737, 32)
    # Constant values give a check across a nontrivial hot/cold split
    # independent of how concentrated the original distribution is.
    logits = torch.full((1, 2, h.tokens), -13.5)
    logits[0, 0, 100] = -.1
    logits[0, 1, -1] = 2.33
    values = torch.full((1, 1, h.tokens, 1), 2.)
    hot = torch.zeros((1, h.pages), dtype=torch.bool)
    hot[:, list(h.protected)] = True
    dense, hot_only, corrected = compensation_outputs(logits, values, hot, h)
    torch.testing.assert_close(dense, torch.full_like(dense, 2.))
    torch.testing.assert_close(corrected, dense)
    torch.testing.assert_close(hot_only, dense)
