import math

import pytest
import torch

from gqa_probe.page_read_budget import page_history
from gqa_probe.round7_summary_diag import summary_diagnostics
from gqa_probe.vskip_summary import StaticOnline, StaticPrefix


def make_summary(key):
    methods = ("q4", "mr25")
    prefix = StaticPrefix.build(key[..., :-1, :], 32, methods)
    return StaticOnline(prefix, methods).append(key[0, :, -1], key.shape[-2])


def test_fixed_count_protection_ties_partial_page_and_per_head_metrics():
    key = torch.zeros(1, 2, 645, 8)
    value = torch.ones_like(key)
    query = torch.zeros(2, 2, 8)
    history = page_history(645, 32)
    rows = summary_diagnostics(
        query,
        key,
        value,
        history,
        make_summary(key),
        1,
        count=len(history.protected) + 1,
        evidence=[1, 1, 2, 20],
        answer_start=True,
    )
    assert {(r["method"], r["group"]) for r in rows} == {
        (method, group) for method in ("q4", "mr25", "oracle_a1") for group in range(2)
    }
    for row in rows:
        expected = sorted((*history.protected, 1))
        assert row["pages"] == expected
        assert len(expected) == row["budget_pages"]
        assert row["head_coverage"] == pytest.approx([(len(expected) - 1) * 32 / 645 + 5 / 645] * 2)
        assert row["head_output_absolute_errors"] == pytest.approx([0, 0], abs=2e-5)
        assert row["head_output_relative_errors"] == pytest.approx([0, 0], abs=7e-6)
        assert row["answer_start"] is True
        assert (row["evidence_selected"], row["evidence_total"]) == (2, 3)
        assert row["gold_page_recall"] == pytest.approx(2 / 3)


def test_oracle_uses_full_history_softmax_mass_to_avoid_single_token_false_peak():
    # Quest favors the single score-4 token. Oracle chooses the page whose
    # 32 score-2 tokens have greater total true mass, while retaining Sink/Local.
    key = torch.zeros(1, 1, 645, 8)
    key[0, 0, 32, 0] = 4
    key[0, 0, 64:96, 0] = 2
    value = torch.zeros_like(key)
    value[0, 0, 64:96, 0] = 1
    query = torch.tensor([[[1.0, 0, 0, 0, 0, 0, 0, 0]]])
    history = page_history(645, 32)
    logits = torch.bmm(query, key[0].transpose(1, 2))
    dense = torch.bmm(logits.softmax(-1), value[0])
    rows = {
        row["method"]: row
        for row in summary_diagnostics(
            query, key, value, history, make_summary(key), 1,
            count=len(history.protected) + 1, evidence=[2],
            logits=logits, dense_output=dense,
        )
    }
    q4, oracle = rows["q4"], rows["oracle_a1"]
    assert 1 in q4["pages"] and 2 in oracle["pages"]
    assert oracle["head_coverage"][0] > q4["head_coverage"][0]
    assert oracle["gold_page_recall"] == 1 and q4["gold_page_recall"] == 0
    assert not oracle["answer_start"]
    forced_tokens = (len(history.protected) - 1) * 32 + 5
    target_mass = 32 * math.exp(2)
    dense_denominator = 645 - 33 + math.exp(4) + target_mass
    assert oracle["head_coverage"][0] == pytest.approx(
        (forced_tokens + target_mass) / dense_denominator
    )
    assert q4["head_output_absolute_errors"][0] == pytest.approx(target_mass / dense_denominator)
    assert oracle["head_output_absolute_errors"][0] == pytest.approx(
        target_mass / (forced_tokens + target_mass) - target_mass / dense_denominator
    )


def test_missing_evidence_and_too_small_page_budget():
    key = torch.zeros(1, 1, 645, 8)
    query = torch.zeros(1, 1, 8)
    history = page_history(645, 32)
    packed = make_summary(key)
    count = len(history.protected)
    with pytest.raises(ValueError, match="page count"):
        summary_diagnostics(query, key, key, history, packed, 1, count=count - 1)
    rows = summary_diagnostics(query, key, key, history, packed, 1, count=count)
    assert all(row["evidence_total"] == 0 and row["gold_page_recall"] is None for row in rows)
