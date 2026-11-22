from types import SimpleNamespace

import pytest
import torch

from gqa_probe.round7_attention import Attention
from gqa_probe.vskip_summary import StaticPrefix
from scripts.round7_run import CAPPED, COMPACT, FIXED, HIST, SINGLE, Common


def stage():
    t = 32737
    k = torch.zeros(1, 1, t, 128)
    v = torch.randn_like(k)
    q = torch.zeros(1, 4, 1, 128)
    prefix = (StaticPrefix.build(k[:, :, :-1], 32, ("q4", "q6", "mr25")),)*8
    return (SimpleNamespace(layer_idx=7), q, k, v, None), prefix


@pytest.mark.parametrize("policy", [CAPPED, SINGLE, HIST, COMPACT[1], COMPACT[-1]])
def test_decode_attention_accounting_uses_same_addresses(policy):
    args, prefix = stage()
    attn = Attention(policy,prefix,[(2400,2401)],audit=True)
    attn.step = 0
    y, _ = attn.attention(*args)
    assert torch.isfinite(y).all()
    stats = attn.stats()
    assert stats["over_budget_group_steps"] == 0
    assert stats["read"] == attn.timing[0]["reads_bytes"]
    assert stats["qk_mac"] == attn.timing[0]["qk_mac"]
    assert stats["pv_mac"] == attn.timing[0]["pv_mac"]
    assert stats["scorer_mac"] == attn.timing[0]["scorer_mac"]
    assert attn.group_stats[0][3][0,0] == stats["read"]


def test_common_state_exports_groups_with_same_initial_k():
    args, prefix = stage()
    row = {"gold_spans":[(2400,2401)]}
    attn = Common(row,prefix,{0:{75}},FIXED,sides=True)
    attn.step = 0
    y, _ = attn.attention(*args)
    assert torch.isfinite(y).all()
    assert {x["k_pages"] for x in attn.gate_rows} == {68}
    assert all(x["answer_start"] for x in attn.gate_rows)
    assert len(attn.summary_rows) == 3
    assert attn.side.finalize()["rows"]
