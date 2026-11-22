from types import SimpleNamespace

import pytest
import torch

from gqa_probe.pagepack_attention import CompressionAttention, Policy, build_prefix
from gqa_probe.pagepack_probe import PACKINGS


@pytest.mark.parametrize("name,gate", [("raw","hist64"),("v2","hist64"),
                                       ("kv2","full"),("mixed_k_0.5","hist64")])
def test_decode_closure_pins_and_accounts_every_burst(name,gate):
    t=32737
    key=torch.zeros(1,1,t,128)
    value=torch.zeros_like(key)
    pref=SimpleNamespace(cache=SimpleNamespace(key_cache=[key[:,:,:-1]],value_cache=[value[:,:,:-1]]))
    spec=next(x for x in PACKINGS if x.name==name)
    compiled=build_prefix(pref,[spec],("mr25",))
    p=Policy("test","mr25",name,gate,5)
    a=CompressionAttention(p,compiled,[(3000,3001)],audit=True)
    a.step=0
    y,_=a.attention(SimpleNamespace(layer_idx=0),torch.ones(1,4,1,128),key,value,None)
    assert y.shape==(1,1,4,128) and torch.isfinite(y).all()
    s=a.stats()
    assert s["over_budget_group_steps"]==0
    assert s["read"]==a.timing[0]["reads_bytes"]
    assert s["qk_mac"]==a.timing[0]["qk_mac"]
