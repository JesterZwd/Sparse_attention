"""Check one real-geometry group step in every Matrix route without GPU."""
from runpy import run_path
from types import SimpleNamespace

import torch
from scripts.run_round10 import prefix_layers
from scripts.run_round11 import WorkpointAttention

m=run_path('results/round13_llama_budgets/run_matrix.py')
m['verify_allocation']()
torch.manual_seed(5)
torch.set_num_threads(2)
with torch.inference_mode():
    channels={'mr96_score':{0:torch.arange(96).expand(8,-1)}}
    for n in (32768-127,65536-127):
        q=torch.randn((1,32,1,128))
        k=torch.randn((1,8,n,128),dtype=torch.bfloat16)
        v=torch.randn_like(k)
        state=SimpleNamespace(cache=SimpleNamespace(key_cache=[k[...,:-1,:]]))
        for policy in m['POLICIES']:
            attention=WorkpointAttention(policy,prefix_layers(state,policy,channels),channels,[])
            attention.step=0
            y,_=attention.attention(SimpleNamespace(layer_idx=0),q,k,v,None)
            cur=attention.stats()
            assert tuple(y.shape)==(1,1,32,128) and torch.isfinite(y).all()
            assert cur['kv_group_calls']==8 and cur['over_budget_group_steps']==0
            assert cur['read']<=cur['cap_bytes']<=cur['dense_comparator_bytes']*policy.budget//100
            print(n,policy.budget,policy.name,cur['read'],cur['cap_bytes'],flush=True)
