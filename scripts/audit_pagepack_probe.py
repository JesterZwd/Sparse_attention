"""Recheck traffic after V pooling on the stored common raw QK logits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from gqa_probe.page_read_budget import page_history
from gqa_probe.round7_gates import gate_mask
from gqa_probe.ruler_experiment import _write_json
from gqa_probe.shared_refill import State
from scripts.probe_pagepack import ROOT, collect


def charged(mask):
    union=mask.any(1)
    codes=64*union.sum()
    scale=32*union.unflatten(-1,(-1,4)).any(-1).sum()
    zp=32*union.unflatten(-1,(-1,8)).any(-1).sum()
    return int(codes+scale+zp)


def gate_traffic(root):
    states={}
    mixed_by_policy=defaultdict(dict)
    for row in collect(root,"a1"):
        key=(row["sample_id"],row["step"],row["layer"])
        states[key]=row["history_tokens"]
        if row["precision"]=="kv4":
            mixed_by_policy[row["policy"]][(*key,row["group"])]=row
    cfg=json.loads((root/"probe_config.json").read_text())
    aggregate={i:defaultdict(int) for i in (2,4)}
    for idx,identifier in enumerate(cfg["ids"]):
        folder=root/"traces"/f"{idx:04d}"
        for path in sorted(folder.glob("*.npz")):
            layer,step=map(int,path.stem.split("_"))
            history=page_history(states[identifier,step,layer],32)
            with np.load(path) as data:
                logits=torch.from_numpy(data["a_logits"])
                ids=torch.from_numpy(data["a_page_ids"])
            valid=torch.isfinite(logits[:,0])
            pinned=torch.isin(ids,ids.new_tensor(history.protected))
            raw=gate_mask(State(ids,ids,logits,valid,pinned,valid[:,None].expand_as(logits)),"hist64")
            oldv=charged(raw)
            oldpv=int(raw.sum()*128)
            slots=torch.arange(32)
            for n,a in aggregate.items():
                mapping=torch.where(pinned[...,None],slots,slots//n)
                counts=torch.zeros_like(raw,dtype=torch.int32).scatter_add_(
                    -1,mapping[:,None].expand_as(raw),raw.int())
                keep=counts>0
                shared=raw.any(1)
                union=torch.zeros_like(shared,dtype=torch.int32).scatter_add_(
                    -1,mapping.expand_as(shared),shared.int())
                remote=(~pinned)[...,None]
                a["groups"]+=ids.shape[0]
                a["raw_v_bytes"]+=oldv
                a["pooled_v_bytes"]+=charged(keep)
                a["raw_pv_mac"]+=oldpv
                a["pooled_pv_mac"]+=int(keep.sum()*128)
                a["pool_reduce_adds"]+=int(raw.sum()-keep.sum())
                a["remote_kept_representatives"]+=int(((union>0)&remote).sum())
                a["remote_single_hot_representatives"]+=int(((union==1)&remote).sum())
    totals={}
    for n,a in aggregate.items():
        totals[f"v{n}"]=dict(a,v_physical_reduction=1-a["pooled_v_bytes"]/a["raw_v_bytes"],
                pv_mac_reduction=1-a["pooled_pv_mac"]/a["raw_pv_mac"],
                singleton_hot_fraction=a["remote_single_hot_representatives"]/a["remote_kept_representatives"])
    raw=mixed_by_policy["raw"]
    metadata={}
    for p in sorted(x for x in mixed_by_policy if x.startswith("mixed")):
        rows=mixed_by_policy[p]
        physical=lambda r:r["k_burst_bytes"]+r["v_burst_bytes"]
        before=sum(physical(r) for r in raw.values())
        after=sum(physical(r) for r in rows.values())
        directory=sum(((16*((r["history_tokens"]+31)//32)+31)//32)*32 for r in rows.values())
        metadata[p]=dict(states=len(rows),mean_kv_saved=(before-after)/len(rows),
                         mean_directory_bytes=directory/len(rows),
                         mean_upper_bound_saving=(before-after-directory)/len(rows))
    _write_json(root/"a1_layout_audit.json",dict(v_only_after_hist64=totals,
           mixed_after_directory_before_tags=metadata,
           scope="same K and exact raw K logits; post-gate traffic only, no candidate PV quality recomputation"))
    print(json.dumps(totals))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    gate_traffic(p.parse_args().root)
