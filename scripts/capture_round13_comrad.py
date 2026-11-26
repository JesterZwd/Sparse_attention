"""Capture previously unsampled whole steps for equally bounded A/B/M replay."""
import argparse
import gzip
import json
from pathlib import Path

import torch

from gqa_probe.round11_trace import CommonAttention
from gqa_probe.round11_workpoints import A, B, M
from gqa_probe.round7_attention import Attention
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts.round7_run import DENSE
from scripts.run_round10 import ROOT as R10
from scripts.run_round10 import answer_spans, init_model, prefill, prompt_bundle, read_bundle, read_masks, seal
from scripts.run_round11 import gold_spans, source_hashes
from scripts.run_round13 import ROOT as R13
from scripts.run_ruler_round3 import load_rows

R12=Path("results/round12_frontend")
INDICES={"32k": (12,37,61,85,110), "64k": (5,17,30,43,55)}
PHASES=("first","first_value","last","last_value","late")


@torch.inference_mode()
def capture(root,length):
    data=R12/"data"/length
    manifest,_,population=load_rows(data,"test")
    rows=[population[i] for i in INDICES[length]]
    policies=(A,B,M)
    model,tokenizer,eos,hashes=init_model(manifest)
    masks=read_masks(R10,"cuda")
    output=root/length/"common"
    output.mkdir(parents=True,exist_ok=True)
    seal(output/"config.json",dict(indices=list(INDICES[length]),sample_ids=[r["sample_id"] for r in rows],
            phases=list(PHASES),models=hashes,mask_sha256=_sha(R10/"channels.json"),
            dataset_sha256=_sha(data/"manifest.json"),policies=[p.name for p in policies],
            sources={**source_hashes(),__file__:_sha(Path(__file__))}))
    frames=[]; plans=[]; receipts={}
    for i,(row,name) in enumerate(zip(rows,PHASES,strict=True)):
        bundle=output/"bundles"/f"{i:04d}.json.gz"
        if not bundle.is_file():
            state=prefill(model,row)
            dense=generate_from_prefill(model,state,row["max_new_tokens"],eos,Attention(DENSE,None,row["gold_spans"]))
            starts=answer_spans(tokenizer,dense.token_ids,row)
            last=len(dense.token_ids)-2
            if last<0:
                raise ValueError("capture needs at least one decoded token")
            beginnings=sorted(starts)
            points=(0,beginnings[0] if beginnings else min(7,last),last,
                    beginnings[-1] if beginnings else min(15,last),last*3//4)
            step=points[i]
            cap=CommonAttention(DENSE,policies,masks,gold_spans(row),starts,row["sample_id"],length,i,step)
            replay=generate_from_prefill(model,state,row["max_new_tokens"],eos,cap)
            if replay.token_ids!=dense.token_ids:
                raise ValueError("physical capture changed Dense continuation")
            full=[x for x in cap.traces if x["whole_step"]]
            if len(full)!=len(policies)*model.config.num_hidden_layers*model.config.num_key_value_heads:
                raise ValueError("incomplete full-step capture")
            prompt_bundle(bundle,dict(sample_id=row["sample_id"],phase=name,step=step,
                                      answer_start=step in starts,rows=full))
            del state,cap
        saved=read_bundle(bundle)
        frames.extend(saved["rows"])
        plans.append({k:saved[k] for k in ("sample_id","phase","step","answer_start")})
        receipts[bundle.name]=_sha(bundle)
        print(length,name,i+1,len(rows),flush=True)
    archive=output/"states.jsonl.gz"
    tmp=archive.with_suffix(".tmp")
    with gzip.open(tmp,"wt",compresslevel=4) as stream:
        for row in frames:
            stream.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    tmp.replace(archive)
    _write_json(output/"complete.json",dict(rows=len(frames),states_sha256=_sha(archive),
                                           endpoints=plans,bundles=receipts))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("length",choices=tuple(INDICES))
    p.add_argument("--root",type=Path,default=R13)
    a=p.parse_args()
    capture(a.root,a.length)
