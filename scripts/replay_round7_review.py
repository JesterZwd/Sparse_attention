"""Capture the Round 7 common states, then replay actual read dependencies."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import transformers

from gqa_probe.fixed_k_v import gather_tokens, masked_output, ranked_pages
from gqa_probe.lpddr5_banked import read_banked
from gqa_probe.oracle_attention import grouped_logits, grouped_output
from gqa_probe.page_read_budget import page_history
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.refill_replay import (
    build_phases, dynamic_histogram, index_addresses, single_routes, workset,
)
from gqa_probe.round6_attention import SAMPLE_LAYERS, SAMPLE_STEPS
from gqa_probe.round7_attention import Attention
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.vskip_summary import StaticOnline, StaticPrefix, weights
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DATA, DENSE, MODEL, answer_queries, make_answer
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round7_hardware_review")
ROUND7 = Path("results/round7_budget_gates")
BASE = "a8db79b7e81526f0fa2281b7b85fdcb20e36ea7f"
PHASES = ["summary", "index", "first_k", "refill_index", "refill_k", "v"]
INDEX_MODES = ("direct","table")


def items(path):
    with gzip.open(path,"rt") as f:
        yield from (json.loads(line) for line in f)


def write_line(out,row):
    out.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")


def execute_route(stream, row, route, name, mode, h, ranks, value, dense, evidence):
    state = route.state
    groups,heads = state.logits.shape[:2]
    y = masked_output(state.logits,gather_tokens(value,state.positions),state.mask)
    err = (y-dense).norm(dim=-1).cpu().tolist()
    rel = ((y-dense).norm(dim=-1)/(dense.norm(dim=-1)+1e-8)).cpu().tolist()
    packed = torch.stack((route.added,route.fallback.long(),route.prefix,
                          route.bytes,route.v_bytes,route.index,
                          state.mask.sum((1,2,3))),dim=-1).cpu().tolist()
    masks = state.mask.any(1).cpu().numpy()
    ids = state.ids.cpu().tolist()
    validity = state.valid.any(-1).cpu().tolist()
    priority = ranks[:,:route.initial].cpu().tolist()
    losses = torch.stack((route.all_drop,route.prefix_drop),dim=1).cpu().tolist()
    for g,(extra,back,prefix,spent,v_bytes,index_bytes,pvtokens) in enumerate(packed):
        first = priority[g]
        later = ids[g][route.initial:route.initial+extra]
        v = {p: mask for p,mask,valid in zip(ids[g],masks[g],validity[g]) if valid and mask.any()}
        k_ev = len(evidence.intersection(first+later))
        w = workset(h,len(first),extra,prefix,heads,route.gate,route.fallback_mode if back else "none",
                    index_mode=mode,first_table=32*len(index_addresses(first)),final_table=index_bytes)
        phases,wait = build_phases(h,first,later,v,mode,
                                   score_ck=w["score_ck"],initial_wait_ck=w["initial_wait_ck"],
                                   extra_wait_ck=w["extra_wait_ck"])
        flattened = sorted((adr,adr+size) for stage in phases for adr,size in stage)
        if any(stop>nxt for (_,stop),(nxt,_) in zip(flattened,flattened[1:])):
            raise ValueError("duplicate physical burst")
        if (sum(b-a for a,b in flattened)!=spent or spent>route.cap or
                sum(n for phase in (phases[1],phases[3]) for _,n in phase)!=index_bytes or
                sum(n for _,n in phases[-1])!=v_bytes):
            raise ValueError("physical transaction ledger mismatch")
        write_line(stream,dict(**row,group=g,history_tokens=h.tokens,mode=mode,policy=name,
                  cap=route.cap,read_bytes=spent,index_bytes=index_bytes,v_bytes=v_bytes,
                  initial_ids=first,added_ids=later,retained_prefix=prefix,fallback=bool(back),
                  dense_attention_sram_bytes=4*heads*h.tokens+4*heads*(128+2),
                  dense_logits_sram_bytes=4*heads*h.tokens,
                  abs_error=err[g],relative_error=rel[g],lost_all_k=losses[g][0],lost_prefix=losses[g][1],
                  k_evidence=k_ev,v_evidence=len(evidence.intersection(v)),evidence=len(evidence),
                  pv_mac=pvtokens*128,v_union_tokens=sum(int(mask.sum()) for mask in v.values()),
                  phase_ranges=phases,stage_wait_ck=wait,**w))


class Capture(Attention):
    def __init__(self,row,starts,stream,*,histogram_fallback=False):
        super().__init__(DENSE,None,row["gold_spans"])
        self.sample_id=row["sample_id"]
        self.starts=starts
        self.stream=stream
        self.histogram_fallback=histogram_fallback

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step<0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        g,t,d=key.shape[1:]
        scale=kwargs.get("scaling") or d**-.5
        dense=grouped_output(grouped_logits(query,key,scale),value)
        layer=module.layer_idx
        if layer not in SAMPLE_LAYERS or (self.step not in SAMPLE_STEPS and self.step not in self.starts):
            return self.output(query,dense)
        h=page_history(t,32)
        q=query.reshape(g,-1,d).float()
        prefix=StaticPrefix.build(key[:,:,:-1],32,("q6",))
        records=StaticOnline(prefix,("q6",)).append(key[0,:,-1],t)["q6"]
        ranks=ranked_pages(weights(q,records,"q6",scale),h)
        row=dict(sample_id=self.sample_id,layer=layer,step=self.step,answer_start=self.step in self.starts)
        gold=set(self.starts.get(self.step,self._gold_page_tokens))
        for mode in INDEX_MODES:
            routes=single_routes(q,key,ranks,h,scale,mode,histogram_fallback=self.histogram_fallback)
            routes["dynamic_hist64"]=dynamic_histogram(q,key,ranks,h,scale,mode)
            for name,route in routes.items():
                execute_route(self.stream,row,route,name,mode,h,ranks,value,dense,gold)
        return self.output(query,dense)


@torch.inference_mode()
def capture(root,data,count,histogram_fallback):
    cohort=data/"calibration"
    manifest,context,rows=load_rows(cohort,"calibration")
    if context!=32768 or len(rows)!=16 or count not in range(1,9):
        raise ValueError("expected Round 7 32K cohort and first eight prompt subset")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    root.mkdir(parents=True,exist_ok=True)
    evidence=root/"addresses"
    evidence.mkdir(exist_ok=True)
    files=[Path(__file__),Path("gqa_probe/refill_replay.py"),Path("gqa_probe/lpddr5_banked.py"),
           Path("sim/lpddr5_banked.cpp")]
    config=dict(base=BASE,manifest_sha256=_sha(cohort/"manifest.json"),model_hashes=hashes,
                phases=PHASES,mode=list(INDEX_MODES),ids=[r["sample_id"] for r in rows[:count]],
                page_size=32,group_cap_percent=10,max_k_pages=68,epsilon=.02,tau=8,
                histogram_fallback=histogram_fallback,source_sha256={str(x):_sha(x) for x in files},
                layers=list(SAMPLE_LAYERS),steps=list(SAMPLE_STEPS),gpu=torch.cuda.get_device_name())
    config_path=root/"config.json"
    if config_path.exists() and config!=json.loads(config_path.read_text()):
        raise ValueError("previous captures use different sources, data or config")
    _write_json(config_path,config)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokens=model.generation_config.eos_token_id
    eos=set(tokens if isinstance(tokens,list) else [tokens])
    baseline={r["sample_id"]:r for r in json.loads(
        (ROUND7/"report/screen/predictions.json").read_text()) if r["policy"]==DENSE.name}
    records={}
    for idx,row in enumerate(rows[:count]):
        address=evidence/f"{idx:04d}.jsonl.gz"
        if not address.exists():
            base=prefill_for_quantization(model,row["input_ids"],"affine32")
            attn=Attention(DENSE,None,row["gold_spans"])
            dense=generate_from_prefill(model,base,row["max_new_tokens"],eos,attn)
            now=make_answer(DENSE,dense,tokenizer,row,eos)
            old=baseline[row["sample_id"]]
            if any(now[k]!=old[k] for k in ("official_score","complete_correct","decode_tokens")):
                raise ValueError("common prefix differs from Round 7 numeric reference")
            starts=answer_queries(tokenizer,dense.token_ids,row)
            temp=address.with_suffix(".tmp")
            with gzip.open(temp,"wt",compresslevel=3) as f:
                cap=Capture(row,starts,f,histogram_fallback=histogram_fallback)
                observed=generate_from_prefill(model,base,row["max_new_tokens"],eos,cap)
            if observed.token_ids!=dense.token_ids:
                raise ValueError("capturing the address graph changed the common trajectory")
            temp.replace(address)
        records[address.name]=_sha(address)
        print(f"capture {idx+1}/{count}",flush=True)
    _write_json(root/"capture_receipt.json",dict(config_sha256=_sha(config_path),address_sha256=records))


def replay_file(args):
    source,target=args
    counted=0
    temp=target.with_suffix(".tmp")
    with gzip.open(temp,"wt",compresslevel=4) as out:
        for r in items(source):
            phases=r.pop("phase_ranges")
            waits=r.pop("stage_wait_ck")
            ranges=sorted((address,address+size) for phase in phases for address,size in phase)
            if (any(stop>nxt for (_,stop),(nxt,_) in zip(ranges,ranges[1:])) or
                    sum(stop-start for start,stop in ranges)!=r["read_bytes"] or
                    r["read_bytes"]>r["cap"]):
                raise ValueError("invalid or over-budget physical range")
            banked=read_banked(phases,waits,queue_depth=32)
            if banked["bursts"]*32!=r["read_bytes"]:
                raise ValueError("banked scheduler lost physical reads")
            pv=8*r["v_union_tokens"]
            r.update(chronology_ck=banked,
                     banked_cycle=banked["phase_end_ck"][-1]+pv,
                     banked_first_v_ready=banked["phase_start_ck"][-1],
                     banked_first_v_command=banked["phase_first_command_ck"][-1],
                     pv_cycles=pv)
            # Same addresses and phase dependencies, with a single in-flight
            # burst as the concurrency sensitivity, sampled on two layers.
            if r["layer"] in (7,21) and r["step"] in (0,7,15):
                q1=read_banked(phases,waits,queue_depth=1)
                r["one_slot_cycle"]=q1["phase_end_ck"][-1]+pv
            write_line(out,r)
            counted+=1
    temp.replace(target)
    return counted


def replay(root,workers):
    cfg=json.loads((root/"config.json").read_text())
    receipt=json.loads((root/"capture_receipt.json").read_text())
    if _sha(root/"config.json")!=receipt["config_sha256"]:
        raise ValueError("capture config mismatch")
    path=root/"timing"
    path.mkdir(exist_ok=True)
    pairs=[]
    for name,expected in receipt["address_sha256"].items():
        src=root/"addresses"/name
        if _sha(src)!=expected:
            raise ValueError("address trace checksum mismatch")
        pairs.append((src,path/name))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts=list(pool.map(replay_file,pairs))
    _write_json(root/"timing_receipt.json",dict(routes=sum(counts),
               timing_sha256={x.name:_sha(x) for _,x in pairs},phase_count=len(cfg["phases"]),
               replay_source_sha256={p:_sha(Path(p)) for p in (str(Path(__file__)),
                   "gqa_probe/lpddr5_banked.py","sim/lpddr5_banked.cpp")}))


def summarize(root):
    cap=json.loads((root/"config.json").read_text())
    receipt=json.loads((root/"timing_receipt.json").read_text())
    states=set()
    table=defaultdict(list)
    for name,checksum in receipt["timing_sha256"].items():
        path=root/"timing"/name
        if _sha(path)!=checksum:
            raise ValueError("timing checksum mismatch")
        for r in items(path):
            key=(r["sample_id"],r["step"],r["layer"],r["group"])
            states.add(key)
            table[r["mode"],r["policy"]].append(r)
    old={(r["sample_id"],r["step"],r["layer"],r["group"]) for r in items(
        ROUND7/"report/screen/gate_diag.jsonl.gz")
        if r["policy"]=="single_hist64" and r["sample_id"] in cap["ids"]}
    if states!=old or any(len(points)!=len(states) for points in table.values()):
        raise ValueError("replay does not pair the same common states as Round 7")
    summary=[]
    for (mode,name),rs in sorted(table.items()):
        per=defaultdict(list)
        for r in rs:
            per[r["sample_id"]].append(r)
        numeric=("read_bytes","index_bytes","v_bytes","fallback","first_gate_ck","fallback_ck",
                 "banked_cycle","banked_first_v_ready","banked_first_v_command","attention_sram_bytes",
                 "qk_mac","pv_mac","scorer_mac")
        means={f"mean_{k}":mean(r[k] for r in rs) for k in numeric}
        peaks={f"peak_{k}":max(r[k] for r in rs) for k in ("control_sram_bytes","logits_sram_bytes",
               "attention_sram_bytes","dense_logits_sram_bytes","dense_attention_sram_bytes")}
        tails={}
        for k in ("abs_error","relative_error","lost_all_k","lost_prefix"):
            for pct in (.95,.99):
                tails[f"{k}_p{int(100*pct)}"]=mean(float(np.quantile([x for r in p for x in r[k]],pct))
                                                  for p in per.values())
        sample=[r for r in rs if "one_slot_cycle" in r]
        recall=lambda key,points: sum(r[key] for r in points)/max(1,sum(r["evidence"] for r in points))
        summary.append(dict(mode=mode,policy=name,states=len(rs),prompts=len(per),
                            k_recall=recall("k_evidence",rs),v_recall=recall("v_evidence",rs),
                            answer_start_v_recall=recall("v_evidence",[r for r in rs if r["answer_start"]]),
                            max_queue=max(r["chronology_ck"]["max_outstanding"] for r in rs),
                            one_slot_states=len(sample),
                            one_slot_cycle=mean(r["one_slot_cycle"] for r in sample),
                            banked_same_sample_cycle=mean(r["banked_cycle"] for r in sample),
                            **means,**peaks,**tails))
    write_csv(root/"summary.csv",summary)
    _write_json(root/"verification.json",dict(common_states=len(states),read_budgets_pass=True,
                cohort_unchanged=True,source_config_sha256=_sha(root/"config.json"),routes=receipt["routes"]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("capture","replay","summary"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--histogram-fallback",action="store_true")
    a=p.parse_args()
    if a.stage=="capture":
        capture(a.root,a.data,a.count,a.histogram_fallback)
    elif a.stage=="replay":
        replay(a.root,a.workers)
    else:
        summarize(a.root)


if __name__=="__main__":
    main()
