"""Collect a shared A1 compression and B gate diagnostic cohort."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from gqa_probe.fixed_k_v import byte_quota, ranked_pages
from gqa_probe.oracle_attention import grouped_logits, grouped_output
from gqa_probe.page_read_budget import page_history
from gqa_probe.pagepack import segment_statistics, summary_pages
from gqa_probe.pagepack_probe import PACKINGS, probe
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.range_gate import screen_four_tiers
from gqa_probe.round6_attention import SAMPLE_LAYERS, SAMPLE_STEPS
from gqa_probe.round7_attention import Attention
from gqa_probe.round7_single import prepare_single
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.vskip_summary import weights
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DATA, DENSE, MODEL, answer_queries, make_answer
from scripts.run_ruler_round3 import load_rows


ROOT=Path("results/round8_pagepack")
BASE="da8f8985bf6ce164a1a6afe891518985f8cc54f8"
A1_ABS_TOLERANCE=.05  # absolute P99 inflation relative to uncompressed fixed K
A1_VAL_REL_PACK_MAX=.35
A1_KV_READ_SAVING=.10


def history_pages(cache,h):
    padding=h.pages*h.page_size-h.tokens
    return F.pad(cache[0],(0,0,0,padding)).unflatten(1,(h.pages,h.page_size))


class Capture(Attention):
    def __init__(self,row,starts,output_a,output_b,trace):
        super().__init__(DENSE,None,row["gold_spans"])
        self.ident=row["sample_id"]
        self.starts=starts
        self.output_a=output_a
        self.output_b=output_b
        self.trace=trace

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step<0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        groups,t,dim=key.shape[1:]
        scale=kwargs.get("scaling") or dim**-.5
        dense=grouped_output(grouped_logits(query,key,scale),value)
        layer=module.layer_idx
        if layer not in SAMPLE_LAYERS or (self.step not in SAMPLE_STEPS and self.step not in self.starts):
            return self.output(query,dense)
        h=page_history(t,32)
        q=query.reshape(groups,-1,dim).float()
        hp=history_pages(key,h)
        lengths=(t-torch.arange(h.pages,device=key.device)*32).clamp(0,32)
        shared=segment_statistics(hp,lengths)
        mr=summary_pages(hp,lengths,stats=shared,method="mr25")
        q6=summary_pages(hp,lengths,stats=shared,method="q6")
        del hp,shared
        rank=ranked_pages(weights(q,mr,"mr25",scale),h)
        ids=rank[:,:byte_quota(t,32,"mr25",5)].sort(-1).values
        gold=set(self.starts.get(self.step,self._gold_page_tokens))
        result,raw=probe(q,key,value,h,ids,scale,dense,gold)
        priority=ranked_pages(weights(q,q6,"q6",scale),h)
        prepared=prepare_single(q,key,priority,t,scale,"q6",1000,68)
        side=screen_four_tiers(prepared,priority,value,dense,evidence=gold)
        row=dict(sample_id=self.ident,step=self.step,layer=layer,history_tokens=t,
                 answer_start=self.step in self.starts)
        for stream,rows in ((self.output_a,result),(self.output_b,side)):
            for r in rows:
                stream.write(json.dumps({**row,**r},allow_nan=False,separators=(",",":"))+"\n")
        np.savez(self.trace/f"{layer:02d}_{self.step:03d}.npz",
                 a_page_ids=ids.cpu().numpy(),a_logits=raw.cpu().numpy(),
                 b_page_ids=prepared[0].ids.cpu().numpy(),b_logits=prepared[0].logits.cpu().numpy(),
                 answer_start=row["answer_start"])
        return self.output(query,dense)


def sources():
    paths=(Path(__file__),Path("gqa_probe/pagepack.py"),Path("gqa_probe/pagepack_probe.py"),
           Path("gqa_probe/range_gate.py"),Path("gqa_probe/lpddr5_banked.py"),Path("sim/lpddr5_banked.cpp"))
    return {str(x):_sha(x) for x in paths}


@torch.inference_mode()
def capture(root,data,count):
    cohort=data/"calibration"
    manifest,context,rows=load_rows(cohort,"calibration")
    if context!=32768 or len(rows)!=16 or count not in range(1,9):
        raise ValueError("expected 32K Round 7 calibration cohort")
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    config=dict(base=BASE,model_sha256=hashes,manifest_sha256=_sha(cohort/"manifest.json"),
                ids=[r["sample_id"] for r in rows[:count]],sources=sources(),
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                packing=list(map(asdict,PACKINGS)),fixed_pages="5% uncompressed MR25 full-V quota",
                error_margin=A1_ABS_TOLERANCE,pack_only_rel_p99=A1_VAL_REL_PACK_MAX,
                kv_read_saving=A1_KV_READ_SAVING)
    cfg=root/"probe_config.json"
    if cfg.exists() and config!=json.loads(cfg.read_text()):
        raise ValueError("probe configuration or code differs")
    _write_json(cfg,config)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokens=model.generation_config.eos_token_id
    eos=set(tokens if isinstance(tokens,list) else [tokens])
    proofs={}
    for idx,row in enumerate(rows[:count]):
        out=root/"probe"
        hold=[out/m/f"{idx:04d}.jsonl.gz" for m in ("a1","b")]
        if not all(x.exists() for x in hold):
            for x in hold:
                x.parent.mkdir(parents=True,exist_ok=True)
            trace=root/"traces"/f"{idx:04d}"
            trace.mkdir(parents=True,exist_ok=True)
            prefix=prefill_for_quantization(model,row["input_ids"],"affine32")
            dense=generate_from_prefill(model,prefix,row["max_new_tokens"],eos,
                                        Attention(DENSE,None,row["gold_spans"]))
            starts=answer_queries(tokenizer,dense.token_ids,row)
            tmp=[x.with_suffix(".tmp") for x in hold]
            with gzip.open(tmp[0],"wt",compresslevel=3) as a,gzip.open(tmp[1],"wt",compresslevel=3) as b:
                attn=Capture(row,starts,a,b,trace)
                other=generate_from_prefill(model,prefix,row["max_new_tokens"],eos,attn)
            if dense.token_ids!=other.token_ids:
                raise ValueError("probe modified the numerical Dense trajectory")
            _write_json(trace/"verification.json",dict(dense=make_answer(DENSE,dense,tokenizer,row,eos),
                        answer_starts=sorted(starts),raw_qk_files=len(list(trace.glob("*.npz")))))
            for x,y in zip(tmp,hold):
                x.replace(y)
        proofs.update({str(x.relative_to(root)):_sha(x) for x in hold})
        print(f"probe {idx+1}/{count}",flush=True)
    _write_json(root/"probe_complete.json",dict(config_sha256=_sha(cfg),files=proofs))


def collect(root,kind):
    receipt=json.loads((root/"probe_complete.json").read_text())
    if _sha(root/"probe_config.json")!=receipt["config_sha256"]:
        raise ValueError("probe changed after capture")
    for name,sha in receipt["files"].items():
        path=root/name
        if path.parent.name!=kind:
            continue
        if _sha(path)!=sha:
            raise ValueError("probe trace checksum mismatch")
        with gzip.open(path,"rt") as f:
            yield from (json.loads(line) for line in f)


def stat(points):
    by_prompt=defaultdict(list)
    for r in points:
        by_prompt[r["sample_id"]].append(r)
    fields=("head_dense_abs","head_dense_rel","head_pack_abs","head_pack_rel",
            "worst_segment_logit_span","worst_evidence_logit_span")
    tails={f"{k}_p{int(q*100)}":mean(float(np.quantile([x for r in s for x in r[k]],q))
                for s in by_prompt.values()) for k in fields for q in (.95,.99)}
    costs={k:mean(r[k] for r in points) for k in ("k_records","v_records","k_burst_bytes",
                                                 "v_burst_bytes","qk_mac","pv_mac")}
    return dict(states=len(points),prompts=len(by_prompt),**tails,**costs)


def report(root):
    cfg=json.loads((root/"probe_config.json").read_text())
    fit=set(cfg["ids"][:len(cfg["ids"])//2])
    groups=defaultdict(list)
    for r in collect(root,"a1"):
        for subset in ("all","fit" if r["sample_id"] in fit else "verify"):
            groups[r["policy"],r["precision"],subset,False].append(r)
            if r["answer_start"]:
                groups[r["policy"],r["precision"],subset,True].append(r)
    rows=[dict(policy=k[0],precision=k[1],subset=k[2],answer_start=k[3],**stat(v))
          for k,v in sorted(groups.items())]
    write_csv(root/"a1_fixed_pages.csv",rows)
    table={(r["policy"],r["precision"],r["subset"],r["answer_start"]):r for r in rows}
    checks=[]
    eligible=[]
    for p in PACKINGS[1:]:
        sub=[]
        for phase in ("fit","verify"):
            base=table["raw","kv4",phase,False]
            now=table[p.name,"kv4",phase,False]
            first=table.get((p.name,"kv4",phase,True),now)
            raw_first=table.get(("raw","kv4",phase,True),base)
            d=now["head_dense_abs_p99"]/base["head_dense_abs_p99"]-1
            e=first["head_dense_abs_p99"]/raw_first["head_dense_abs_p99"]-1
            b=1-(now["k_burst_bytes"]+now["v_burst_bytes"])/(base["k_burst_bytes"]+base["v_burst_bytes"])
            loss=now["head_pack_rel_p99"]
            sub.append(dict(phase=phase,abs_p99_delta=d,answer_start_abs_p99_delta=e,
                            kv_burst_reduction=b,pack_error_p99=loss,passes=(
                            max(d,e)<=A1_ABS_TOLERANCE and loss<=A1_VAL_REL_PACK_MAX and b>=A1_KV_READ_SAVING)))
        passed=all(r["passes"] for r in sub)
        checks.append(dict(policy=p.name,passes=passed,subsets=sub))
        if passed:
            eligible.append((sub[0]["kv_burst_reduction"],p.name))
    chosen=max(eligible)[1] if eligible else None
    _write_json(root/"a1_decision.json",dict(candidate=chosen,checks=checks,
          rule="both fixed-page cohorts must pass; fit-side KV savings break ties",
          bytes="32-byte aligned KV planes; per-segment metadata and directory charged in physical A2"))
    print(json.dumps(dict(candidate=chosen, rows=len(rows)),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("capture","report"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--count",type=int,default=8)
    a=parser.parse_args()
    if a.phase=="capture":
        capture(a.root,a.data,a.count)
    else:
        report(a.root)


if __name__=="__main__":
    main()
