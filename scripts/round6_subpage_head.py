"""Run frozen Round 6 cohorts and capture numerical and physical ledgers."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re

import torch
import transformers

from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.round6_attention import Attention,Policy
from gqa_probe.round6_audit import Audit
from gqa_probe.ruler_experiment import _sha,_verified_model,_write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.vskip_summary import StaticPrefix
from scripts.run_ruler_round3 import load_rows


ROOT=Path("results/round6_subpage_head")
MODEL=Path("models/Qwen3-4B-Instruct-2507")
BASE="d5f0b612291fd7fa2836afa9d7640caa3440adaf"
FILES=("scripts/round6_subpage_head.py","scripts/report_round6.py","scripts/round6_pipeline.py",
       "sim/round6_lpddr5.cpp","gqa_probe/round6_memory.py","gqa_probe/round6_attention.py",
       "gqa_probe/round6_audit.py","gqa_probe/light_a1_reference.py","gqa_probe/shared_refill.py",
       "gqa_probe/vskip_summary.py","gqa_probe/fixed_k_v.py","gqa_probe/quantization_round2.py",
       "gqa_probe/oracle_attention.py","gqa_probe/paired_statistics.py","gqa_probe/shared_refill_audit.py")


def sources():
    return {p:_sha(Path(p)) for p in FILES}


def baseline():
    return [Policy("dense"),Policy("static"),Policy("dynamic"),
            *[Policy("single",n) for n in (68,74,78)]]


@torch.inference_mode()
def run_one(model,tokenizer,eos,row,policies,phase,do_audit,common,windows):
    state=prefill_for_quantization(model,row["input_ids"],"affine32")
    methods=sorted({p.method for p in policies if p.kind!="dense"})
    prefix=tuple(StaticPrefix.build(k,32,methods) for k in state.cache.key_cache) if methods else None
    answers=[]
    audits=[]
    locality=[]
    common_data=[]
    for p in policies:
        audit=Audit(common if phase=="screen" else None,dense_cost=True) if do_audit else None
        attn=Attention(p,prefix,gold=row["gold_spans"],sample=audit,windows=windows.get(p.map_id))
        generated=generate_from_prefill(model,state,row["max_new_tokens"],eos,attn)
        nd=len(generated.token_ids)-1
        stats=generated.attention_stats
        if stats["decode_layer_calls"]!=nd*model.config.num_hidden_layers:
            raise ValueError("callback coverage was incomplete")
        text=re.sub(r"[\x00-\x1f]","\n",tokenizer.decode(generated.token_ids,skip_special_tokens=True)).lower()
        hits=[r.lower() in text for r in row["references"]]
        ans=dict(policy=p.name,**asdict(p),official_score=sum(hits)/len(hits),complete_correct=all(hits),
                 reference_hits=hits,decode_tokens=nd,truncated=generated.token_ids[-1] not in eos,
                 read_per_answer=stats["read"],read_per_decode=stats["read"]/max(nd,1),**stats)
        answers.append(ans)
        if audit:
            audits.extend(audit.rows)
            locality.extend(audit.local)
            common_data.extend(audit.common)
    return dict(sample_id=row["sample_id"],task=row["task"],answers=answers,
                audit=audits,locality=locality,common=common_data)


def select_rows(rows,phase):
    if phase=="smoke":
        return rows[:1]
    if phase in ("valid","compact"):
        return rows[8:]
    return rows


def run(root,phase,policies,config):
    split="test" if phase=="test" else "calibration"
    manifest,context,rows=load_rows(root/"data"/split,split)
    rows=select_rows(rows,phase)
    if phase=="test":
        if config["sources"]!=sources() or config["test_manifest"]!=_sha(root/"data/test/manifest.json"):
            raise ValueError("test freeze differs from local code/data")
        if config["test_ids"]!=[r["sample_id"] for r in rows]:
            raise ValueError("held-out ID order differs from freeze")
        if len(rows)!=128:
            raise ValueError("test requires 128 prompts")
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    out=root/phase
    bundles=out/"bundles"
    bundles.mkdir(parents=True,exist_ok=True)
    cfg=out/"config.json"
    setup=dict(base=BASE,context=context,phase=phase,source_sha256=sources(),
               model_sha256=hashes,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
               transformers=transformers.__version__,manifest=_sha(root/"data"/split/"manifest.json"),
               policies=[asdict(p) for p in policies],config=config,
               ids=[r["sample_id"] for r in rows])
    if cfg.exists() and json.loads(cfg.read_text())!=setup:
        raise ValueError(f"existing {phase} settings differ")
    _write_json(cfg,setup)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    e=model.generation_config.eos_token_id
    eos=set(e if isinstance(e,list) else [e])
    windows={k:torch.tensor(v,device="cuda",dtype=torch.bool) for k,v in config.get("maps",{}).items()}
    common=Policy(**config["anchor"]) if phase=="screen" else None
    n_audit={"smoke":1,"baseline":4,"screen":len(rows),"valid":4,"compact":2,"test":8}[phase]
    record={}
    for i,row in enumerate(rows):
        k=f"{i:04d}.json.gz"
        path=bundles/k
        if not path.exists():
            result=run_one(model,tokenizer,eos,row,policies,phase,i<n_audit,common,windows)
            tmp=path.with_suffix(".tmp")
            with gzip.open(tmp,"wt",compresslevel=4) as f:
                json.dump(result,f,allow_nan=False,separators=(",",":"))
            tmp.replace(path)
        record[k]=_sha(path)
        print(f"{phase} {i+1}/{len(rows)}",flush=True)
    _write_json(out/"complete.json",dict(config_sha256=_sha(cfg),prompts=record,predictions=len(rows)*len(policies)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("smoke","baseline","screen","valid","compact","test"))
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    config={}
    if a.phase in ("baseline","smoke"):
        policies=baseline() if a.phase=="baseline" else [Policy("dense"),Policy("single",74),Policy("dynamic")]
    else:
        path=a.root/("frozen_test.json" if a.phase=="test" else f"{a.phase}_plan.json")
        config=json.loads(path.read_text())
        policies=[Policy(**p) for p in config["policies"]]
    run(a.root,a.phase,policies,config)


if __name__=="__main__":
    main()
