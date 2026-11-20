"""Calibrate and test common INT6 refill schedules, with frozen prompt receipts."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re

import torch
import transformers

from gqa_probe.shared_refill import RefillPolicy
from gqa_probe.shared_refill_attention import SharedAttention
from gqa_probe.vskip_summary import StaticPrefix
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.ruler_experiment import _sha,_verified_model,_write_json
from scripts.run_ruler_round3 import load_rows


ROOT=Path("results/round5_shared_refill")
MODEL=Path("models/Qwen3-4B-Instruct-2507")
BASE="9cfae202c992e67ef20c511b84bf49a5086e4d18"
CODE=("gqa_probe/shared_refill.py","gqa_probe/shared_refill_attention.py","gqa_probe/shared_refill_audit.py",
      "gqa_probe/fixed_k_v.py","gqa_probe/vskip_summary.py","gqa_probe/vskip_replay.py",
      "gqa_probe/page_read_budget.py","gqa_probe/oracle_attention.py","gqa_probe/quantization_round2.py",
      "gqa_probe/ruler_inference.py","gqa_probe/paired_statistics.py","scripts/round5_shared_refill.py",
      "scripts/report_round5_shared_refill.py","scripts/round5_pipeline.py")


def code_hashes():
    return {str(f):_sha(Path(f)) for f in CODE}


def calibration1():
    return [*[RefillPolicy("all",budget_bp=b) for b in (1000,950,900)],
            *[RefillPolicy("mass",e) for e in (.02,.03,.05)],
            *[RefillPolicy("mass",.05,b) for b in (950,900)],
            *[RefillPolicy("blasst",e) for e in (.001,.01,.05)],
            *[RefillPolicy("sparta",e,recompute=r) for e in (.01,.02,.05) for r in (False,True)]]


@torch.inference_mode()
def run_one(model,tokenizer,eos,row,policies,sample,frozen=None):
    state=prefill_for_quantization(model,row["input_ids"],"affine32")
    prefix=tuple(StaticPrefix.build(k,32,("q6",)) for k in state.cache.key_cache)
    gold=[]
    for target in row["references"]:
        spans=[x["association_span"] for x in row["source_metadata"]["gold_evidence"] if x["value"]==target]
        gold.append(sorted({t//32 for a,b in spans for t in range(a,b)}))
    result=[]
    traces=[]
    for policy in policies:
        shadow=[]
        if sample and policy==RefillPolicy("all") and frozen:
            shadow=[RefillPolicy(**x) for x in frozen["common_state_shadow"]]
        elif sample and policy==RefillPolicy("all"):
            shadow=[p for p in policies if p.mode!="all" and p.budget_bp==1000]
        attention=SharedAttention(policy,prefix,gold_spans=row["gold_spans"],sample=sample,shadow=shadow,
                                  gold_by_target=gold)
        g=generate_from_prefill(model,state,row["max_new_tokens"],eos,attention)
        decode_tokens=len(g.token_ids)-1
        if g.attention_stats["decode_layer_calls"]!=decode_tokens*model.config.num_hidden_layers:
            raise ValueError("incomplete full-layer decode")
        text=re.sub(r"[\x00-\x1f]","\n",tokenizer.decode(g.token_ids,skip_special_tokens=True).strip()).strip().lower()
        matches=[r.lower() in text for r in row["references"]]
        stats=g.attention_stats
        result.append(dict(policy=policy.name,**asdict(policy),official_score=sum(matches)/len(matches),
                           complete_correct=all(matches),reference_hits=matches,
                           decode_tokens=decode_tokens,truncated=not g.token_ids[-1] in eos,
                           read_per_answer=stats.get("total",0),
                           read_per_decode=stats.get("total",0)/max(1,decode_tokens),**stats))
        traces.extend(attention.rows)
    return dict(sample_id=row["sample_id"],task=row["task"],
                reference_count=len(row["references"]),predictions=result,groups=traces)


def write_bundle(path,result):
    tmp=path.with_suffix(".tmp")
    with gzip.open(tmp,"wt",compresslevel=4) as f:
        json.dump(result,f,separators=(",",":"),allow_nan=False)
    tmp.replace(path)


def run(root,phase,policies,limit=None,frozen=None):
    split="test" if phase=="test" else "calibration"
    manifest,context,rows=load_rows(root/"data"/split,split)
    if context!=32768 or any(r["task"]!="niah_multivalue" for r in rows):
        raise ValueError("unexpected experiment cohort")
    if limit:
        rows=rows[:limit]
    if frozen:
        if (len(rows)!=128 or _sha(root/"data/test/manifest.json")!=frozen["test_manifest_sha256"]
                or code_hashes()!=frozen["sources"]):
            raise ValueError("stale freeze, code, or test manifest")
        if set(frozen["test_ids"])!={r["sample_id"] for r in rows}:
            raise ValueError("test IDs differ from frozen protocol")
        if set(frozen["calibration_instances"]) & {r["source_metadata"]["instance_sha256"] for r in rows}:
            raise ValueError("calibration and test overlap")
        audit_ids=set(frozen["audit_test_ids"])
    else:
        audit_ids={r["sample_id"] for r in rows[:4] if phase in ("cal1","smoke")}
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    config=dict(base=BASE,context=context,phase=phase,source_sha256=code_hashes(),
                model_sha256=hashes,torch=torch.__version__,transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(),manifest_sha256=_sha(root/"data"/split/"manifest.json"),
                policies=[asdict(p) for p in policies],ids=[r["sample_id"] for r in rows],
                audit_ids=sorted(audit_ids),frozen_sha256=_sha(root/"frozen_test.json") if frozen else None)
    out=root/phase
    bundles=out/"bundles"
    bundles.mkdir(parents=True,exist_ok=True)
    cfg=out/"config.json"
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise ValueError(f"cannot resume with changed source/settings at {out}")
    _write_json(cfg,config)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    receipt={}
    for i,row in enumerate(rows):
        name=f"{i:04d}.json.gz"
        path=bundles/name
        if not path.exists():
            write_bundle(path,run_one(model,tokenizer,eos,row,policies,row["sample_id"] in audit_ids,frozen))
        receipt[name]=_sha(path)
        print(f"{phase} {i+1}/{len(rows)}",flush=True)
    _write_json(out/"complete.json",dict(config_sha256=_sha(cfg),predictions=len(rows)*len(policies),
                                        prompts=receipt))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("smoke","cal1","cal2","test"))
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    frozen=None
    if a.phase=="test":
        frozen=json.loads((a.root/"frozen_test.json").read_text())
        policies=[RefillPolicy(**row) for row in frozen["policies"]]
    elif a.phase=="cal2":
        settings=json.loads((a.root/"cal2_policies.json").read_text())
        policies=[RefillPolicy(**row) for row in settings]
    elif a.phase=="smoke":
        policies=[RefillPolicy("all",budget_bp=950),RefillPolicy("mass",.02),
                  RefillPolicy("blasst",.01),RefillPolicy("sparta",.02,recompute=True)]
    else:
        policies=calibration1()
    run(a.root,a.phase,policies,limit=1 if a.phase=="smoke" else None,frozen=frozen)


if __name__=="__main__":
    main()
