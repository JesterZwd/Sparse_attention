"""Run same-K and closed-loop experiments on pinned RULER cohorts."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import torch
import transformers

from gqa_probe.oracle_attention import OracleAttention, grouped_logits, torch_page_mass
from gqa_probe.page_read_budget import page_history
from gqa_probe.page_summary import OnlineSummary, PrefixSummary, summary_weights
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import _copy_prefix, generate_from_prefill
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.vskip_attention import FixedKAttention, VPolicy
from gqa_probe.vskip_summary import METHOD_BYTES, StaticOnline, StaticPrefix, weights
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round4_fixed_k_v")
MODEL = Path("models/Qwen3-4B-Instruct-2507")
ORIGINAL = Path("results/round3_summary_bytes/data/16k")
SOURCES = (
    "scripts/run_round4_fixed_k_v.py",
    "gqa_probe/vskip_attention.py", "gqa_probe/vskip_summary.py", "gqa_probe/vskip_replay.py",
    "gqa_probe/fixed_k_v.py", "gqa_probe/oracle_attention.py", "gqa_probe/page_summary.py",
    "gqa_probe/page_read_budget.py", "gqa_probe/ruler_inference.py",
    "gqa_probe/ruler_data.py", "gqa_probe/quantization_round2.py",
)
BASE = "4fd60cac0f62295c499e7d0e708ef2f2073fa01a"


def policies_for(cohort):
    if cohort == "diagnostic":
        return [
            VPolicy("q8",legacy_layout=True),
            VPolicy("q6"), VPolicy("q4"),
            VPolicy("q6",equal_pages=True,legacy_layout=True),
            VPolicy("q4",equal_pages=True,legacy_layout=True),
            VPolicy("q8",64,15,legacy_layout=True),
            VPolicy("oracle",64,15,legacy_layout=True),
            VPolicy("mr25"),VPolicy("mr50"),
            *[VPolicy("q8",mode="mass",threshold=e) for e in (.005,.01,.02,.05)],
            *[VPolicy("q8",mode="blasst",threshold=e) for e in (.01,.1,.5)],
            VPolicy("q8",mode="sparta",threshold=.02),VPolicy("q8",mode="sparta",threshold=.05),
        ]
    return [
        VPolicy("dense"), VPolicy("q8"),VPolicy("q6"),VPolicy("q4"),
        VPolicy("q6",equal_pages=True),VPolicy("q4",equal_pages=True),
        VPolicy("q8",percent=8),VPolicy("q6",percent=8),VPolicy("q4",percent=8),
        VPolicy("mr25"),VPolicy("mr50"),
        *[VPolicy("q8",mode="mass",threshold=e) for e in (.005,.01,.02,.05)],
        *[VPolicy("q8",mode="blasst",threshold=e) for e in (.01,.1,.5)],
        VPolicy("q8",mode="sparta",threshold=.02),VPolicy("q8",mode="sparta",threshold=.05),
        *[VPolicy("q8",percent=8,mode="mass",threshold=e) for e in (.005,.01,.02,.05)],
        *[VPolicy("q8",percent=8,mode="blasst",threshold=e) for e in (.01,.1,.5)],
        VPolicy("q8",percent=8,mode="sparta",threshold=.02),
        VPolicy("q8",percent=8,mode="sparta",threshold=.05),
    ]


class LegacyCentroid(OracleAttention):
    def __init__(self,state):
        super().__init__()
        self.prefix=[PrefixSummary.build(k,32) for k in state.cache.key_cache]
        self.online={}
    def attention(self,module,query,key,value,attention_mask,**kwargs):
        from gqa_probe.oracle_attention import grouped_output
        t=key.shape[-2]
        h=page_history(t,32)
        g=key.shape[1]
        layer=module.layer_idx
        scaling=kwargs.get("scaling") or query.shape[-1]**-.5
        if layer not in self.online:
            self.online[layer]=OnlineSummary(self.prefix[layer])
        metadata=self.online[layer].append(key[0,:,-1],t)
        q=summary_weights(query,metadata,"centroid8",scaling)
        from gqa_probe.page_read_budget import read_ledger
        count=read_ledger(h,"centroid8",percent=10)["selected_pages"]
        ids=ranked_pages(q,h)[:,:count]
        mask=torch.zeros_like(q,dtype=torch.bool).scatter_(1,ids,True)
        output=grouped_output(grouped_logits(query,key,scaling),value,mask,32)
        return output.reshape(1,query.shape[1],1,query.shape[-1]).to(query.dtype).transpose(1,2).contiguous(),None


class ForcedDiagnostic(FixedKAttention):
    def __init__(self,policy,prefix,step,gold):
        super().__init__(policy,prefix=prefix)
        self.target=step
        self.gold=sorted({i//32 for a,b in gold for i in range(a,b)})
        self.rows=[]
    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step==self.target:
            tokens=key.shape[-2]
            h=page_history(tokens,32)
            scaling=kwargs.get("scaling") or query.shape[-1]**-.5
            q=query.reshape(key.shape[1],-1,query.shape[-1]).float()
            methods=("q8","q6","q4","mr25","mr50")
            prefix=StaticPrefix.build(key[:,:,:-1],32,methods)
            data=StaticOnline(prefix,methods).append(key[0,:,-1],tokens)
            logits=grouped_logits(query,key,scaling)
            mass=torch_page_mass(logits,32)
            per_page=torch.nn.functional.pad(logits,(0,(-tokens)%32),value=-torch.inf).unflatten(-1,(-1,32))
            peak=per_page.amax(-1)
            concentration=(peak-per_page.logsumexp(-1)).exp()
            centroid=OnlineSummary(PrefixSummary.build(key[:,:,:-1],32)).append(key[0,:,-1],tokens)
            scores={m:weights(q,data[m],m,scaling) for m in methods}
            scores["centroid8"]=summary_weights(query,centroid,"centroid8",scaling)
            scores["oracle"]=mass.sum(-1)
            from gqa_probe.page_read_budget import read_ledger
            budgets={m:VPolicy(m).count(tokens) for m in methods}
            budgets["centroid8"]=read_ledger(h,"centroid8",percent=10)["selected_pages"]
            budgets["oracle"]=self.policy.count(tokens)
            rankings={m:(ranked_pages(score,h).argsort(-1)[:,self.gold]+1).cpu().tolist() for m,score in scores.items()}
            masses=mass[:,self.gold].cpu().tolist()
            peaks=peak[:,:,self.gold].transpose(1,2).cpu().tolist()
            concentrations=concentration[:,:,self.gold].transpose(1,2).cpu().tolist()
            for g in range(q.shape[0]):
                for i,page in enumerate(self.gold):
                    self.rows.append(dict(layer=module.layer_idx,group=g,page=page,mass=masses[g][i],
                                          peak=peaks[g][i], concentration=concentrations[g][i],
                                          **{f"{m}_rank":v[g][i] for m,v in rankings.items()},
                                          **{f"{m}_budget":n for m,n in budgets.items()}))
        return super().attention(module,query,key,value,attention_mask,**kwargs)


def fixed_diagnostic(model,tokenizer,state,row,eos,prefix,q_tokens):
    limit=row["max_new_tokens"]
    alt=generate_from_prefill(model,state,limit,eos,LegacyCentroid(state)).token_ids
    divergence=next((i for i,(a,b) in enumerate(zip(q_tokens,alt)) if a!=b),None)
    if divergence is None or divergence==0:
        return dict(first_divergence=divergence,reason="no divergent decode token",rows=[])
    # Collect aligned states at first divergence and at the first missing
    # Quest answer value, if it occurs later than the first token difference.
    targets={divergence:"first_divergence"}
    offsets=tokenizer.decode(q_tokens,skip_special_tokens=True)
    alt_text=tokenizer.decode(alt,skip_special_tokens=True)
    for ref in row["references"]:
        if ref.lower() not in offsets.lower() or ref.lower() in alt_text.lower():
            continue
        position=next((i for i in range(divergence,len(q_tokens))
                       if ref.lower() in tokenizer.decode(q_tokens[:i+1],skip_special_tokens=True).lower()),None)
        if position:
            targets.setdefault(position,"first_completed_missing_reference")
    attention=ForcedDiagnostic(VPolicy("q8",legacy_layout=True),prefix,min(targets)-1,row["gold_spans"])
    all_rows=[]
    cache=_copy_prefix(state.cache)
    with attention:
        for i,token in enumerate(q_tokens[:max(targets)]):
            attention.step=i
            attention.target=i if i+1 in targets else -2
            prev=len(attention.rows)
            model(torch.tensor([[token]],device=state.last_logits.device),past_key_values=cache,use_cache=True,logits_to_keep=1)
            all_rows.extend(dict(kind=targets[i+1],decode_step=i,**d) for d in attention.rows[prev:])
    return dict(first_divergence=divergence,common_prefix_tokens=divergence,
                branches=dict(quest=score_prediction(offsets,row["references"]),
                              centroid=score_prediction(alt_text,row["references"])),rows=all_rows)


@torch.inference_mode()
def run_subject(model,tokenizer,eos,row,policies,cohort,sample_curves):
    state=prefill_for_quantization(model,row["input_ids"],"affine32")
    sets={p:{x.scorer for x in policies if x.page==p and x.scorer in METHOD_BYTES} for p in (32,64)}
    prefixes={p:tuple(StaticPrefix.build(k,p,sorted(ms)) for k in state.cache.key_cache) for p,ms in sets.items() if ms}
    result=dict(sample_id=row["sample_id"],task=row["task"],paths=[],curve=[],schedule=[],progressive=[])
    for policy in policies:
        audit_budget = policies[0].percent if cohort == "test" else 10
        audit=sample_curves and policy==VPolicy("q8",percent=audit_budget,legacy_layout=cohort=="diagnostic")
        attn=FixedKAttention(policy,gold_spans=row["gold_spans"],prefix=prefixes.get(policy.page),
                             audit=audit,residual=cohort=="diagnostic" and row["sample_id"].endswith(("0000","0006","0010")))
        generated=generate_from_prefill(model,state,row["max_new_tokens"],eos,attn)
        text=re.sub(r"[\x00-\x1f]","\n",tokenizer.decode(generated.token_ids,skip_special_tokens=True).strip())
        if generated.attention_stats["decode_layer_calls"]!=(len(generated.token_ids)-1)*model.config.num_hidden_layers:
            raise ValueError("decode callbacks incomplete")
        result["paths"].append(dict(policy=policy.name,**asdict(policy),
                                    **score_prediction(text,row["references"]),**generated.attention_stats,
                                    decode_tokens=len(generated.token_ids)-1,
                                    truncated=generated.token_ids[-1] not in eos))
        if audit:
            for key in ("curve","schedule","progressive"):
                result[key]=getattr(attn,key)
        if cohort=="diagnostic" and row["sample_id"].endswith("0006") and audit:
            result["diagnostic"]=fixed_diagnostic(model,tokenizer,state,row,eos,prefixes[32],generated.token_ids)
    return result


def write_bundle(path,result):
    with gzip.open(path.with_suffix(".tmp"),"wt",compresslevel=4) as f:
        json.dump(result,f,separators=(",",":"),allow_nan=False)
    path.with_suffix(".tmp").replace(path)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("cohort",choices=("diagnostic","32k_mv","16k_fwe","test"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--frozen",type=Path)
    args=parser.parse_args()
    frozen=json.loads(args.frozen.read_text()) if args.frozen else None
    if (args.cohort=="test")!=(frozen is not None):
        raise ValueError("held-out inference requires frozen config")
    data=ORIGINAL if args.cohort=="diagnostic" else args.root/"data"/args.cohort
    manifest,context,rows=load_rows(data,"test" if frozen else "calibration")
    rows=[r for r in rows if args.cohort!="diagnostic" or r["task"]=="niah_multivalue"]
    if frozen:
        if _sha(data/"manifest.json")!=frozen["test_manifest_sha256"]:
            raise ValueError("test data differs from frozen manifest")
        if len(rows)!=frozen["sample_count"] or _sha(Path(__file__))!=frozen["runner_sha256"]:
            raise ValueError("test size or runner differs from frozen config")
        if {r["source_metadata"]["instance_sha256"] for r in rows}&set(frozen["calibration_instances"]):
            raise ValueError("calibration and test retrieval instances overlap")
    if args.limit:
        rows=rows[:args.limit]
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model_source,model_hashes=_verified_model(args.model,manifest)
    output=args.root/args.cohort
    output.mkdir(parents=True,exist_ok=True)
    bundle=output/"bundles"
    bundle.mkdir(exist_ok=True)
    policies=[VPolicy(**p) for p in frozen["policies"]] if frozen else policies_for(args.cohort)
    config=dict(base=BASE,context=context,model=model_source.get("model"),model_hashes=model_hashes,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
                data_manifest_sha256=_sha(data/"manifest.json"),
                source_sha256={f:_sha(Path(f)) for f in SOURCES},
                policies=[asdict(p) for p in policies],ids=[r["sample_id"] for r in rows],
                frozen_sha256=_sha(args.frozen) if frozen else None)
    cfg=output/"config.json"
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise ValueError("cannot resume changed cohort")
    _write_json(cfg,config)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        args.model,torch_dtype=torch.bfloat16,attn_implementation="sdpa",
        device_map="cuda",low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    hashes={}
    for i,row in enumerate(rows):
        path=bundle/(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:20]+".json.gz")
        if not path.exists():
            write_bundle(path,run_subject(model,tokenizer,eos,row,policies,args.cohort,
                                          sample_curves=not frozen or i<8))
        hashes[path.name]=_sha(path)
        print(f"{args.cohort}: {i+1}/{len(rows)}",flush=True)
    _write_json(output/"complete.json",dict(config_sha256=_sha(cfg),prompts=hashes,n=len(rows)))
    print(f"{args.cohort} complete",flush=True)


if __name__=="__main__":
    main()
