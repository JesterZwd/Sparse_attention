"""Run paired A2 closures under the same byte rule and gate independent testing."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
from statistics import mean

import torch
import transformers

from gqa_probe.pagepack_attention import CompressionAttention, Policy, build_prefix
from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.round7_attention import Attention
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts.probe_pagepack import BASE, ROOT
from scripts.report_ruler_round3 import write_csv
from scripts.round7_pipeline import verify_independence
from scripts.round7_run import DATA, DENSE, MODEL, make_answer
from scripts.run_ruler_round3 import load_rows


SUPPORT=Policy("support_single64","q6","raw","hist64",10,0.,68)
NI=.01


def source_hashes():
    modules=sorted(Path("gqa_probe").glob("pagepack*.py"))
    return {str(p):_sha(p) for p in [Path("scripts/run_pagepack.py"),*modules]}


def variants(candidate,budget):
    rows=[("q4","raw"),("mr25","raw")]
    if candidate:
        rows.append(("mr25",candidate))
    return [Policy(f"{s}_{p}_{g}",s,p,g,budget) for s,p in rows for g in ("full","hist64")]


def pack_bundle(data,root):
    print(json.dumps(data,ensure_ascii=False),flush=True)
    _write_json(root,data)


@torch.inference_mode()
def run(root,data,phase,policies,*,count=None,support=False,dense=True,freeze=None):
    split="test" if phase=="independent" else "calibration"
    manifest,context,rows=load_rows(data/split,split)
    if context!=32768 or len(rows)!=(128 if split=="test" else 16):
        raise ValueError("unexpected 32K cohort")
    rows=rows[:count]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    out=root/phase
    bundles=out/"bundles"
    bundles.mkdir(parents=True,exist_ok=True)
    setup=dict(base=BASE,phase=phase,sources=source_hashes(),model=hashes,device=torch.cuda.get_device_name(),
               policies=list(map(asdict,policies)),support=asdict(SUPPORT) if support else None,
               dense=dense,ids=[x["sample_id"] for x in rows],
               manifest_sha256=_sha(data/split/"manifest.json"))
    if freeze and any(setup[k]!=freeze[k] for k in ("ids","policies","manifest_sha256","sources")):
        raise ValueError("independent inputs differ from freeze")
    cfg=out/"config.json"
    if cfg.exists() and json.loads(cfg.read_text())!=setup:
        raise ValueError(f"{phase} cannot reuse results from a different setup")
    _write_json(cfg,setup)
    model=tokenizer=eos=None
    complete={}
    for idx,row in enumerate(rows):
        filename=f"{idx:04d}.json.gz"
        path=bundles/filename
        if not path.exists():
            opts=policies+([SUPPORT] if support and idx<4 else [])
            if model is None:
                model=transformers.AutoModelForCausalLM.from_pretrained(
                    MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
                    low_cpu_mem_usage=True,local_files_only=True).eval()
                tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
                tokens=model.generation_config.eos_token_id
                eos=set(tokens if isinstance(tokens,list) else [tokens])
            base=prefill_for_quantization(model,row["input_ids"],"affine32")
            prefix=build_prefix(base,set(x.spec for x in opts),sorted({x.summary for x in opts}))
            results,timings,groups=[],[],[]
            order=([DENSE] if dense else [])+opts
            for p in order:
                gold=row["gold_spans"]
                attn=Attention(DENSE,None,gold) if p==DENSE else CompressionAttention(p,prefix,gold,audit=True)
                result=generate_from_prefill(model,base,row["max_new_tokens"],eos,attn)
                r=make_answer(p,result,tokenizer,row,eos)
                if r["decode_layer_calls"]!=r["decode_tokens"]*model.config.num_hidden_layers:
                    raise ValueError("decode accounting missing layers")
                if r.get("over_budget_group_steps"):
                    raise ValueError("decode route exceeded physical cap")
                results.append(dict(sample_id=row["sample_id"],task=row["task"],**r))
                timings.extend(dict(policy=p.name,**v) for v in attn.timing)
                if p!=DENSE and attn.group_steps:
                    steps=torch.stack([v[3] for v in attn.group_steps]).cpu().tolist()
                    checked=[]
                    for (step,layer,cap,_),gs in zip(attn.group_steps,steps):
                        if any(g[0]>cap or g[0]%32 for g in gs):
                            raise ValueError("real group step violates read budget")
                        checked.append([step,layer,cap,gs])
                    if sum(g[0] for x in checked for g in x[3])!=r["read"]:
                        raise ValueError("group trace and prompt ledger disagree")
                    groups.append(dict(policy=p.name,steps=checked))
            tmp=path.with_suffix(".tmp")
            with gzip.open(tmp,"wt",compresslevel=4) as f:
                json.dump(dict(sample_id=row["sample_id"],results=results,timings=timings,groups=groups),
                          f,allow_nan=False,separators=(",",":"))
            tmp.replace(path)
        complete[filename]=_sha(path)
        print(f"{phase} {idx+1}/{len(rows)}",flush=True)
    _write_json(out/"complete.json",dict(config_sha256=_sha(cfg),files=complete))


def report(root,phase):
    out=root/phase
    cfg=out/"config.json"
    rec=json.loads((out/"complete.json").read_text())
    if rec["config_sha256"]!=_sha(cfg):
        raise ValueError("closure config mismatch")
    dest=root/"report"/phase
    dest.mkdir(parents=True,exist_ok=True)
    preds=[]
    timing=defaultdict(list)
    with gzip.open(dest/"group_steps.jsonl.gz","wt",compresslevel=4) as ledger, \
         gzip.open(dest/"timings.jsonl.gz","wt",compresslevel=4) as trace:
        for filename,sha in rec["files"].items():
            path=out/"bundles"/filename
            if _sha(path)!=sha:
                raise ValueError("prompt receipt mismatch")
            with gzip.open(path,"rt") as fh:
                x=json.load(fh)
            preds.extend(x["results"])
            for v in x["timings"]:
                timing[v["policy"]].append(v)
                trace.write(json.dumps(dict(sample_id=x["sample_id"],**v),separators=(",",":"))+"\n")
            for v in x["groups"]:
                ledger.write(json.dumps(dict(sample_id=x["sample_id"],**v),separators=(",",":"))+"\n")
    _write_json(dest/"predictions.json",preds)
    policies=defaultdict(list)
    for r in preds:
        policies[r["policy"]].append(r)
    rows=[]
    for p,rs in policies.items():
        nd=sum(r["decode_tokens"] for r in rs)
        ng=sum(r["kv_group_calls"] for r in rs)
        scale=lambda k:sum(r.get(k,0) for r in rs)/max(1,nd)
        t=timing[p]
        rows.append(dict(policy=p,prompts=len(rs),official_score=mean(r["official_score"] for r in rs),
                      complete=sum(r["complete_correct"] for r in rs),group_steps=ng,
                      over_budget=sum(r.get("over_budget_group_steps",0) for r in rs),
                      fallback_rate=sum(r.get("fallback_group_steps",0) for r in rs)/max(1,ng),
                      bytes_per_decode=scale("read"),
                      bytes_per_answer=mean(r["read"] for r in rs),
                      reads_k_per_decode=scale("k_bytes"), reads_v_per_decode=scale("v_bytes"),
                      summary_bytes_per_decode=scale("summary_bytes"), directory_bytes_per_decode=scale("directory_bytes"),
                      tags_bytes_per_decode=scale("tag_bytes"),
                      qk_mac_per_decode=scale("qk_mac"),pv_mac_per_decode=scale("pv_mac"),
                      scorer_per_decode=scale("scorer_mac"),v_reduce_adds_per_decode=scale("v_pool_reduction_adds"),
                      mean_fetched_pages=sum(r.get("fetched_pages",0) for r in rs)/max(1,ng),
                      mean_retained_prefix=sum(r.get("retained_prefix_pages",0) for r in rs)/max(1,ng),
                      k_evidence_recall=sum(r.get("evidence_k_page_hits",0) for r in rs)/
                                  max(1,sum(r.get("evidence_k_page_total",0) for r in rs)),
                      v_evidence_recall=sum(r["gold_page_hits"] for r in rs)/max(1,sum(r["gold_page_total"] for r in rs)),
                      avg_prefill_summary_write=mean(r.get("prefill_summary_write_bytes",0) for r in rs),
                      avg_prefill_kv_write=mean(r.get("prefill_representative_write_bytes",0) for r in rs),
                      kv_write_per_decode=scale("representative_write_bytes"),
                      k_stat_adds_per_decode=scale("shared_k_stat_channel_adds"),
                      k_stat_compare_per_decode=scale("shared_k_stat_channel_compare"),
                      v_stat_adds_per_decode=scale("v_stat_released_page_channel_adds"),
                      timing_states=len(t),
                      banked_ck=mean(r["banked_completion_ck"] for r in t) if t else None,
                      peak_attention_sram_bytes=max((r["attention_sram_bytes"] for r in t),default=None),
                      peak_logit_sram_bytes=max((r["logits_sram_bytes"] for r in t),default=None),
                      peak_control_sram_bytes=max((r["control_sram_bytes"] for r in t),default=None)))
    write_csv(dest/"quality_cost.csv",rows)
    return rows,preds


def pairing(path,preds,base,candidate):
    r={k:paired_bootstrap(preds,base,candidate,k,confidence=.95,method_key="policy")
       for k in ("official_score","read_per_decode","read")}
    a={p["sample_id"]:p for p in preds if p["policy"]==base}
    b={p["sample_id"]:p for p in preds if p["policy"]==candidate}
    r["prompt_deltas"]=[dict(sample_id=s,score_delta=b[s]["official_score"]-x["official_score"],
                             complete_base=x["complete_correct"],complete_candidate=b[s]["complete_correct"])
                        for s,x in sorted(a.items())]
    _write_json(path,r)
    return r


def gain(row,ref,key):
    return 1-row[key]/ref[key]


def nominate(rows,opts):
    by={r["policy"]:r for r in rows}
    ref=max((p for p in opts if p.packing=="raw"),
            key=lambda p:(by[p.name]["official_score"],-by[p.name]["bytes_per_decode"]))
    reference=by[ref.name]
    values=[]
    for p in opts:
        if p.packing=="raw":
            continue
        r=by[p.name]
        quality=r["official_score"]-reference["official_score"]
        saving=gain(r,reference,"bytes_per_decode")
        cycle=gain(r,reference,"banked_ck")
        reads_close=saving>=-.01
        improvement=quality>=.01 or max(saving,cycle)>=.05
        passed=not r["over_budget"] and quality>=-NI and reads_close and improvement
        values.append((p,dict(packing=p.name,delta=quality,read_saving=saving,cycle_saving=cycle,passes=passed)))
    choice=max((x for x in values if x[1]["passes"]),key=lambda x:(x[1]["delta"],x[1]["read_saving"]),default=None)
    return ref,choice[0] if choice else None,[v for _,v in values]


def program(root,data):
    choice=json.loads((root/"a1_decision.json").read_text())
    candidate=choice["candidate"]
    opts=variants(candidate,5)
    _write_json(root/"a2_protocol.json",dict(packing=candidate,calibration=list(map(asdict,opts)),
                gates="shared .65 optional V reservation; hist64 safe-prefix rerun",
                first_more_budget_percent=7,ni_margin=NI,
                criteria="calibration at least −1 pp, within +1% actual reads, +1 pp score or 5% cycle/read gain"))
    run(root,data,"cap5",opts,support=True)
    rows,pred=report(root,"cap5")
    phase="cap5"
    if candidate:
        max_cap5=max(r["official_score"] for r in rows if any(r["policy"]==p.name and p.packing==candidate for p in opts))
        matched=[replace(p,name="fixed_"+p.name,max_pages=20) for p in opts
                 if p.summary=="mr25" and p.gate=="full"]
        run(root,data,"fixed_pages",matched,count=4,dense=False)
        _,m=report(root,"fixed_pages")
        pairing(root/"report"/"fixed_pages"/"pair.json",m,matched[0].name,matched[1].name)
    reference,final,observed=nominate(rows,opts)
    if candidate and final is None and max_cap5<.95:
        opts=variants(candidate,7)
        phase="cap7"
        run(root,data,phase,opts,dense=False)
        rows,pred=report(root,phase)
        reference,final,extra=nominate(rows,opts)
        observed+=extra
    if candidate:
        for policy in opts:
            if policy.packing==candidate:
                raw=next(p for p in opts if p.summary=="mr25" and p.packing=="raw" and p.gate==policy.gate)
                pairing(root/"report"/phase/f"pair_{policy.gate}.json",pred,raw.name,policy.name)
    outcome=dict(status="calibration_complete",candidate=final.name if final else None,
                 a1_candidate=candidate,reference=reference.name,base_phase=phase,checks=observed,
                 heldout=False,passes=False)
    if final:
        verify_independence(data)
        _,_,test=load_rows(data/"test","test")
        freeze=dict(ids=[r["sample_id"] for r in test],policies=list(map(asdict,(reference,final))),
                    manifest_sha256=_sha(data/"test"/"manifest.json"),sources=source_hashes())
        _write_json(root/"frozen_test.json",freeze)
        run(root,data,"independent",[reference,final],freeze=freeze)
        _,pred=report(root,"independent")
        paired=pairing(root/"report"/"independent"/"pair.json",pred,reference.name,final.name)
        score=paired["official_score"]
        outcome.update(status="independent_complete",heldout=True,passes=score["ci_low"]>=-NI,
                       score_delta=score["point_delta"],ci_low=score["ci_low"],ci_high=score["ci_high"])
    pack_bundle(outcome,root/"finished.json")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    a=p.parse_args()
    program(a.root,a.data)


if __name__=="__main__":
    main()
