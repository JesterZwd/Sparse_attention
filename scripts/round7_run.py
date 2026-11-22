"""Run paired Round 7 phases using the immutable affine32 prefix."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import mean

import torch
import transformers

from gqa_probe.fixed_k_v import byte_quota, gather_tokens, masked_output, ranked_pages
from gqa_probe.oracle_attention import grouped_logits, grouped_output
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.page_read_budget import page_history
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.round6_attention import SAMPLE_LAYERS, SAMPLE_STEPS, physical_ledger
from gqa_probe.round7_attention import Attention, Policy, timing_rows
from gqa_probe.round7_sides import SideScreen
from gqa_probe.round7_single import decide_v, prepare_single
from gqa_probe.round7_summary_diag import summary_diagnostics
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.vskip_summary import StaticOnline, StaticPrefix, weights
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round7_budget_gates")
DATA = Path("results/round6_subpage_head/data")
MODEL = Path("models/Qwen3-4B-Instruct-2507")
BASE = "4f0dd8a7d2b1f5ab5a29627f6c0218916db16c0c"
DENSE = Policy("dense_kv4", kind="dense")
DYNAMIC = Policy("dynamic", kind="dynamic", max_pages=0)
CAPPED = Policy("dynamic_68", kind="dynamic")
SINGLE = Policy("single_exact")
HIST = Policy("single_hist64", gate="hist64")
STATIC = Policy("static_int6", gate="full", bonus=0)
COMPACT = [Policy(f"{m}_5_{label}", method=m, budget=5, gate="full" if bonus == 0 else "exact",
                  bonus=bonus) for m in ("q4", "mr25") for label, bonus in
           (("allv", 0), ("plus4", 4), ("max", None))]
FIXED = [SINGLE, HIST, *[Policy(f"fixed_blasst_{x:g}", gate="blasst", threshold=x)
                        for x in (.001, .003, .01)],
         *[Policy(f"fixed_checked_{x:g}", gate="checked", threshold=x) for x in (6., 8., 10., 12.)]]


def sources():
    paths = [*Path("gqa_probe").glob("round7*.py"), *Path("scripts").glob("round7*.py")]
    paths += list(map(Path, ("gqa_probe/shared_refill.py", "gqa_probe/fixed_k_v.py",
                            "gqa_probe/round6_memory.py", "gqa_probe/round6_audit.py",
                            "gqa_probe/vskip_summary.py", "sim/round6_lpddr5.cpp")))
    return {str(p): _sha(p) for p in sorted(paths)}


def pages_for_span(span):
    return range(span[0]//32, (span[1]-1)//32+1)


def answer_queries(tokenizer, ids, row):
    prefixes = [tokenizer.decode(ids[:i+1], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False).lower() for i in range(len(ids))]
    starts = {}
    if not prefixes:
        return starts
    for evidence in row["source_metadata"]["gold_evidence"]:
        off = prefixes[-1].find(evidence["value"].lower())
        if off < 0:
            continue
        idx = next((i for i, p in enumerate(prefixes) if p[:off+1] == prefixes[-1][:off+1]), None)
        if idx is None or idx == 0:
            continue
        starts.setdefault(idx-1, set()).update(pages_for_span(evidence["value_span"]))
    return starts


class Common(Attention):
    def __init__(self, row, prefix, starts, policies, sides):
        super().__init__(DENSE, None, row["gold_spans"])
        self.starts = starts
        self.policy_screen = policies
        self.replay_prefix = prefix
        self.side = SideScreen() if sides else None
        self.summary_rows = []
        self.gate_rows = []

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        g, tokens, dim = key.shape[1:]
        h = page_history(tokens, 32)
        scale = kwargs.get("scaling") or dim**-.5
        layer = module.layer_idx
        q = query.reshape(g, -1, dim).float()
        raw = grouped_logits(query, key, scale)
        dense = grouped_output(raw, value)
        side_step = self.side and layer in self.side.layers
        sampled = layer in SAMPLE_LAYERS and (self.step in SAMPLE_STEPS or self.step in self.starts)
        if not (sampled or side_step):
            return self.output(query, dense)
        # Rebuild only inside offline diagnostics. Closure maintains online records.
        methods = ("q6", "q4", "mr25") if sampled else ("q6",)
        if side_step:
            if layer not in self.online:
                self.online[layer] = StaticOnline(self.replay_prefix[layer], ("q6",))
            record = self.online[layer].append(key[0, :, -1], tokens)
        else:
            record = {}
        if sampled:
            missing = tuple(m for m in methods if m not in record)
            sealed = StaticPrefix.build(key[:, :, :-1], 32, missing)
            record.update(StaticOnline(sealed, missing).append(key[0, :, -1], tokens))
        evidence = sorted(self.starts.get(self.step, self._gold_page_tokens))
        is_answer = self.step in self.starts
        w = weights(q, record["q6"], "q6", scale)
        ranked = ranked_pages(w, h)
        if side_step:
            self.side.observe(self.step, layer, h, q, key, value, raw, ranked, w,
                              record["q6"], scale, evidence, new_value_head=is_answer)
        if not sampled:
            return self.output(query, dense)
        rows = summary_diagnostics(q, key, value, h, record, scale,
                                   count=byte_quota(tokens,32,"q4",5), evidence=evidence,
                                   answer_start=is_answer, logits=raw, dense_output=dense)
        self.summary_rows.extend(dict(step=self.step, layer=layer, **x) for x in rows)
        for budget, method in sorted({(p.budget, p.method) for p in self.policy_screen}):
            policies = [p for p in self.policy_screen if (p.budget,p.method) == (budget,method)]
            priority = ranked if method == "q6" else ranked_pages(weights(q,record[method],method,scale),h)
            # Fixed read cap and repeated policies must consume exactly the same K.
            limit = policies[0].page_limit(tokens)
            if any(p.page_limit(tokens) != limit for p in policies):
                raise ValueError("fixed-K replay has unequal page limits")
            prepared = prepare_single(q,key,priority,tokens,scale,method,budget*100,limit)
            source = prepared[0]
            vv = gather_tokens(value,source.positions)
            ref = masked_output(source.logits,vv,source.mask)
            group_active = torch.ones(g,dtype=torch.bool,device=q.device)
            for p in policies:
                out = decide_v(prepared,method,p.gate,p.threshold)
                z = masked_output(source.logits,vv,out.final.mask)
                diff=(z-ref).norm(dim=-1)
                abs_err=diff.cpu().tolist()
                rel_err=(diff/(ref.norm(dim=-1)+1e-8)).cpu().tolist()
                dense_err=(z-dense).norm(dim=-1).cpu().tolist()
                kept = out.final.mask.any((1,3))
                evidence_v = (torch.isin(source.ids, torch.tensor(evidence, dtype=torch.int64,
                               device=q.device)) & kept).sum(-1).cpu().tolist()
                led = physical_ledger(out.final,h,method,group_active)
                vb = led["v_bytes"].cpu().tolist()
                for x in timing_rows(h,p,priority,out,led):
                    i=x["group"]
                    self.gate_rows.append(dict(step=self.step,layer=layer,policy=p.name,
                                               answer_start=is_answer, head_abs=abs_err[i],
                                               head_rel=rel_err[i], head_dense_abs=dense_err[i],
                                               v_bytes=vb[i], evidence_v_pages=evidence_v[i],
                                               evidence_total=len(evidence), **x))
        return self.output(query, dense)


def make_answer(policy, result, tokenizer, row, eos):
    n = len(result.token_ids)-1
    pred = tokenizer.decode(result.token_ids, skip_special_tokens=True)
    stats = result.attention_stats
    return dict(policy=policy.name, **score_prediction(pred, row["references"]),
                truncated=result.token_ids[-1] not in eos, decode_tokens=n,
                numerical_decode_seconds=result.elapsed_seconds,
                read_per_decode=stats["read"]/max(n,1), **stats)


@torch.inference_mode()
def run_one(model, tokenizer, eos, row, policies, common_policies, audit, sides):
    input_ids = row["input_ids"]
    base = prefill_for_quantization(model, input_ids, "affine32")
    methods = tuple(sorted({p.method for p in policies if p.kind != "dense"} |
                           ({"q6"} if common_policies else set())))
    prefix = tuple(StaticPrefix.build(k,32,methods) for k in base.cache.key_cache) if methods else None
    result = dict(sample_id=row["sample_id"],task=row["task"],answers=[],
                  prefill_seconds=base.elapsed_seconds,timings=[],summary_diag=[],
                  gate_diag=[],sides=None,group_steps=[])
    dense = None
    for p in policies:
        attn = Attention(p,prefix,row["gold_spans"],audit=audit)
        answer = generate_from_prefill(model,base,row["max_new_tokens"],eos,attn)
        r = make_answer(p,answer,tokenizer,row,eos)
        if r["decode_layer_calls"] != r["decode_tokens"] * model.config.num_hidden_layers:
            raise ValueError("callback coverage mismatch")
        result["answers"].append(r)
        result["timings"].extend(dict(policy=p.name,**x) for x in attn.timing)
        if p.kind == "dense":
            dense = answer
        if attn.group_stats:
            raw = torch.stack([x[3] for x in attn.group_stats]).cpu().tolist()
            result["group_steps"].append(dict(policy=p.name, steps=[
                [step, layer, cap, values] for (step,layer,cap,_),values in zip(attn.group_stats,raw)]))
    if common_policies:
        if dense is None:
            attn = Attention(DENSE,None,row["gold_spans"])
            dense = generate_from_prefill(model,base,row["max_new_tokens"],eos,attn)
        starts = answer_queries(tokenizer,dense.token_ids,row)
        attn = Common(row,prefix,starts,common_policies,sides)
        replay = generate_from_prefill(model,base,row["max_new_tokens"],eos,attn)
        if replay.token_ids != dense.token_ids:
            raise ValueError("common dense trace changed its output")
        result.update(summary_diag=attn.summary_rows,gate_diag=attn.gate_rows,
                      sides=attn.side.finalize() if attn.side else None)
    return result


def run(root, data, phase, policies, *, start=0, count=None, common=(), side_count=0, audits=None,
        device="cuda", threads=4):
    split = "test" if phase == "test" else "calibration"
    path = data/split
    manifest,context,rows = load_rows(path,split)
    rows = rows[start:None if count is None else start+count]
    if not rows or context != 32768 or any(r["task"] != "niah_multivalue" for r in rows):
        raise ValueError("Round 7 expects nonempty 32K Multi-Value")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes = _verified_model(MODEL,manifest)
    output = root/phase
    bundles = output/"bundles"
    bundles.mkdir(parents=True,exist_ok=True)
    path_config = output/"config.json"
    config = dict(base=BASE,phase=phase,context=context,model_sha256=hashes,
                  manifest_sha256=_sha(path/"manifest.json"),ids=[r["sample_id"] for r in rows],
                  policies=list(map(asdict,policies)),common=list(map(asdict,common)),
                  source_sha256=sources(),torch=torch.__version__,transformers=transformers.__version__,
                  device=device,compute_device=torch.cuda.get_device_name() if device=="cuda" else "CPU",
                  threads=threads,side_count=side_count,audits=audits)
    if phase == "test":
        freeze = json.loads((root/"frozen_test.json").read_text())
        if len(rows) != 128 or any(config[k] != freeze[k] for k in
                                   ("manifest_sha256","source_sha256","ids","policies")):
            raise ValueError("held-out freeze mismatch")
    if path_config.exists() and json.loads(path_config.read_text()) != config:
        raise ValueError(f"previous {phase} settings changed")
    _write_json(path_config,config)
    receipt = {}
    model = tokenizer = eos = None
    for i,row in enumerate(rows):
        name = f"{i:04d}.json.gz"
        target = bundles/name
        if not target.exists():
            if model is None:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map=device,
                    low_cpu_mem_usage=True,local_files_only=True).eval()
                tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
                ids = model.generation_config.eos_token_id
                eos = set(ids if isinstance(ids,list) else [ids])
            out = run_one(model,tokenizer,eos,row,policies,common,audits is None or i<audits,i<side_count)
            temp = target.with_suffix(".tmp")
            with gzip.open(temp,"wt",compresslevel=3) as stream:
                json.dump(out,stream,allow_nan=False,separators=(",",":"))
            temp.replace(target)
        receipt[name] = _sha(target)
        print(f"{phase} {i+1}/{len(rows)}",flush=True)
    _write_json(output/"complete.json",dict(config_sha256=_sha(path_config),prompts=receipt,
                                           predictions=len(rows)*len(policies)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase",choices=("smoke","baseline","compact","screen"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--threads",type=int,default=4)
    a = p.parse_args()
    opts = dict(device=a.device, threads=a.threads)
    if a.phase == "baseline":
        run(a.root,a.data,a.phase,[DENSE,DYNAMIC,CAPPED,SINGLE,STATIC],**opts)
    elif a.phase == "compact":
        run(a.root,a.data,a.phase,COMPACT,**opts)
    elif a.phase == "screen":
        run(a.root,a.data,a.phase,[DENSE],count=8,common=FIXED,side_count=4,audits=0,**opts)
    else:
        run(a.root,a.data,a.phase,[DENSE,CAPPED,SINGLE,COMPACT[1],HIST],
            count=1,common=FIXED[:2],side_count=1,**opts)


if __name__ == "__main__":
    main()
