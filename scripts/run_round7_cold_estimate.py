"""Use estimated cold mass only if the exact-mass tail screen improved."""
import argparse
import json
from pathlib import Path

import torch
import transformers

from gqa_probe.cold_page_compensation import CompensatedAttention, OnlineMeans
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.vskip_summary import StaticPrefix
from scripts.round7_report import pairs
from scripts.round7_run import DATA, MODEL, ROOT, STATIC, make_answer
from scripts.run_ruler_round3 import load_rows


@torch.inference_mode()
def run(root,data,device,threads):
    screen=root/"report"/"screen"/"side_summary.json"
    quality=json.loads(screen.read_text())["cold_compensation"]
    out=root/"report"/"cold_estimate"
    out.mkdir(exist_ok=True)
    reference=dict(screen_sha256=_sha(screen),candidate_uses_calibration_only=True)
    if not quality or quality["cold_oracle_abs_p99"]>=quality["hot_abs_p99"]:
        _write_json(out/"decision.json",dict(estimate_ran=False,**reference))
        return
    manifest,_,rows=load_rows(data/"calibration","calibration")
    rows=rows[:4]
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,hashes=_verified_model(MODEL,manifest)
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map=device,
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=model.generation_config.eos_token_id
    eos=set(ids if isinstance(ids,list) else [ids])
    predictions=[]
    for row in rows:
        state=prefill_for_quantization(model,row["input_ids"],"affine32")
        prefix=tuple(StaticPrefix.build(k,32,("q6",)) for k in state.cache.key_cache)
        means=tuple(OnlineMeans.build(k,v) for k,v in zip(state.cache.key_cache,state.cache.value_cache))
        attn=CompensatedAttention(prefix,means,row["gold_spans"])
        generated=generate_from_prefill(model,state,row["max_new_tokens"],eos,attn)
        p=make_answer(attn.policy,generated,tokenizer,row,eos)
        if p["over_budget_group_steps"]:
            raise ValueError("cold-compensation budget violation")
        predictions.append(dict(sample_id=row["sample_id"],task=row["task"],**p))
        print(row["sample_id"],flush=True)
    controls=json.loads((root/"report"/"baseline"/"predictions.json").read_text())
    ids={r["sample_id"] for r in predictions}
    static=[r for r in controls if r["policy"]==STATIC.name and r["sample_id"] in ids]
    delta=pairs(out/"paired.json",static+predictions,STATIC.name,"cold_mean_estimate")
    _write_json(out/"predictions.json",predictions)
    _write_json(out/"decision.json",dict(estimate_ran=True,n=len(predictions),model_sha256=hashes,
                solid_quality_gain=delta["official_score"]["ci_low"]>0,
                code={str(p):_sha(p) for p in (Path(__file__),Path("gqa_probe/cold_page_compensation.py"))},
                **reference))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    run(args.root,args.data,args.device,args.threads)
