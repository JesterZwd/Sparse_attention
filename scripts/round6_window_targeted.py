"""Replay the already selected window head locations on validation prompts."""
import json
from pathlib import Path

import numpy as np
import torch
import transformers

import gqa_probe.round6_audit as audit_module
from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.page_read_budget import page_history
from gqa_probe.round6_attention import Policy
from gqa_probe.ruler_experiment import _sha,_write_json
from scripts.report_round6 import summary_rows,write_csv
from scripts.round6_subpage_head import MODEL,ROOT,run_one
from scripts.run_ruler_round3 import load_rows


def main():
    p=ROOT/"report/screen/mechanism.json"
    s=json.loads(p.read_text())
    key="p95_3pct"
    remote=np.asarray(s["maps"][key])
    locations=[tuple(a) for a in np.argwhere(~remote).tolist()]
    targets={(l,g) for l,g,_ in locations}
    if not targets:
        raise ValueError("no head coordinates to replay")
    audit_module.SAMPLE_LAYERS=tuple(sorted({l for l,_ in targets}))
    _,_,rows=load_rows(ROOT/"data/calibration","calibration")
    indices=set(s["screen_ids"])
    rows=[r for r in rows if r["sample_id"] in indices]
    anchor=Policy(**json.loads((ROOT/"screen_plan.json").read_text())["anchor"])
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tok=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    e=model.generation_config.eos_token_id
    eos=set(e if isinstance(e,list) else [e])
    all_rows={"base":[],"head":[],"group":[]}
    error_heads={tag:[] for tag in locations}
    for row in rows:
        b=run_one(model,tok,eos,row,[Policy("dense")],"screen",True,anchor,{})
        for r in b["locality"]:
            for tag in locations:
                l,g,h=tag
                if r["layer"]==l:
                    error_heads[tag].append(r["relative"][g][h])
        for r in b["common"]:
            l,g=r["layer"],r["group"]
            if (l,g) not in targets:
                continue
            meta=dict(sample_id=row["sample_id"],layer=l,group=g,step=r["step"])
            all_rows["base"].append(dict(**meta,**r["base"]))
            h=page_history(r["tokens"],32)
            for kind in ("head","group"):
                x=audit_module.fixed_k_window(h,r,remote[l,g],kind)
                all_rows[kind].append(dict(**meta,**x))
    stats=summary_rows(all_rows)
    out=ROOT/"report/screen"
    write_csv(out/"window_targeted_per_group.csv",[dict(variant=k,**row) for k,rs in all_rows.items() for row in rs])
    _write_json(out/"window_targeted.json",dict(mechanism_sha256=_sha(p),map_id=key,locations=locations,
                quality_against_dense=[dict(layer=l,group=g,head=h,relative_p95=float(np.quantile(err,.95)),
                                            observations=len(err)) for (l,g,h),err in error_heads.items()],
                context="same candidates; sampled only at selected layer/group positions; no new fitted mask",
                summary=stats))
    d=json.loads((ROOT/"report/compact/predictions.json").read_text())
    reports={mode:paired_bootstrap(d,mode,"compact_mr25","official_score",
                            confidence=.95,method_key="policy") for mode in ("compact_q4","compact_q6")}
    _write_json(ROOT/"report/compact/paired.json",reports)
    print("targeted head locations:",locations,"common group states:",len(all_rows["base"]))


if __name__=="__main__":
    main()
