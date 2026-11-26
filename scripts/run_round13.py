"""Official-task coeval of Dense, Quest INT4 + 64-bin, 144 B A, and 112 B CoMRAD M."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import torch

from gqa_probe.compact_summary import ChannelCalibration
from gqa_probe.paired_statistics import paired_bootstrap, score_prediction
from gqa_probe.round7_attention import Attention
from gqa_probe.round13_comrad import GATES, QUEST, SPARSE, compare_gate, fwe_insufficient_disagreement
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DENSE
from scripts.run_round10 import ROOT as R10
from scripts.run_round10 import (FitAttention, init_model, prefill, prefix_layers, prompt_bundle,
                                 read_bundle, read_masks, seal, summarize_predictions)
from scripts.run_round11 import WorkpointAttention, gold_spans, read_json, source_hashes, split_inventory
from scripts.run_ruler_round3 import load_rows

ROOT = Path("results/round13_comrad")
R12_DATA = Path("results/round12_frontend/data/64k")
QWEN25 = Path("models/Qwen2.5-7B-Instruct")
LABELS = {"dense_kv4": "Dense KV4", "dense_bf16": "Dense BF16 KV",
          "Quest": "Quest INT4 + 64-bin（SPARTA 式算法对照）", "A": "144 B A", "M": "112 B CoMRAD M"}


def panel_data(root, panel, split):
    dataset = R12_DATA if panel == "qwen3_64k_multivalue" else root / "data" / panel
    if panel.endswith("longbench_v2"):
        from scripts.prepare_round13_data import load_panel

        manifest, context, rows = load_panel(dataset, split)
    else:
        manifest, context, rows = load_rows(dataset, split)
    return dataset, manifest, context, rows


def initialize(panel, manifest):
    if panel.startswith("qwen25"):
        from gqa_probe.round13_models import load_local_model

        return load_local_model(QWEN25, manifest)
    return init_model(manifest)


def make_prefill(panel, model, row, bits):
    if panel.startswith("qwen25"):
        from gqa_probe.round13_models import prefill_generic

        return prefill_generic(model, row["input_ids"], max_new_tokens=row["max_new_tokens"], kv_bits=bits)
    if bits != 4:
        raise ValueError("Qwen3 primary comparison uses affine32")
    return prefill(model, row)


def geometry(config):
    heads = config.num_attention_heads
    kv = config.num_key_value_heads
    dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    if dim != 128 or heads % kv:
        raise ValueError("unsupported attention geometry for affine32/page summaries")
    return dict(model_type=config.model_type, layers=config.num_hidden_layers, query_heads=heads,
                kv_heads=kv, head_dim=dim, q_per_kv=heads//kv, rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling, max_positions=config.max_position_embeddings,
                qk_norm=config.model_type == "qwen3")


def evaluate(policy, model, tokenizer, eos, state, row, masks, geom, *, dense_bits=4):
    spans = gold_spans(row)
    if policy is None:
        name = "dense_kv4" if dense_bits == 4 else "dense_bf16"
        attention = Attention(DENSE, None, spans)
    else:
        name = policy.name
        attention = WorkpointAttention(policy, prefix_layers(state, policy, masks), masks, spans)
    run = generate_from_prefill(model, state, row["max_new_tokens"], eos, attention)
    prediction = tokenizer.decode(run.token_ids, skip_special_tokens=True)
    cleaned = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
    info = {}
    if row["task"] == "longbench_v2":
        from scripts.prepare_round13_data import extract_answer

        option = extract_answer(cleaned)
        correct = option == row["references"][0]
        score = dict(official_score=float(correct), complete_correct=correct)
        info.update(answer_choice=option, domain=row["source_metadata"]["domain"])
    else:
        score = score_prediction(cleaned, row["references"])
    stats = run.attention_stats
    steps = len(run.token_ids)-1
    if stats["kv_group_calls"] != steps*geom["layers"]*geom["kv_heads"]:
        raise ValueError("incomplete group-step ledger")
    if policy is not None and (stats.get("over_budget_group_steps",0) or stats["read"] > stats.get("cap_bytes",0)):
        raise ValueError("physical budget violation")
    if policy == QUEST:
        # One signed-extremum term is active per Query channel. The two scale
        # buckets in the numerical reference share this physical dot count.
        stats["scorer_terms"] = stats.get("fused_updates",0)
    if policy is None:
        records = stats.get("qk_mac",0) // (geom["q_per_kv"]*geom["head_dim"])
        stats.update(k_records=records, v_records=records)
        if dense_bits == 16:
            stats["k_bytes"] = stats["v_bytes"] = records*geom["head_dim"]*2
            stats["read"] = stats["k_bytes"]+stats["v_bytes"]
    for key in ("scorer_terms","fused_updates","fallback_group_steps","over_budget_group_steps"):
        stats.setdefault(key,0)
    raw_hash = hashlib.sha256(json.dumps(run.token_ids,separators=(",",":")).encode()).hexdigest()
    return dict(policy=name, task=row["task"], sample_id=row["sample_id"], **score, **info,
                decode_tokens=steps, generated_tokens=len(run.token_ids), output_ids_sha256=raw_hash,
                first_token=run.token_ids[0], truncated=run.token_ids[-1] not in eos,
                read_per_decode=stats["read"]/max(steps,1), numeric_seconds=run.elapsed_seconds, **stats)


def summarize(predictions):
    rows = summarize_predictions(predictions)
    for cost in rows:
        data = [r for r in predictions if r["policy"] == cost["policy"]]
        steps = max(1, sum(x["decode_tokens"] for x in data))
        fused = sum(x["fused_updates"] for x in data)/steps
        cost.update(label=LABELS[cost["policy"]], fused_updates_per_decode=fused,
                    arithmetic_per_decode=fused+cost["qk_per_decode"]+cost["pv_per_decode"],
                    generated_tokens_per_answer=sum(x["generated_tokens"] for x in data)/len(data),
                    truncated_rate=cost["truncated"]/len(data))
    return rows


def comparisons(predictions, stats):
    cost = {r["policy"]:r for r in stats}
    pairs = []
    for name in ("A","Quest","dense_kv4"):
        if name not in cost or "M" not in cost:
            continue
        metrics = ("official_score","complete_correct","read","read_per_decode","generated_tokens",
                   "qk_mac","pv_mac","scorer_terms","fused_updates")
        intervals = {m: paired_bootstrap(predictions, name, "M", m, method_key="policy", confidence=.95)
                     for m in metrics}
        score = intervals["official_score"]
        base = {r["sample_id"]:r for r in predictions if r["policy"]==name}
        deltas = [dict(sample_id=r["sample_id"], score_delta=r["official_score"]-base[r["sample_id"]]["official_score"],
                       bytes_delta=r["read"]-base[r["sample_id"]]["read"],
                       complete_candidate=r["complete_correct"], complete_reference=base[r["sample_id"]]["complete_correct"])
                  for r in predictions if r["policy"]=="M"]
        gate = compare_gate(cost[name],cost["M"],score["point_delta"],score["ci_low"]) if name in ("A","Quest") else None
        pairs.append(dict(reference=name, candidate="M", gate=gate, intervals=intervals, prompt_deltas=deltas))
    return pairs


def masks_for(root, panel):
    if panel.startswith("qwen25"):
        path = root / "model_adaptation" / "qwen25_channels.json"
        saved = read_json(path)
        masks = {name:{int(k):torch.tensor(v,device="cuda") for k,v in layers.items()}
                 for name,layers in saved["masks"].items()}
    else:
        path = R10/"channels.json"
        masks = read_masks(R10,"cuda")
    return path, masks


def run_panel(root, panel, split, *, quantization=False, limit=None):
    data, manifest, context, rows = panel_data(root,panel,split)
    rows = rows[:limit]
    model, tokenizer, eos, model_files = initialize(panel,manifest)
    mask_path, masks = masks_for(root,panel)
    geom = geometry(model.config)
    if any(row["input_tokens"]+row["max_new_tokens"]>geom["max_positions"] for row in rows):
        raise ValueError("final input and generation exceed configured native context")
    extra_sources = ["scripts/run_round13.py","gqa_probe/round13_comrad.py"]
    if panel.startswith("qwen25"):
        extra_sources.append("gqa_probe/round13_models.py")
    if panel.endswith("longbench_v2"):
        extra_sources.append("scripts/prepare_round13_data.py")
    config = dict(panel=panel, split=split, context=context,
                  sources={**source_hashes(),**{p:_sha(Path(p)) for p in extra_sources}},
                  dataset=dict(manifest_sha256=_sha(data/"manifest.json"),
                               sample_ids=[r["sample_id"] for r in rows],
                               instance_sha256=[r["source_metadata"]["instance_sha256"] for r in rows]),
                  model=model_files, geometry=geom, channel_mask_sha256=_sha(mask_path),
                  policies=[asdict(p) for p in SPARSE], decoder="greedy", prefill_chunk=2048,
                  gates=GATES, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  quantization_ablation=quantization,
                  quest_source="https://arxiv.org/abs/2610.34824")
    output=root/panel/split
    output.mkdir(parents=True,exist_ok=True)
    if split=="test":
        seal(root/panel/"frozen_test.json",config)
    seal(output/"config.json",config)
    predictions=[]; receipts={}
    for i,row in enumerate(rows):
        bundle=output/"bundles"/f"{i:04d}.json.gz"
        if not bundle.is_file():
            prompt=[]
            if quantization:
                state=make_prefill(panel,model,row,16)
                prompt.append(evaluate(None,model,tokenizer,eos,state,row,masks,geom,dense_bits=16))
                del state
            torch.cuda.reset_peak_memory_stats()
            state=make_prefill(panel,model,row,4)
            stats=dict(prefill_peak_allocated=torch.cuda.max_memory_allocated(),
                       prefill_peak_reserved=torch.cuda.max_memory_reserved())
            for p in (None,*SPARSE):
                result=evaluate(p,model,tokenizer,eos,state,row,masks,geom)
                prompt.append(dict(**result,**stats))
            del state
            prompt_bundle(bundle,prompt)
        current=read_bundle(bundle)
        expected=(["dense_bf16"] if quantization else [])+["dense_kv4","Quest","A","M"]
        if [r["policy"] for r in current]!=expected or any(r["sample_id"]!=row["sample_id"] for r in current):
            raise ValueError("prompt bundle differs from frozen protocol")
        predictions+=current
        receipts[bundle.name]=_sha(bundle)
        print(f"{panel} {split} {i+1}/{len(rows)}",flush=True)
    summary=summarize(predictions)
    _write_json(output/"predictions.json",predictions)
    write_csv(output/"quality_cost.csv",summary)
    _write_json(output/"paired.json",comparisons(predictions,summary))
    if quantization:
        bf={r["sample_id"]:r for r in predictions if r["policy"]=="dense_bf16"}
        kv={r["sample_id"]:r for r in predictions if r["policy"]=="dense_kv4"}
        _write_json(output/"quantization.json",dict(
            paired=paired_bootstrap(predictions,"dense_bf16","dense_kv4","official_score",method_key="policy"),
            first_tokens_equal=sum(r["first_token"]==kv[i]["first_token"] for i,r in bf.items()),
            prompts=len(bf), truncated_bf16=sum(r["truncated"] for r in bf.values()),
            truncated_kv4=sum(r["truncated"] for r in kv.values())))
    _write_json(output/"complete.json",dict(config_sha256=_sha(output/"config.json"),
                                           predictions=len(predictions),bundles=receipts))
    return predictions


@torch.inference_mode()
def fit_qwen25(root):
    panel = "qwen25_32k_multivalue"
    data, manifest, _, rows = panel_data(root,panel,"fit")
    if len(rows)!=4:
        raise ValueError("Qwen2.5 mask fit must have four prompts")
    model, _, eos, hashes = initialize(panel,manifest)
    geom=geometry(model.config)
    cal=ChannelCalibration()
    for row in rows:
        prefix=make_prefill(panel,model,row,4)
        generate_from_prefill(model,prefix,min(16,row["max_new_tokens"]),eos,FitAttention(cal))
        del prefix
    frozen=cal.freeze(96)["score"]
    coverage=[]
    for layer,stats in cal.statistics.items():
        part=stats.score_abs
        coverage.extend((part.gather(-1,frozen[layer]).sum(-1)/part.sum(-1).clamp_min(1e-12)).tolist())
    if len(frozen)!=geom["layers"] or len(coverage)!=geom["layers"]*geom["kv_heads"]:
        raise ValueError("incomplete learned channel mask")
    out=root/"model_adaptation"
    out.mkdir(parents=True,exist_ok=True)
    seal(out/"qwen25_channels.json",dict(geometry=geom, model=hashes,
        fit=split_inventory(data,"fit")[0], coverage_p05=float(torch.tensor(coverage).quantile(.05)),
        coverage_min=min(coverage),coverage_median=float(torch.tensor(coverage).quantile(.5)),
        masks={"mr96_score":{str(layer):ids.tolist() for layer,ids in frozen.items()}}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("qwen3_64k","qwen25_fit","qwen25_cal","qwen25_test","fwe_screen",
                                   "fwe_test","longbench"))
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    root=a.root
    with torch.inference_mode():
        if a.stage=="qwen3_64k":
            from scripts.prepare_round13_data import verify_rebuilt_manifest

            seal_path=root/"qwen3_64k_seal.json"
            evidence=read_json(seal_path)
            manifest_path = R12_DATA / "manifest.json"
            if _sha(manifest_path) != evidence["source_manifest_sha256"]:
                verify_rebuilt_manifest(manifest_path, root/"inputs/round12_64k_manifest.json",
                                        evidence["source_manifest_sha256"])
            _, _, rows=load_rows(R12_DATA,"test")
            if (evidence["source_rows_sha256"]!=_sha(R12_DATA/"test.jsonl") or
                    not evidence["zero_historical_instance_overlap"] or len(rows)!=64 or
                    set(evidence["instance_sha256"])!={r["source_metadata"]["instance_sha256"] for r in rows}):
                raise ValueError("64K test data differs from unopened sample seal")
            run_panel(root,"qwen3_64k_multivalue","test")
        elif a.stage=="qwen25_fit":
            fit_qwen25(root)
        elif a.stage in ("qwen25_cal","qwen25_test"):
            run_panel(root,"qwen25_32k_multivalue","calibration" if a.stage.endswith("cal") else "test",
                      quantization=a.stage.endswith("cal"))
        elif a.stage=="fwe_screen":
            pred=run_panel(root,"qwen3_32k_fwe","calibration")
            switch, different = fwe_insufficient_disagreement(pred)
            _write_json(root/"task_selection.json",dict(chosen="qwen3_32k_cwe" if switch else "qwen3_32k_fwe",
                                                        sparse_different_prompts=different))
        elif a.stage=="fwe_test":
            chosen=read_json(root/"task_selection.json")["chosen"]
            run_panel(root,chosen,"test")
        else:
            run_panel(root,"qwen3_longbench_v2","test")


if __name__=="__main__":
    main()
