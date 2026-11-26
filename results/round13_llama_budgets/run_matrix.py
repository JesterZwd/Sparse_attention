"""Llama CoMRAD six-cell matrix with one sealed mask and per-prompt receipts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import traceback

import torch

from gqa_probe.compact_summary import ChannelCalibration
from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.round13_models import load_local_model, prefill_generic
from gqa_probe.ruler_inference import generate_from_prefill
from scripts import run_round13 as r13
from scripts.run_round10 import FitAttention, prompt_bundle, read_bundle, seal
from scripts.run_ruler_round3 import load_rows

ROOT = Path('results/round13_llama_budgets')
MODEL = Path('models/Llama-3.1-8B-Instruct')
REGISTERED_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
LENGTHS = ('32k','64k')
BUDGETS = (5,7,10)
POLICIES = tuple(replace(policy, budget=budget) for budget in BUDGETS for policy in r13.SPARSE)
MASK = ROOT/'channels.json'
CAL_ROUTES = ((None,16),(None,4))
TEST_ROUTES = ((None,4), *((policy,4) for policy in POLICIES))

def progress(state, **details):
    r13._write_json(ROOT/'status.json',dict(state=state,**details))

def receipt(data, split):
    manifest, limit, rows = load_rows(data,split)
    return manifest, limit, rows, dict(manifest_sha256=r13._sha(data/'manifest.json'),
        rows_sha256=manifest['splits'][split]['sha256'], sample_ids=[x['sample_id'] for x in rows],
        instance_sha256=[x['source_metadata']['instance_sha256'] for x in rows],
        record_path=str(data/f'{split}.jsonl'))

def verify_allocation():
    from gqa_probe.round11_workpoints import allocation
    from gqa_probe.fixed_k_v import ranked_pages
    from gqa_probe.page_read_budget import page_history
    for context in (32768,65536):
        history=page_history(context,32)
        ranks=ranked_pages(torch.arange(history.pages,dtype=torch.float32)[None,:], history)
        for budget in BUDGETS:
            quest, reference, candidate=[allocation(p,ranks,history) for p in POLICIES if p.budget==budget]
            assert quest.cap==reference.cap==candidate.cap==reference.dense_bytes*budget//100
            assert quest.ids.shape==reference.ids.shape
            assert candidate.ids.shape[1]<=reference.ids.shape[1]+4

def fit_mask(model, eos, geometry, model_receipt, fit, rows):
    plan = dict(geometry=geometry,model=model_receipt,fit=fit,decode_prefix_tokens=16,
                quantizer='affine32 KV4',channels=96)
    if MASK.is_file():
        saved=json.loads(MASK.read_text())
        if saved['plan']!=plan:
            raise ValueError('Saved Llama mask differs from fit registration')
        return {name:{int(k):torch.tensor(v,device='cuda') for k,v in layer.items()}
                for name, layer in saved['masks'].items()}
    cal=ChannelCalibration()
    for i,row in enumerate(rows,1):
        progress('fit',length='32k',current=i,total=len(rows))
        state=prefill_generic(model,row['input_ids'],max_new_tokens=row['max_new_tokens'],kv_bits=4)
        generate_from_prefill(model,state,min(16,row['max_new_tokens']),eos,FitAttention(cal))
        del state
        print(f'32k mask fit {i}/{len(rows)}',flush=True)
    masks=cal.freeze(96)['score']
    coverage=[]
    for layer,stats in cal.statistics.items():
        work=stats.score_abs
        coverage.extend((work.gather(-1,masks[layer]).sum(-1)/work.sum(-1).clamp_min(1e-12)).tolist())
    assert len(masks)==geometry['layers'] and len(coverage)==geometry['layers']*geometry['kv_heads']
    assert all(tuple(x.shape)==(geometry['kv_heads'],96) for x in masks.values())
    seal(MASK,dict(plan=plan,coverage_median=float(torch.tensor(coverage).quantile(.5)),
                   coverage_p05=float(torch.tensor(coverage).quantile(.05)),
                   masks={'mr96_score':{str(k):v.tolist() for k,v in masks.items()}}))
    return {'mr96_score':{k:v.to('cuda') for k,v in masks.items()}}

def route_key(policy,bits):
    return ((None,'dense_bf16' if bits==16 else 'dense_kv4') if policy is None else
            (policy.budget,policy.name))

def run_split(model,tokenizer,eos,geometry,masks,length,split,rows,routes,config):
    out=ROOT/length/split
    out.mkdir(parents=True,exist_ok=True)
    seal(out/'config.json',config)
    predictions, bundles=[],{}
    expected=[route_key(policy,bits) for policy,bits in routes]
    for i,row in enumerate(rows,1):
        path=out/'bundles'/f'{i-1:04d}.json.gz'
        if not path.is_file():
            records=[]
            state=None
            previous_bits=None
            for policy,bits in routes:
                budget,name=route_key(policy,bits)
                progress(split,length=length,current=i,total=len(rows),budget_percent=budget,policy=name)
                if previous_bits!=bits:
                    state=None
                    torch.cuda.reset_peak_memory_stats()
                    state=prefill_generic(model,row['input_ids'],max_new_tokens=row['max_new_tokens'],kv_bits=bits)
                    peak=dict(prefill_peak_allocated=torch.cuda.max_memory_allocated(),
                              prefill_peak_reserved=torch.cuda.max_memory_reserved())
                    previous_bits=bits
                simulated=r13.evaluate(policy,model,tokenizer,eos,state,row,masks,geometry,dense_bits=bits)
                if name=='Quest':
                    simulated['scorer_terms']=simulated['fused_updates']
                if policy is not None:
                    calls=simulated['kv_group_calls']
                    for key in ('cap_bytes','dense_comparator_bytes','qk_mac','pv_mac'):
                        simulated.setdefault(key,0)
                    nominal_cap=simulated['dense_comparator_bytes']*budget//100
                    assert 0 <= nominal_cap-simulated['cap_bytes'] < max(1,calls)
                records.append(dict(**simulated,**peak,budget_percent=budget))
                print(f'{length} {split} {i}/{len(rows)} {str(budget or "Dense")} {name} done',flush=True)
            del state
            prompt_bundle(path,records)
        current=read_bundle(path)
        if [(x['budget_percent'],x['policy']) for x in current]!=expected or any(
                x['sample_id']!=row['sample_id'] for x in current):
            raise ValueError(f'Invalid per-prompt receipt {path}')
        predictions.extend(current)
        bundles[path.name]=r13._sha(path)
        print(f'{length} {split} prompt {i}/{len(rows)} saved',flush=True)
    return out,predictions,bundles

def finish(out,predictions,bundles):
    r13._write_json(out/'predictions.json',predictions)
    r13._write_json(out/'complete.json',dict(config_sha256=r13._sha(out/'config.json'),
                    predictions=len(predictions),bundles=bundles))

@torch.inference_mode()
def main():
    verify_allocation()
    data={}
    all_instances=set()
    pilot_rows=load_rows(Path('.cache/round13-cloud/llama32k/data'),'calibration')[2]
    all_instances.update(x['source_metadata']['instance_sha256'] for x in pilot_rows)
    for length in LENGTHS:
        splits=('fit','calibration','test') if length=='32k' else ('calibration','test')
        for split in splits:
            m,context,rows,registered=receipt(ROOT/'data'/length,split)
            ids=registered['instance_sha256']
            assert len(rows)=={'fit':4,'calibration':16,'test':64}[split]
            assert len(ids)==len(set(ids)) and not set(ids)&all_instances
            all_instances.update(ids)
            data[length,split]=(m,context,rows,registered)
    m32,_,fit_rows,fit=data['32k','fit']
    assert all(v[0]['tokenizer']==m32['tokenizer'] for v in data.values())
    progress('load_model')
    model,tokenizer,eos,model_info=load_local_model(MODEL,m32)
    assert model_info['source_revision']==REGISTERED_REVISION
    geometry=r13.geometry(model.config)
    assert geometry['layers']==32 and geometry['kv_heads']==8 and geometry['q_per_kv']==4
    masks=fit_mask(model,eos,geometry,model_info,fit,fit_rows)
    sources={**r13.source_hashes(),**{name:r13._sha(Path(name)) for name in
        ('scripts/run_round13.py','gqa_probe/round13_models.py','gqa_probe/round13_comrad.py',__file__)}}
    shared=dict(model=model_info,geometry=geometry,mask_sha256=r13._sha(MASK),sources=sources,
                decoder='greedy',prefill_chunk=2048,quantizer='affine32 KV4',
                comparison_gates=r13.GATES,torch=str(torch.__version__),gpu=torch.cuda.get_device_name())

    gates={}
    for length in LENGTHS:
        _,context,rows,registered=data[length,'calibration']
        conf=dict(**shared,context=context,split='calibration',dataset=registered,routes=[list(route_key(p,b)) for p,b in CAL_ROUTES])
        out,results,bundles=run_split(model,tokenizer,eos,geometry,masks,length,'calibration',rows,CAL_ROUTES,conf)
        quality=paired_bootstrap(results,'dense_bf16','dense_kv4','official_score',method_key='policy',confidence=.95)
        quant=[x for x in results if x['policy']=='dense_kv4']
        gate=dict(quality=quality,truncated_kv4=sum(x['truncated'] for x in quant),
                  passed=quality['candidate_mean']>0)
        r13._write_json(out/'dense_quantization.json',gate)
        finish(out,results,bundles)
        gates[length]=gate
        print(f'{length} Dense gate BF16={quality["baseline_mean"]:.4f} KV4={quality["candidate_mean"]:.4f} passed={gate["passed"]}',flush=True)

    matrix=[]
    blocked=[]
    for length in LENGTHS:
        if not gates[length]['passed']:
            blocked.append(length)
            continue
        _,context,rows,registered=data[length,'test']
        pairs={}
        conf=dict(**shared,context=context,split='test',dataset=registered,
            dense_gate_sha256=r13._sha(ROOT/length/'calibration/dense_quantization.json'),
            policies=[asdict(p) for p in POLICIES],routes=[list(route_key(p,b)) for p,b in TEST_ROUTES])
        seal(ROOT/length/'frozen_test.json',conf)
        out,results,bundles=run_split(model,tokenizer,eos,geometry,masks,length,'test',rows,TEST_ROUTES,conf)
        for budget in BUDGETS:
            matching=[x for x in results if x['budget_percent'] in (None,budget)]
            table=r13.summarize(matching)
            matrix.extend(dict(length=length,budget_percent=budget,**x) for x in table)
            pairs[str(budget)]=r13.comparisons(matching,table)
        r13._write_json(out/'paired.json',pairs)
        r13.write_csv(out/'quality_cost.csv',[x for x in matrix if x['length']==length])
        finish(out,results,bundles)
        r13.write_csv(ROOT/'quality_cost.csv',matrix)
        print(f'{length} 5% / 7% / 10% independent paired results saved',flush=True)
    progress('incomplete' if blocked else 'complete',blocked_lengths=blocked,
             completed_cells=len(BUDGETS)*(len(LENGTHS)-len(blocked)))

if __name__=='__main__':
    try:
        main()
    except Exception:
        progress('error',error=traceback.format_exc())
        raise
