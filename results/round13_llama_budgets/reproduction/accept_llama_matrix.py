"""Read-only acceptance of the frozen Llama 32K/64K budget matrix."""
from collections import Counter
import csv
import gzip
from hashlib import file_digest
import json
from math import isclose
from pathlib import Path

from gqa_probe.paired_statistics import paired_bootstrap
from scripts import run_round13 as runner
from scripts.run_ruler_round3 import load_rows

ROOT=Path('results/round13_llama_budgets')
BUDGETS=(5,7,10)
SPLITS={'32k':('fit','calibration','test'),'64k':('calibration','test')}
POLICIES=('Quest','A','M')
MODEL=Path('models/Llama-3.1-8B-Instruct')

def load(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()

def table(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def compare_table(expected, saved):
    assert len(expected)==len(saved)
    for expected_row, saved_row in zip(expected,saved,strict=True):
        assert set(expected_row)==set(saved_row)
        for k,v in expected_row.items():
            observed=saved_row[k]
            if isinstance(v,(int,float)) and not isinstance(v,bool):
                assert isclose(v,float(observed),rel_tol=1e-12,abs_tol=1e-12), (k,v,observed)
            else:
                assert str(v)==observed, (k,v,observed)

status=load(ROOT/'status.json')
assert status==dict(state='complete',blocked_lengths=[],completed_cells=6)
source=load(MODEL/'source_manifest.json')
assert source['verified'] and source['revision']=='0e9e39f249a16976918f6564b8830bc894c89659'
weights={x['Path']:x['Sha256'] for x in source['files']}
assert sum(k.endswith('.safetensors') for k in weights)==4
for path, expected in weights.items():
    assert sha(MODEL/path)==expected, path
print('Pinned model assets match source SHA256',flush=True)

mask_path=ROOT/'channels.json'
mask=load(mask_path)
layers=mask['masks']['mr96_score']
assert set(layers)=={str(i) for i in range(32)}
assert all(len(layer)==8 for layer in layers.values())
assert all(len(head)==96 and len(set(head))==96 and min(head)>=0 and max(head)<128 for layer in layers.values() for head in layer)
mask_sha=sha(mask_path)
pilot=load_rows(Path('.cache/round13-cloud/llama32k/data'),'calibration')[2]
seen={r['source_metadata']['instance_sha256'] for r in pilot}
verified_sources=set()
all_summary=[]
sparse_steps=Counter()
counts=Counter()
for length,splits in SPLITS.items():
    data=ROOT/'data'/length
    for split in splits:
        manifest,context,registered=load_rows(data,split)
        instance=[r['source_metadata']['instance_sha256'] for r in registered]
        assert len(instance)==len(set(instance))==dict(fit=4,calibration=16,test=64)[split]
        assert not set(instance)&seen
        seen.update(instance)
        if split=='fit':
            fit=mask['plan']['fit']
            assert fit['rows_sha256']==manifest['splits'][split]['sha256']
            assert fit['instance_sha256']==instance
            continue
        out=ROOT/length/split
        config=load(out/'config.json')
        complete=load(out/'complete.json')
        assert sha(out/'config.json')==complete['config_sha256']
        assert config['mask_sha256']==mask_sha
        assert config['context']==context
        assert config['model']['source_revision']==source['revision']
        assert config['model']['files_sha256']==weights
        assert config['model']['source_manifest_sha256']==sha(MODEL/'source_manifest.json')
        assert config['geometry']['layers']==32 and config['geometry']['kv_heads']==8 and config['geometry']['q_per_kv']==4
        assert config['dataset']['manifest_sha256']==sha(data/'manifest.json')
        assert config['dataset']['rows_sha256']==manifest['splits'][split]['sha256']
        assert config['dataset']['sample_ids']==[r['sample_id'] for r in registered]
        assert config['dataset']['instance_sha256']==instance
        for name,digest in config['sources'].items():
            if (name,digest) not in verified_sources:
                assert sha(Path(name))==digest, name
                verified_sources.add((name,digest))
        if split=='test':
            assert config==load(ROOT/length/'frozen_test.json')
            route=[(None,'dense_kv4')]+[(b,p) for b in BUDGETS for p in POLICIES]
            assert [(p['budget'],p['name']) for p in config['policies']]==route[1:]
            assert all(p['page']==32 and p['v_reserve']==.80 and p['extra_pages']==(4 if p['name']=='M' else None) for p in config['policies'])
            assert config['dense_gate_sha256']==sha(ROOT/length/'calibration/dense_quantization.json')
        else:
            route=[(None,'dense_bf16'),(None,'dense_kv4')]
        assert [tuple(items) for items in config['routes']]==route
        bundles=sorted((out/'bundles').glob('*.json.gz'))
        assert len(bundles)==len(complete['bundles'])==len(registered)
        records=[]
        for path, prompt in zip(bundles,registered,strict=True):
            assert complete['bundles'][path.name]==sha(path), path
            with gzip.open(path,'rt') as f:
                pack=json.load(f)
            assert [(r['budget_percent'],r['policy']) for r in pack]==route
            for x in pack:
                assert x['sample_id']==prompt['sample_id']
                assert x['generated_tokens']==x['decode_tokens']+1
                assert x['kv_group_calls']==x['decode_tokens']*32*8
                budget=x['budget_percent']
                assert 0<=x['official_score']<=1
                if budget is not None:
                    assert x['over_budget_group_steps']==x.get('over_bytes',0)==0
                    assert x['read']<=x['cap_bytes']
                    assert x['read']==sum(x.get(k,0) for k in ('summary_bytes','k_bytes','v_bytes'))
                    assert 0<=x['dense_comparator_bytes']*budget//100-x['cap_bytes']<max(1,x['kv_group_calls'])
                    assert x['qk_mac']==x.get('k_records',0)*4*128
                    assert x['scorer_terms']==x['fused_updates']*(1 if x['policy']=='Quest' else 2)
                    sparse_steps[length,budget]+=x['kv_group_calls']
            records+=pack
        assert records==load(out/'predictions.json') and len(records)==complete['predictions']
        counts['bundles']+=len(bundles)
        counts['records']+=len(records)
        if split=='calibration':
            observed=load(out/'dense_quantization.json')
            paired=paired_bootstrap(records,'dense_bf16','dense_kv4','official_score',method_key='policy',confidence=.95)
            assert paired==observed['quality'] and observed['passed']
            print(length,split,len(bundles),len(records),'BF16',paired['baseline_mean'],'KV4',paired['candidate_mean'],flush=True)
            continue
        local=[]
        pairs=load(out/'paired.json')
        for b in BUDGETS:
            same=[x for x in records if x['budget_percent'] in (None,b)]
            summary=runner.summarize(same)
            local.extend(dict(length=length,budget_percent=b,**row) for row in summary)
            assert runner.comparisons(same,summary)==pairs[str(b)]
        compare_table(local,table(out/'quality_cost.csv'))
        all_summary+=local
        print(length,split,len(bundles),len(records),'groups',sum(v for (l,_),v in sparse_steps.items() if l==length),flush=True)
        m_only=[dict(x,route=f'M{x["budget_percent"]}') for x in records if x['policy']=='M']
        for lo,hi in ((5,7),(7,10),(5,10)):
            gain=paired_bootstrap(m_only,f'M{lo}',f'M{hi}','official_score',method_key='route',confidence=.95)
            print('M budget paired',length,lo,hi,*(round(gain[k]*100,3) for k in ('point_delta','ci_low','ci_high')),flush=True)
compare_table(all_summary,table(ROOT/'quality_cost.csv'))
print('All summary tables and paired intervals recomputed',flush=True)
print('Bundles and rows:',dict(counts),'Sparse groups:',sum(sparse_steps.values()),dict(sparse_steps),flush=True)
