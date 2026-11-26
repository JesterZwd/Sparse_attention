"""Small Llama Dense BF16/affine32 KV4 calibration gate."""
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import torch

from gqa_probe.round13_models import load_local_model, prefill_generic
from scripts import run_round13 as runner
from scripts.check_round13_model import short_parity
from scripts.prepare_round13_data import load_panel

root=Path('.cache/round13-cloud/llama32k')
dataset=root/'data'
manifest,_,rows=load_panel(dataset,'calibration')
assert len(rows)==4
model,tokenizer,eos,source=load_local_model(Path('models/Llama-3.1-8B-Instruct'),manifest)
geom=runner.geometry(model.config)
assert geom['q_per_kv']==4 and geom['head_dim']==128

with torch.inference_mode():
    parity=short_parity(model,tokenizer,eos)
    print('Short-context parity',json.dumps(parity),flush=True)
    records=[]
    for row in rows:
        for bits in (16,4):
            torch.cuda.reset_peak_memory_stats()
            state=prefill_generic(model,row['input_ids'],max_new_tokens=row['max_new_tokens'],kv_bits=bits)
            peak=torch.cuda.max_memory_reserved()
            observed=runner.evaluate(None,model,tokenizer,eos,state,row,None,geom,dense_bits=bits)
            record=dict(bits=bits,sample=row['sample_id'],input_tokens=row['input_tokens'],peak_prefill_reserved=peak,
                        **{field:observed[field] for field in ('official_score','complete_correct','generated_tokens','first_token','truncated')})
            records.append(record)
            print(json.dumps(record),flush=True)
            del state
def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()
summary={str(bits):mean(v['official_score'] for v in records if v['bits']==bits) for bits in (16,4)}
result=dict(summary=summary,records=records,parity=parity,model=source,geometry=geom,
            data_rows_sha256=digest(dataset/'calibration.jsonl'))
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('Llama mean official scores',json.dumps(summary),flush=True)
