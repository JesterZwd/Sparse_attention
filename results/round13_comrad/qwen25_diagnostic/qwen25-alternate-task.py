"""Calibration-only Qwen2.5 quantization diagnostic, outside frozen results."""
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import torch

from gqa_probe import round13_models as generic
from gqa_probe.quantization_round2 import affine_quantize_dequantize
from scripts import run_round13 as runner

base = Path('.cache/round13-cloud/qwen25-alternate-task')
fwe_path = base / 'fwe'
panel = 'qwen25_32k_multivalue'
_, mv_manifest, _, mv_rows = runner.panel_data(runner.ROOT, panel, 'calibration')
from scripts.prepare_round13_data import load_panel
fwe_manifest, _, fwe_rows = load_panel(fwe_path, 'calibration')
assert len(fwe_rows) == 4 and mv_manifest['tokenizer'] == fwe_manifest['tokenizer']
model, tokenizer, eos, _ = runner.initialize(panel, mv_manifest)
geom = runner.geometry(model.config)
reference_cache = generic.PreallocatedAffineKV4Cache

def affine8_32(tensor):
    x = tensor.float().reshape(*tensor.shape[:-1], -1, 32)
    lo = x.amin(dim=-1, keepdim=True).clamp_max(0)
    hi = x.amax(dim=-1, keepdim=True).clamp_min(0)
    scale = ((hi-lo) / 255).clamp_min(2**-24).half().float()
    zero = (-lo/scale).round().clamp(0, 255)
    code = ((x/scale).round()+zero).clamp(0, 255)
    return ((code-zero)*scale).reshape_as(tensor).to(tensor.dtype)

class DiagnosticCache(generic.PreallocatedNativeCache):
    def __init__(self, capacity, mode):
        super().__init__(capacity)
        self.mode = mode

    def update(self, key, value, layer_idx, cache_kwargs=None):
        key = affine8_32(key) if self.mode == 'K8G32+V4G32' else affine_quantize_dequantize(key, 16)
        value = affine_quantize_dequantize(value, 32)
        return super().update(key, value, layer_idx, cache_kwargs)

def prefill(row, mode):
    if mode in ('BF16','KV4G32'):
        return runner.make_prefill(panel,model,row,16 if mode=='BF16' else 4)
    generic.PreallocatedAffineKV4Cache = lambda capacity: DiagnosticCache(capacity, mode)
    try:
        return runner.make_prefill(panel,model,row,4)
    finally:
        generic.PreallocatedAffineKV4Cache = reference_cache

trials = [
    ('FWE', fwe_rows, ('BF16','KV4G32','K4G16+V4G32','K8G32+V4G32')),
    ('Multi-Value', mv_rows[:2], ('K4G16+V4G32','K8G32+V4G32')),
]
records = []
with torch.inference_mode():
    for task, rows, modes in trials:
        for row in rows:
            for name in modes:
                state = prefill(row,name)
                observed = runner.evaluate(None,model,tokenizer,eos,state,row,None,geom,
                                           dense_bits=16 if name=='BF16' else 4)
                outcome = {field: observed[field] for field in ('official_score','complete_correct','generated_tokens','first_token','truncated')}
                record = dict(task=task,sample=row['sample_id'],mode=name,**outcome)
                records.append(record)
                print(json.dumps(record),flush=True)
                del state

def digest(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()

scores = defaultdict(list)
for row in records:
    scores[row['task'],row['mode']].append(row['official_score'])
summary=[dict(task=task,mode=mode,samples=len(values),score=mean(values)) for (task,mode),values in scores.items()]
assert len(records)==20
output={'summary':summary,'records':records,'fwe_split_sha256':digest(fwe_path/'calibration.jsonl'),
        'multi_value_receipt_sha256':digest(Path('results/round13_comrad/qwen25_32k_multivalue/calibration/config.json')),
        'source_sha256':digest(Path(__file__)),'model':model.config._name_or_path,'device':torch.cuda.get_device_name()}
(base/'results.json').write_text(json.dumps(output,indent=2)+'\n')
print('Summary',json.dumps(summary),flush=True)
