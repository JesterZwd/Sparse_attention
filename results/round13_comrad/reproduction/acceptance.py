"""Read-only Round 13 acceptance checks for the AutoDL artifacts."""
from collections import Counter
from hashlib import file_digest
from itertools import zip_longest
import csv
import gzip
import json
import math
from pathlib import Path

from scripts import run_round13 as runner

r = Path('results/round13_comrad')

def sha(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()

def doc(path):
    return json.loads(path.read_text())

def bundle(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)

def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

status = doc(Path('.cache/round13/queue_status.json'))
assert status['status'] == 'complete' and status['failures'] == {}
complete = doc(r/'panels_complete.json')
assert not complete['missing_panels']
fwe = doc(r/'task_selection.json')['chosen']
panels = ('qwen3_64k_multivalue', 'qwen25_32k_multivalue', fwe, 'qwen3_longbench_v2')
assert set(complete['panels']) == set(panels)
assert complete['replay_sha256'] == sha(r/'replay/replay_receipt.json')
splits = [(name, 'test') for name in panels] + [('qwen25_32k_multivalue','calibration'), (fwe,'calibration')]
sparse_groups = Counter()
total_bundles = total_predictions = 0
for panel, split in splits:
    path = r/panel/split
    receipt, config = (doc(path/name) for name in ('complete.json','config.json'))
    source, manifest, _, inputs = runner.panel_data(r, panel, split)
    assert sha(path/'config.json') == receipt['config_sha256']
    assert sha(source/'manifest.json') == config['dataset']['manifest_sha256']
    assert sha(source/f'{split}.jsonl') == manifest['splits'][split]['sha256']
    assert config['dataset']['sample_ids'] == [row['sample_id'] for row in inputs]
    assert config['dataset']['instance_sha256'] == [row['source_metadata']['instance_sha256'] for row in inputs]
    if split == 'test':
        assert len(inputs) == 64 and config == doc(path.parent/'frozen_test.json')
        assert complete['panels'][panel]['frozen_sha256'] == receipt['config_sha256']
    for file, expected_sha in config['sources'].items():
        assert sha(Path(file)) == expected_sha, file
    policy_order = ['dense_kv4','Quest','A','M']
    if config['quantization_ablation']:
        policy_order.insert(0,'dense_bf16')
    predictions = []
    paths = sorted((path/'bundles').glob('*.json.gz'))
    assert len(paths) == len(inputs) == len(receipt['bundles'])
    for source_path, row in zip(paths, inputs, strict=True):
        assert sha(source_path) == receipt['bundles'][source_path.name]
        entries = bundle(source_path)
        assert [entry['policy'] for entry in entries] == policy_order
        for entry in entries:
            assert entry['sample_id'] == row['sample_id']
            assert entry['generated_tokens'] == entry['decode_tokens'] + 1
            assert entry['kv_group_calls'] == entry['decode_tokens'] * config['geometry']['layers'] * config['geometry']['kv_heads']
            if not entry['policy'].startswith('dense'):
                assert entry['over_budget_group_steps'] == 0 and entry['read'] <= entry['cap_bytes']
                sparse_groups[split] += entry['kv_group_calls']
        predictions.extend(entries)
    assert len(predictions) == receipt['predictions'] and predictions == doc(path/'predictions.json')
    table = runner.summarize(predictions)
    assert runner.comparisons(predictions,table) == doc(path/'paired.json')
    stored = {entry['policy']:entry for entry in csv_rows(path/'quality_cost.csv')}
    for row in table:
        actual = stored[row['policy']]
        for key in ('official_score','complete','read_per_decode','arithmetic_per_decode','group_steps','overrun_groups'):
            assert math.isclose(float(actual[key]),float(row[key]),rel_tol=1e-12)
    total_bundles += len(paths)
    total_predictions += len(predictions)
    print(panel, split, len(paths), len(predictions), 'checks OK', flush=True)
    if panel == 'qwen3_longbench_v2':
        groups = {p:{v['sample_id']:v for v in predictions if v['policy']==p} for p in policy_order}
        correct = {k for k,v in groups['dense_kv4'].items() if v['complete_correct']}
        print('LongBench Dense-correct subset',len(correct),{p:sum(g[k]['complete_correct'] for k in correct) for p,g in groups.items()})

replay=r/'replay'
rr=doc(replay/'replay_receipt.json')
for name, expected in rr['files'].items():
    assert sha(replay/name)==expected, name
for rec in rr['inputs']:
    assert sha(Path(rec['file']))==rec['sha256'],rec['file']
for name, expected in rr['model_source_sha256'].items():
    assert sha(Path(name)) == expected, name
assert len(csv_rows(replay/'step_timing.csv'))==rr['step_timing_rows']
assert len(csv_rows(replay/'paired_timing.csv'))==rr['paired_timing_rows']
for length in ('32k','64k'):
    p=r/length/'common'
    capture=doc(p/'complete.json')
    archive=p/'states.jsonl.gz'
    assert sha(archive)==capture['states_sha256']
    assert len(capture['endpoints'])==len(capture['bundles'])==5
    expected_rows=[]
    for name, expected in sorted(capture['bundles'].items()):
        assert sha(p/'bundles'/name)==expected
        entry=bundle(p/'bundles'/name)
        rows=entry['rows']
        assert {x['policy'] for x in rows}=={'A','B','M'}
        assert len({(x['policy'],x['layer'],x['group']) for x in rows})==3*36*8
        assert all(x['whole_step'] and x['total_bytes'] <= x['cap_bytes'] for x in rows)
        expected_rows.extend(rows)
    with gzip.open(archive,'rt') as stream:
        assert all(json.loads(line)==row for line,row in zip_longest(stream,expected_rows))
    assert len(expected_rows)==capture['rows']==4320
print('QA totals',total_bundles,total_predictions,dict(sparse_groups))
print('Replay receipts, 10 capture bundles and 8640 new group-policy trace records OK')
