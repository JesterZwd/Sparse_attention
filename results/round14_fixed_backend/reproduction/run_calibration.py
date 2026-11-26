"""Paired CoMRAD backend calibration on registered calibration prompts only."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import re

import torch

from gqa_probe.paired_statistics import score_prediction
from gqa_probe.round13_comrad import M
from gqa_probe.round13_models import load_local_model, prefill_generic
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from scripts import run_round13 as r13
from scripts.run_round10 import prefix_layers, prompt_bundle, read_bundle, seal
from scripts.run_ruler_round3 import load_rows

RUN = Path('.cache/comrad-round14')
LLAMA = Path('results/round13_llama_budgets')
PANELS = {
    'qwen3_32k': Path('results/round13_comrad/data/qwen3_32k_fwe'),
    'qwen3_64k': Path('.cache/round13-cloud/old_64k_calibration'),
    'llama_32k': LLAMA / 'data/32k',
}


def model_setup(panel, manifest):
    if panel != 'llama_32k':
        model, tokenizer, eos, files = r13.initialize(panel, manifest)
        mask_path, masks = r13.masks_for(r13.ROOT, panel)
        return model, tokenizer, eos, files, mask_path, masks
    model, tokenizer, eos, files = load_local_model('models/Llama-3.1-8B-Instruct', manifest)
    mask_path = LLAMA / 'channels.json'
    saved = r13.read_json(mask_path)
    masks = {'mr96_score': {int(k): torch.tensor(v, device='cuda') for k, v in saved['masks']['mr96_score'].items()}}
    return model, tokenizer, eos, files, mask_path, masks


def report_status(path, message):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(message + '\n', encoding='utf-8')
    tmp.replace(path)
    print(message, flush=True)


@torch.inference_mode()
def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('route', choices=('reference', 'uint8', 'uint16'))
    parser.add_argument('panel', choices=PANELS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    panel, route = args.panel, args.route
    data = PANELS[panel]
    manifest, context, rows = load_rows(data, 'calibration')
    if args.limit is not None:
        rows = rows[:args.limit]
    assert rows
    output = RUN / 'output' / panel / route
    status = RUN / f'{route}.status'
    output.mkdir(parents=True, exist_ok=True)
    report_status(status, f'{panel} {route} loading model ({len(rows)} prompts)')
    model, tokenizer, eos, files, mask_path, masks = model_setup(panel, manifest)
    geom = r13.geometry(model.config)
    backend = None
    if route != 'reference':
        import fixed_backend as backend

        backend.install_cache()
    seal(output / 'config.json', dict(
        panel=panel, split='calibration', route=route, context=context, samples=[r['sample_id'] for r in rows],
        manifest_sha256=_sha(data / 'manifest.json'), rows_sha256=_sha(data / 'calibration.jsonl'),
        model=files, geometry=geom, mask_sha256=_sha(mask_path), policy=asdict(M),
        runner_sha256=_sha(Path(__file__)), backend_sha256=None if backend is None else _sha(Path(backend.__file__))))
    predictions = []
    for i, row in enumerate(rows, 1):
        bundle = output / 'bundles' / f'{i:04d}.json.gz'
        if not bundle.is_file():
            report_status(status, f'{panel} {route} {i}/{len(rows)} prefill')
            state = (prefill_generic(model, row['input_ids'], max_new_tokens=row['max_new_tokens'])
                     if panel == 'llama_32k' else r13.make_prefill(panel, model, row, 4))
            report_status(status, f'{panel} {route} {i}/{len(rows)} decode')
            if backend is None:
                result = r13.evaluate(M, model, tokenizer, eos, state, row, masks, geom)
            else:
                attention = backend.FixedAttention(
                    M, prefix_layers(state, M, masks), masks, r13.gold_spans(row),
                    cache=state.cache, bits=int(route[4:]), fixture_dir=output / 'vectors' if i <= 3 else None,
                    prompt=i, layers=geom['layers'])
                run = generate_from_prefill(model, state, row['max_new_tokens'], eos, attention)
                answer = tokenizer.decode(run.token_ids, skip_special_tokens=True)
                answer = re.sub(r'[\x00-\x1f]', '\n', answer.strip()).strip()
                stats = run.attention_stats
                steps = len(run.token_ids) - 1
                if stats.get('kv_group_calls', 0) != steps * geom['layers'] * geom['kv_heads']:
                    raise ValueError('incomplete backend group-step ledger')
                if stats.get('over_budget_group_steps', 0) or stats.get('read', 0) > stats.get('cap_bytes', 0):
                    raise ValueError('physical budget violation')
                result = dict(sample_id=row['sample_id'], policy=M.name, task=row['task'],
                              **score_prediction(answer, row['references']), generated_tokens=len(run.token_ids),
                              output_ids=run.token_ids, decode_tokens=steps, numeric_seconds=run.elapsed_seconds, **stats)
            prompt_bundle(bundle, result)
            del state
        item = read_bundle(bundle)
        if item['sample_id'] != row['sample_id']:
            raise ValueError('incorrect per-prompt receipt')
        predictions.append(item)
        report_status(status, f'{panel} {route} {i}/{len(rows)} saved')
    _write_json(output / 'predictions.json', predictions)
    report_status(status, f'{panel} {route} DONE {len(rows)}/{len(rows)}')


if __name__ == '__main__':
    main()
