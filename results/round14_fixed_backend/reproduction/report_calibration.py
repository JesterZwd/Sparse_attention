"""Build paired calibration tables from fully saved per-prompt bundles."""

from argparse import ArgumentParser
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROUTES = ('reference', 'uint8', 'uint16')
PANELS = ('qwen3_32k', 'qwen3_32k_mv', 'qwen3_64k', 'llama_32k')
MATCH = ('panel', 'split', 'context', 'samples', 'manifest_sha256', 'rows_sha256', 'model', 'geometry',
         'mask_sha256', 'policy', 'runner_sha256')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_route(root, panel, route):
    source = root / panel / route
    path = source / 'predictions.json'
    if not path.is_file():
        return None
    config, predictions = load(source / 'config.json'), load(path)
    if [r['sample_id'] for r in predictions] != config['samples'] or config['split'] != 'calibration':
        raise ValueError(f'bad route data: {source}')
    if len(config['samples']) != len(set(config['samples'])):
        raise ValueError(f'duplicate prompt IDs: {source}')
    return config, predictions, sha256(path.read_bytes()).hexdigest()


def compare(root, require_all=False):
    rng = np.random.default_rng(20261126)
    summary, configs = [], []
    for panel in PANELS:
        base = read_route(root, panel, 'reference')
        if base is None:
            if require_all:
                raise FileNotFoundError(f'{panel} reference incomplete')
            continue
        base_config, base_predictions, _ = base
        baseline = np.array([r['official_score'] for r in base_predictions])
        resample = rng.integers(len(baseline), size=(10_000, len(baseline)))
        for route in ROUTES:
            result = read_route(root, panel, route)
            if result is None:
                if require_all:
                    raise FileNotFoundError(f'{panel} {route} incomplete')
                continue
            config, pred, digest = result
            if any(config[name] != base_config[name] for name in MATCH):
                raise ValueError(f'{panel} {route} does not match its paired reference config')
            score = np.array([r['official_score'] for r in pred])
            delta = score - baseline
            low, high = np.quantile(delta[resample].mean(1), [.025, .975])
            groups = sum(r['kv_group_calls'] for r in pred)
            step_count = sum(r['decode_tokens'] for r in pred)
            overruns = sum(r.get('over_budget_group_steps', 0) for r in pred)
            saturated = sum(r.get('logit_saturations', 0) for r in pred)
            summary.append(dict(panel=panel, task=pred[0]['task'], route=route, prompts=len(pred),
                                official_score_pct=100 * float(score.mean()), delta_pp=100 * float(delta.mean()),
                                paired_ci_lo_pp=100 * float(low), paired_ci_hi_pp=100 * float(high),
                                complete_correct=sum(r['complete_correct'] for r in pred),
                                mean_generated_tokens=sum(r['generated_tokens'] for r in pred) / len(pred),
                                mb_per_decode=sum(r['read'] for r in pred) / max(step_count, 1) / 1e6,
                                group_steps=groups, fallback_group_steps=sum(r.get('fallback_group_steps', 0) for r in pred),
                                over_budget_group_steps=overruns, logit_saturations=saturated,
                                calibration_gate_passes=(route != 'reference' and not overruns and not saturated and
                                                         float(delta.mean()) >= -.02 - 1e-12)))
            configs.append(dict(panel=panel, route=route, predictions_sha256=digest, config=config))
    return summary, configs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    rows, configs = compare(args.root, not args.allow_incomplete)
    if not rows:
        raise SystemExit('no complete prediction bundles')
    target = args.root.parent
    with (target / 'calibration_summary.csv').open('w', newline='', encoding='utf-8') as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (target / 'calibration_receipt.json').write_text(json.dumps(configs, indent=2) + '\n', encoding='utf-8')
    print(f'{len(rows)} result rows verified')


if __name__ == '__main__':
    main()
