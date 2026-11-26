"""Replay published A/M whole steps with common clocks and unchanged memory-side CDC."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
from statistics import fmean

from scripts import replay_round12 as replay
from scripts.replay_round13_comrad import R11, R12, R13, restore_round12_trace

ROOT = Path('.cache/comrad-round14')
PERIODS = (1.25, 1.80, 2.00)


def projected(timing, period):
    return replace(timing, clock_period_ns=period, result_cdc_pe_cycles=0)


def worker(task):
    period, payload = task
    replay.PRIMARY_PE_NS = period
    return replay.work(payload)


def p95(values):
    ordered = sorted(values)
    return ordered[ceil(.95 * len(ordered)) - 1]


def main():
    root = ROOT / 'clock'
    root.mkdir(parents=True, exist_ok=True)
    saved32 = R12 / '32k/extended/states.jsonl.gz'
    if not saved32.exists():
        saved32 = root / 'round12-32k-states.jsonl.gz'
    spec = json.loads((R12 / 'verification.json').read_text())['publication']
    restore_round12_trace(saved32, spec, R12)
    state_sets = {
        'archived_round11': [R11 / n / 'common/states.jsonl.gz' for n in ('32k', '64k')],
        'expanded_round12': [saved32, R12 / '64k/extended/states.jsonl.gz'],
        'new_round13': [R13 / n / 'common/states.jsonl.gz' for n in ('32k', '64k')],
    }
    path = R13 / 'hardware/frontend_calibration.json'
    conditions, _ = replay.read_conditions(path)
    previous = next(c for c in conditions if c.name == 'full_match_s4_c3')
    scenarios = [(1.25, previous)]
    for period in PERIODS[1:]:
        settings = {width: (stride, projected(timing, period)) for width, (stride, timing) in previous.settings.items()}
        scenarios.append((period, replace(previous, name=f'common_{period:.2f}ns', settings=settings)))
    tasks = []
    source_receipts = []
    for name, paths in state_sets.items():
        for source in paths:
            digest = replay.digest(source)
            if source == saved32:
                expected = spec['reassembled_sha256']
            else:
                receipt = json.loads((source.parent / 'complete.json').read_text())
                expected = receipt.get('state_sha256') or receipt['states_sha256']
            if digest != expected:
                raise ValueError(f'trace checksum mismatch: {source}')
            source_receipts.append(dict(file=str(source), sha256=digest))
        _, steps, _ = replay.load_states(paths, ('A', 'M'), 36, 8)
        for rows in steps.values():
            tasks.extend((period, (name, rows, [condition], 36, 8)) for period, condition in scenarios)
    print(f'{len(tasks)} step/policy/clock jobs from {len(source_receipts)} checked state files', flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(worker, tasks), 1):
            completed.extend(result)
            if i % 24 == 0:
                print(f'clock replay {i}/{len(tasks)}', flush=True)
    assert len(completed) == 24 * 2 * 3
    replay.write_csv(root / 'clock_step_timing.csv', completed)
    table = []
    for name in (c.name for _, c in scenarios):
        for cohort in sorted({r['cohort'] for r in completed}):
            relevant = [r for r in completed if r['condition'] == name and r['cohort'] == cohort]
            if not relevant:
                continue
            paired = {}
            for row in relevant:
                key = (row['state_set'], row['sample_id'], row['step'])
                paired.setdefault(key, {})[row['policy']] = row
            assert all(set(pair) == {'A', 'M'} for pair in paired.values())
            a, m = ([pair[p]['completion_ns'] / 1e6 for pair in paired.values()] for p in ('A', 'M'))
            table.append(dict(condition=name, cohort=cohort, paired_steps=len(paired),
                              A_mean_ms=fmean(a), M_mean_ms=fmean(m), M_vs_A_mean_percent=100 * (fmean(m) / fmean(a) - 1),
                              A_p95_ms=p95(a), M_p95_ms=p95(m),
                              A_peak_sram=max(pair['A']['peak_sram_bytes'] for pair in paired.values()),
                              M_peak_sram=max(pair['M']['peak_sram_bytes'] for pair in paired.values())))
    replay.write_csv(root / 'clock_summary.csv', table)
    (root / 'receipt.json').write_text(json.dumps(dict(
        evidence='sampled Qwen3 Attention address replay; no full-engine STA',
        change='shared 1.80/2.00 ns logic and zero result-side CDC cycles; same result credits and memory-side CDC',
        calibration_sha256=replay.digest(path), source_receipts=source_receipts,
        output_sha256={name: replay.digest(root / name) for name in ('clock_summary.csv', 'clock_step_timing.csv')},
        rows=len(completed)), indent=2) + '\n')
    print('Clock replay complete', flush=True)


if __name__ == '__main__':
    main()
