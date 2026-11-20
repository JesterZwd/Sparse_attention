#!/usr/bin/env python3
"""Prepare and checksum official legacy RULER v1, without model inference.

Prerequisites are the pinned git checkout, fully downloaded official essay
corpus, local Qwen tokenizer and punkt_tab (see manifest provenance). Existing
outputs are never overwritten. Exact raw generator rows are retained alongside
the final JSONLs, so span and chat adaptation can be independently audited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gqa_probe.ruler_data import finalize_row, validate_splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def prepare(config_path: Path, output_override: Path | None = None) -> None:
    import yaml
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    source = ROOT / config['generator']['source_directory']
    corpus = ROOT / config['corpus']['path']
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != config['generator']['commit']:
        raise ValueError('generator commit differs from pinned RULER v1 source')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'scripts', 'LICENSE'], check=True)
    if sha256(corpus) != config['corpus']['sha256']:
        raise ValueError('corpus checksum differs from pinned source')
    output = (output_override or ROOT / config['output_directory']).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing dataset: {output}')
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / 'provenance'
    provenance.mkdir()
    original = provenance / 'upstream'
    shutil.copytree(source / 'scripts', original / 'scripts')
    shutil.copy2(source / 'LICENSE', original / 'LICENSE')
    shutil.copy2(source / 'README.md', original / 'README.md')
    shutil.copy2(config_path, provenance / 'config.json')
    shutil.copy2(ROOT / 'requirements-ruler.txt', provenance / 'requirements-ruler.txt')
    corpus_copy = provenance / 'PaulGrahamEssays.json'
    shutil.copy2(corpus, corpus_copy)
    shutil.copy2(ROOT / config['corpus']['sources_manifest'], provenance / 'corpus_sources.json')
    shutil.copy2(ROOT / config['corpus']['download_script'], provenance / 'download_corpus.py')
    shutil.copy2(ROOT / 'gqa_probe/ruler_data.py', provenance / 'ruler_data.py')
    shutil.copy2(Path(__file__), provenance / 'prepare_ruler.py')
    (provenance / 'LICENSES.md').write_text(
        'Official RULER code: NVIDIA Apache License 2.0 (upstream/LICENSE).\n'
        'Paul Graham essays remain copyright of their authors; the upstream RULER\n'
        'download list does not grant an Apache license to essay text. Source URLs,\n'
        'original downloaded pages, and source hashes are retained for this local\n'
        'evaluation; no relicensing is asserted.\n'
        'Python dependency license files are preserved in dependency_licenses/.\n'
    )
    adapted = output / 'adapted'
    shutil.copytree(source / 'scripts/data', adapted / 'data')
    task_definitions = yaml.safe_load((source / 'scripts/synthetic.yaml').read_text())
    families = sorted({task_definitions[task]['task'] for task in config['tasks']})
    patches = []
    for family in families:
        relative = f'data/synthetic/{family}.py'
        script = adapted / relative
        before = script.read_text()
        old_import = 'from tokenizer import select_tokenizer'
        if before.count(old_import) != 1:
            raise ValueError(f'unexpected generator import in {family}')
        after = before.replace(old_import, 'from gqa_probe.ruler_data import select_chat_tokenizer as select_tokenizer')
        script.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                            fromfile=f'upstream/scripts/{relative}',
                                            tofile=f'adapted/scripts/{relative}'))
    (provenance / 'chat_budget.patch').write_text(''.join(patches))
    shutil.copy2(corpus, adapted / 'data/synthetic/json/PaulGrahamEssays.json')
    constants = {}
    exec(compile((source / 'scripts/data/synthetic/constants.py').read_text(), 'official_constants', 'exec'), constants)
    tokenizer_path = ROOT / config['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    dependencies = {}
    license_dir = provenance / 'dependency_licenses'
    requirement_lines = (ROOT / 'requirements-ruler.txt').read_text().splitlines()
    requirement_lines.extend(config.get('extra_requirements', []))
    for line in requirement_lines:
        if '==' not in line or line.startswith('#'):
            continue
        package, expected = line.split('==')
        distribution = importlib.metadata.distribution(package)
        if distribution.version != expected:
            raise ValueError(f'data dependency {package} is {distribution.version}, expected {expected}')
        dependencies[package] = distribution.version
        for file in distribution.files or []:
            if package == 'wonderwords' and str(file).endswith(('nounlist.txt', 'adjectivelist.txt')):
                asset = provenance / 'wonderwords_assets' / Path(str(file)).name
                asset.parent.mkdir(exist_ok=True)
                shutil.copy2(distribution.locate_file(file), asset)
            if any(term in str(file).lower() for term in ['license', 'copying']):
                target = license_dir / package / str(file)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(distribution.locate_file(file), target)
    nltk_data = ROOT / config['nltk_data_directory']
    punkt = nltk_data / 'tokenizers/punkt_tab/english'
    if not punkt.exists():
        raise FileNotFoundError('pinned punkt_tab English data is required')
    shutil.copytree(punkt, provenance / 'punkt_tab_english')
    shutil.copy2(ROOT / config['nltk_english_source_manifest'], provenance / 'nltk_english_source.json')
    nltk_license = nltk_data / 'tokenizers/punkt_tab/README'
    if nltk_license.exists():
        shutil.copy2(nltk_license, provenance / 'punkt_tab_README')
    manifest = {
        'schema_version': 1, 'status': 'preparing',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'benchmark': config['benchmark'], 'configuration': config,
        'config_sha256': sha256(config_path),
        'generator': {'repository': config['generator']['repository'], 'commit': commit,
                      'pipeline': 'official legacy RULER v1 main scripts',
                      'adaptation': config['generator']['adaptation'], 'patch_sha256': sha256(provenance / 'chat_budget.patch')},
        'corpus': {'sha256': sha256(corpus_copy), 'sources_sha256': sha256(provenance / 'corpus_sources.json'),
                   'source_count': len(json.loads((provenance / 'corpus_sources.json').read_text())),
                   'construction': config['corpus']['construction']},
        'tokenizer': {name: sha256(tokenizer_path / name) for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']},
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'dependencies': dependencies,
        'length_policy': 'final Qwen chat IDs including official assistant prefix plus task output reserve within configured context length; no token truncation',
        'gold_span_policy': 'NIAH association and value spans in final chat IDs; no gold spans for VT, CWE or FWE',
        'task_definitions': {task: task_definitions[task] for task in config['tasks']},
        'splits': {},
    }
    write_json(output / 'manifest.json', manifest)
    all_rows = []
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(ROOT)
    environment['NLTK_DATA'] = str(nltk_data)
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    for split, split_config in config['splits'].items():
        split_rows, seeds = [], {}
        for task_index, task in enumerate(config['tasks']):
            family = task_definitions[task]['task']
            task_spec = constants['TASKS'][family]
            max_new_tokens = config.get('task_max_new_tokens', {}).get(task, config['max_new_tokens'])
            script = adapted / f'data/synthetic/{family}.py'
            seed = split_config['seed'] + config['task_seed_stride'] * task_index
            seeds[task] = seed
            command = [sys.executable, str(script), '--save_dir', str(output / 'official_raw'),
                       '--save_name', task, '--subset', split,
                       '--tokenizer_path', str(tokenizer_path), '--tokenizer_type', 'hf',
                       '--max_seq_length', str(config['max_seq_length']),
                       '--tokens_to_generate', str(max_new_tokens),
                       '--num_samples', str(split_config['count_per_task']),
                       '--random_seed', str(seed), '--template', task_spec['template'] + task_spec['answer_prefix']]
            for key, value in task_definitions[task]['args'].items():
                command.extend([f'--{key}', str(value)])
            log = output / 'official_raw' / f'{split}_{task}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'Generating {split}/{task}: n={split_config["count_per_task"]}, seed={seed}', flush=True)
            with log.open('w') as stream:
                subprocess.run(command, check=True, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            raw_path = output / 'official_raw' / task / f'{split}.jsonl'
            raw_hash = sha256(raw_path)
            raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
            if len(raw_rows) != split_config['count_per_task']:
                raise ValueError('official generator returned the wrong row count')
            for raw in raw_rows:
                row = finalize_row(raw, tokenizer, task=task, split=split, seed=seed,
                                   max_seq_length=config['max_seq_length'], max_new_tokens=max_new_tokens,
                                   source_metadata={'generator_commit': commit, 'corpus_sha256': config['corpus']['sha256'],
                                                    'official_raw_file': str(raw_path.relative_to(output)),
                                                    'official_raw_sha256': raw_hash})
                if row['input_tokens'] + max_new_tokens != raw['length']:
                    raise ValueError('generator budget tokenizer and final chat tokenization disagree')
                split_rows.append(row)
        all_rows.extend(split_rows)
        validate_splits(all_rows, config['max_seq_length'])
        split_path = output / f'{split}.jsonl'
        with split_path.open('w') as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        manifest['splits'][split] = {'path': split_path.name, 'count': len(split_rows),
                                     'count_per_task': split_config['count_per_task'], 'seeds': seeds,
                                     'sha256': sha256(split_path),
                                     'input_tokens_min': min(row['input_tokens'] for row in split_rows),
                                     'input_tokens_max': max(row['input_tokens'] for row in split_rows)}
        write_json(output / 'manifest.json', manifest)
        print(f'Completed {split}: {len(split_rows)} rows; SHA256 {sha256(split_path)}', flush=True)
    manifest['status'] = 'complete'
    manifest['total_rows'] = len(all_rows)
    manifest['unique_instances'] = len({row['source_metadata']['instance_sha256'] for row in all_rows})
    manifest['provenance_files'] = {str(path.relative_to(output)): sha256(path)
                                    for path in sorted(provenance.rglob('*')) if path.is_file()}
    write_json(output / 'manifest.json', manifest)
    print(f'Complete: {len(all_rows)} unique official RULER v1 rows at {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ruler_round2.json')
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    prepare(arguments.config.resolve(), arguments.output)
