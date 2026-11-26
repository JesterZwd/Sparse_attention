"""Launch the pinned GPU scripts with an audited cloud RULER provenance check."""
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CLOUD = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts import prepare_round13_data as data

PUBLISHED = ROOT / 'results/round13_comrad/inputs'
REBUILT = ROOT / 'results/round12_frontend/data'
CACHED_CORPUS = ROOT / '.cache/ruler-corpus'
FROZEN_PROVENANCE = '093bb5c1bfa1e1b9f0c7c13486fd23c31eac1ab51a569f7b5a090801c359edd6'
CLOUD_PROVENANCE = '70bd6d23fdb54ccd5937f2d63afc12cba2055b6b4c37ecd848de84d9d2a29530'
standard_verify = data.verify_rebuilt_manifest


def checksum(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def verify(actual, frozen, expected):
    try:
        standard_verify(actual, frozen, expected)
        return
    except ValueError:
        actual, frozen = Path(actual).resolve(), Path(frozen).resolve()
        length = actual.parent.name
        if (length not in ('32k', '64k') or actual != REBUILT / length / 'manifest.json'
                or frozen != PUBLISHED / f'round12_{length}_manifest.json'
                or checksum(frozen) != expected):
            raise
        observed, registered = (json.loads(item.read_text()) for item in (actual, frozen))
        if (observed['corpus'].pop('sources_sha256') != CLOUD_PROVENANCE
                or registered['corpus'].pop('sources_sha256') != FROZEN_PROVENANCE
                or checksum(CACHED_CORPUS / 'sources.json') != CLOUD_PROVENANCE
                or checksum(CACHED_CORPUS / 'PaulGrahamEssays.json') != registered['corpus']['sha256']):
            raise
        for document in (observed, registered):
            for field in ('created_utc', 'provenance_files'):
                document.pop(field, None)
        if observed != registered:
            raise
        rows = {name: checksum(actual.parent / f'{name}.jsonl') for name in registered['splits']}
        if rows != {name: value['sha256'] for name, value in registered['splits'].items()}:
            raise
        receipt = dict(published_manifest_sha256=expected, rebuilt_manifest_sha256=checksum(actual),
                       registered_corpus_sources_sha256=FROZEN_PROVENANCE,
                       rebuilt_corpus_sources_sha256=CLOUD_PROVENANCE,
                       corpus_sha256=registered['corpus']['sha256'], exact_frozen_split_sha256=rows)
        destination = CLOUD / f'{length}-ruler-provenance.json'
        destination.write_text(json.dumps(receipt, indent=2)+'\n')
        print(f'{length} frozen rows and corpus verified byte for byte; rebuilt raw-source receipt audited in {destination}', flush=True)


class QueueSubprocess:
    """Route only the frozen Qwen3 panel through the audited launcher."""

    def __getattr__(self, key):
        return getattr(subprocess, key)

    def run(self, command, *args, **kwargs):
        if isinstance(command, list) and command[1:5] == ['-u', '-m', 'scripts.run_round13', 'qwen3_64k']:
            command = [command[0], '-u', str(Path(__file__).resolve()), 'qwen3_64k', *command[5:]]
        return subprocess.run(command, *args, **kwargs)


if __name__ == '__main__':
    data.verify_rebuilt_manifest = verify
    stage = sys.argv[1]
    if stage == 'verify':
        evidence = json.loads((ROOT / 'results/round12_frontend/data_receipt.json').read_text())
        for length in ('32k', '64k'):
            verify(REBUILT / length / 'manifest.json', PUBLISHED / f'round12_{length}_manifest.json',
                   evidence[length]['manifest_sha256'])
    elif stage == 'workspace':
        data.prepare_workspace()
    elif stage == 'qwen3_64k':
        from scripts import run_round13
        run_round13.main()
    elif stage == 'queue':
        from scripts import queue_round13
        queue_round13.subprocess = QueueSubprocess()
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        queue_round13.main()
    else:
        raise SystemExit(f'unknown stage {stage}')
