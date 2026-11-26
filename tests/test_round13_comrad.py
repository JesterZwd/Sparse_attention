import hashlib
import json

import pytest

from gqa_probe.round13_comrad import compare_gate


def test_comrad_gates_require_paired_quality_and_cost():
    baseline = {"read_per_decode":100, "arithmetic_per_decode":100, "overrun_groups":0}
    new = {"read_per_decode":94, "arithmetic_per_decode":99, "overrun_groups":0}
    assert compare_gate(baseline,new,-.005,-.01)["saving_pass"]
    assert not compare_gate(baseline,new,.02,-.013)["passes"]
    assert compare_gate(baseline,dict(new,read_per_decode=101),.01,.001)["quality_pass"]
    assert not compare_gate(baseline,dict(new,read_per_decode=101.01),.01,.001)["passes"]
    assert not compare_gate(baseline,dict(new,arithmetic_per_decode=100),0,-.005)["passes"]
    assert not compare_gate(baseline,dict(new,overrun_groups=1),.01,.005)["passes"]


def test_rebuilt_ruler_manifest_preserves_frozen_data(tmp_path):
    from scripts.prepare_round13_data import verify_rebuilt_manifest

    original = tmp_path / "original.json"
    actual = tmp_path / "manifest.json"
    frozen = dict(created_utc="original", provenance_files={"helper": "original"},
                  splits={"test": {"sha256": "frozen"}}, tokenizer={"tokenizer.json": "frozen"})
    original.write_text(json.dumps(frozen))
    rebuilt = dict(frozen, created_utc="new workstation", provenance_files={"helper": "rebuilt"})
    actual.write_text(json.dumps(rebuilt))
    verify_rebuilt_manifest(actual, original, hashlib.sha256(original.read_bytes()).hexdigest())

    actual.write_text(json.dumps(dict(rebuilt, splits={"test": {"sha256": "different"}})))
    with pytest.raises(ValueError, match="differs"):
        verify_rebuilt_manifest(actual, original, hashlib.sha256(original.read_bytes()).hexdigest())


def test_round12_replay_restores_only_verified_parts(tmp_path):
    from scripts.replay_round13_comrad import restore_round12_trace

    path = tmp_path / "states.jsonl.gz"
    parts = [b"first part", b"second part"]
    digest = lambda data: hashlib.sha256(data).hexdigest()
    hashes = {}
    for i, part in enumerate(parts):
        name = f"{path.name}.part{i:02}"
        (tmp_path / name).write_bytes(part)
        hashes[name] = digest(part)
    spec = dict(parts=hashes, reassembled_sha256=digest(b"".join(parts)))
    restore_round12_trace(path, spec, tmp_path)
    assert path.read_bytes() == b"".join(parts)
    path.unlink()
    (tmp_path / f"{path.name}.part00").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        restore_round12_trace(path, spec, tmp_path)
    assert not path.exists()
