import gzip
import json
from pathlib import Path

import pytest


def _calibration_rows():
    from gqa_probe.ruler_experiment import shadow_specs

    values = []
    for task, sample_id in (("key", "a"), ("value", "b")):
        for spec in shadow_specs("calibration"):
            values.append(dict(sample_id=sample_id, task=task, split="calibration",
                               spec=spec["name"], config_id=spec["config_id"],
                               absolute_error_p99=100., relative_error_p99=1.))
    return values


def _change(rows, name, absolute, relative):
    for row in rows:
        if row["spec"] == name:
            row.update(absolute_error_p99=absolute, relative_error_p99=relative)


def test_calibration_freeze_uses_only_primary_and_exact_prespecified_eligibility():
    from gqa_probe.ruler_experiment import freeze_candidate

    rows = _calibration_rows()
    _change(rows, "primary:SWAP:d0:s1", 99.5, 1.01)  # Both eligibility boundaries.
    _change(rows, "primary:SWAP:d0.01:s1", 90., 1.011)  # Relative violation.
    _change(rows, "primary:SWAP:d0.02:s1", 99.6, .5)  # Insufficient absolute improvement.
    _change(rows, "stress:SWAP:d0.02:s1", 1., .01)  # Stress cannot select a winner.
    selection = freeze_candidate(rows, expected_ids={("key", "a"), ("value", "b")},
                                 provenance={"quantizer": "affine32"})
    assert selection["selected"] == {"delta": 0., "max_swaps": 1}
    assert selection["status"] == "eligible_candidate"
    assert selection["calibration_prompt_count"] == 2
    assert selection["quantizer"] == "affine32"
    assert len(selection["calibration_metrics"]) == 10
    eligible = [row["spec"] for row in selection["candidate_metrics"] if row["eligible"]]
    assert eligible == ["primary:SWAP:d0:s1"]


def test_calibration_freeze_ties_prefer_fewer_swaps_then_smaller_delta():
    from gqa_probe.ruler_experiment import freeze_candidate

    rows = _calibration_rows()
    for name in ("primary:SWAP:d0:s1", "primary:SWAP:d0.01:s1", "primary:SWAP:d0.01:s2"):
        _change(rows, name, 99., 1.)
    selection = freeze_candidate(rows, expected_ids={("key", "a"), ("value", "b")}, provenance={})
    assert selection["selected"] == {"delta": 0., "max_swaps": 1}


def test_calibration_fallback_is_explicit_falsification_candidate_and_freeze_is_immutable(tmp_path):
    from gqa_probe.ruler_experiment import freeze_candidate

    output = tmp_path / "selection.json"
    selection = freeze_candidate(_calibration_rows(), expected_ids={("key", "a"), ("value", "b")},
                                 provenance={}, output=output)
    assert selection["selected"] == {"delta": .01, "max_swaps": 1}
    assert selection["status"] == "prespecified_falsification_candidate"
    assert "no eligible" in selection["reason"].lower()
    with pytest.raises(FileExistsError):
        freeze_candidate(_calibration_rows(), expected_ids={("key", "a"), ("value", "b")},
                         provenance={}, output=output)
    assert json.loads(output.read_text()) == selection


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "test_split", "nonfinite"])
def test_freeze_refuses_incomplete_duplicate_or_test_metrics(mutation):
    from gqa_probe.ruler_experiment import freeze_candidate

    rows = _calibration_rows()
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "test_split":
        rows[0]["split"] = "test"
    else:
        rows[0]["absolute_error_p99"] = float("nan")
    with pytest.raises(ValueError):
        freeze_candidate(rows, expected_ids={("key", "a"), ("value", "b")}, provenance={})


def test_run_store_refuses_overwrite_and_resume_config_drift(tmp_path):
    from gqa_probe.ruler_experiment import RunStore

    config = dict(phase="calibration", quantizer="affine32", data_sha256="abc")
    RunStore(tmp_path / "run", config)
    with pytest.raises(FileExistsError):
        RunStore(tmp_path / "run", config)
    RunStore(tmp_path / "run", config, resume=True)
    with pytest.raises(ValueError, match="config"):
        RunStore(tmp_path / "run", dict(config, quantizer="symmetric16"), resume=True)


def test_per_prompt_commit_is_atomic_idempotent_and_checksums_payloads(tmp_path):
    from gqa_probe.ruler_experiment import RunStore

    store = RunStore(tmp_path / "run", dict(phase="calibration", quantizer="affine32"))
    row = dict(sample_id="a", task="key", split="calibration")
    stage = store.begin_prompt(row)
    with gzip.open(stage / "shadow.jsonl.gz", "wt") as stream:
        stream.write('{"observed":1}\n')
    assert not store.is_complete(row)
    store.commit_prompt(row, stage, predictions=[{"method": "DenseBF16"}], summaries=[])
    assert store.is_complete(row)
    assert store.begin_prompt(row) is None
    progress = json.loads((store.output / "progress.json").read_text())
    assert progress["completed_prompts"] == 1
    assert progress["completed_generations"] == 1
    assert not {"official_score", "complete_correct", "accuracy"} & progress.keys()
    with pytest.raises(FileExistsError):
        store.commit_prompt(row, stage, predictions=[], summaries=[])
    resumed = RunStore(store.output, store.config, resume=True)
    assert resumed.is_complete(row)
    next((store.output / "prompts").glob("*/predictions.json")).write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        RunStore(store.output, store.config, resume=True)


def test_resume_recomputes_only_incomplete_prompt_work(tmp_path):
    from gqa_probe.ruler_experiment import RunStore

    store = RunStore(tmp_path / "run", dict(phase="calibration", quantizer="affine32"))
    row = dict(sample_id="a", task="key", split="calibration")
    stage = store.begin_prompt(row)
    (stage / "unfinished.json").write_text("partial")
    resumed = RunStore(store.output, store.config, resume=True)
    fresh = resumed.begin_prompt(row)
    assert not (fresh / "unfinished.json").exists()
    assert not resumed.is_complete(row)


def test_test_phase_requires_frozen_selection_before_touching_dataset(tmp_path):
    from gqa_probe.ruler_experiment import main

    with pytest.raises(ValueError, match="selection"):
        main(["--phase", "test", "--output", str(tmp_path / "run"), "--quantizer", "affine32",
              "--data-dir", str(tmp_path / "does-not-exist")])


def test_source_snapshots_remain_valid_after_live_code_changes_and_detect_corruption(tmp_path):
    from gqa_probe.ruler_experiment import snapshot_sources, verify_snapshot

    live = tmp_path / "live.py"
    live.write_text("answer = 1\n")
    payloads = {"gqa_probe/example.py": live.read_bytes()}
    hashes = snapshot_sources(tmp_path / "run", payloads)
    live.write_text("answer = 2\n")
    frozen = verify_snapshot(tmp_path / "run", hashes)
    assert (frozen / "gqa_probe/example.py").read_text() == "answer = 1\n"
    (frozen / "gqa_probe/example.py").write_text("answer = 3\n")
    with pytest.raises(ValueError, match="snapshot|checksum"):
        verify_snapshot(tmp_path / "run", hashes)


def test_test_selection_refuses_quantizer_or_calibration_hash_mismatch():
    from gqa_probe.ruler_experiment import validate_selection

    selection = dict(phase="calibration", quantizer="affine32", status="eligible_candidate",
                     selected={"delta": .01, "max_swaps": 1}, calibration_prompt_count=128,
                     provenance={"calibration_data_sha256": "frozen"})
    manifest = {"splits": {"calibration": {"count": 128, "sha256": "frozen"}}}
    validate_selection(selection, "affine32", manifest=manifest)
    with pytest.raises(ValueError, match="quantizer"):
        validate_selection(selection, "symmetric16", manifest=manifest)
    manifest["splits"]["calibration"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="calibration"):
        validate_selection(selection, "affine32", manifest=manifest)


@pytest.mark.parametrize("phase,expected_generations,expected_shadows", [("calibration", 2, 10), ("test", 6, 4)])
def test_prompt_orchestration_shares_only_the_kv4_prefix_and_uses_fresh_attention(
        tmp_path, phase, expected_generations, expected_shadows):
    from types import SimpleNamespace
    from gqa_probe.ruler_experiment import ExperimentRuntime, execute_prompt

    generated = []
    prefixes = []

    def prefill(model, ids, **kwargs):
        state = SimpleNamespace(cache=object(), input_tokens=len(ids), elapsed_seconds=.25)
        prefixes.append(state)
        return state

    class Attention:
        def __init__(self, algorithm="dense", **kwargs):
            self.algorithm = algorithm
            self.kwargs = kwargs
            self.shadow_work = {}

    def generate(model, state, count, eos, attention):
        generated.append((state, attention))
        for spec in attention.kwargs.get("shadow_specs", []):
            attention.kwargs["callback"](dict(
                spec=spec["name"], algorithm=spec["algorithm"], budget_fraction=spec["budget_fraction"],
                target=spec["target"], delta=spec["delta"], max_swaps=spec["max_swaps"],
                layer=0, step=0, kv_group=0, head_coverage=[.5],
                head_output_absolute_errors=[.1], head_output_relative_errors=[.2],
                pages=[0], original_target_deficit=.3, gate_hit=False, swaps=0,
                candidate_pairs=0, executed_candidate_pairs=0, work={}))
        return SimpleNamespace(token_ids=[7, 9], attention_stats={"shadow_audit_seconds": .1}, elapsed_seconds=1.)

    runtime = ExperimentRuntime(model=object(), tokenizer=SimpleNamespace(decode=lambda *a, **k: "VALUE"),
                                eos_ids=[9], prefill_bf16=prefill, prefill_kv4=prefill,
                                generate=generate, attention_factory=Attention)
    row = dict(sample_id="a", task="key", split=phase, input_ids=[1, 2, 3], input_tokens=3,
               references=["value"], gold_spans=[[0, 1]], max_new_tokens=128)
    selection = {"selected": {"delta": .01, "max_swaps": 1}}
    predictions, summaries = execute_prompt(runtime, row, phase=phase, quantizer="affine32",
                                           stage=tmp_path, run_id="frozen", selection=selection)
    assert len(prefixes) == 2
    assert len(predictions) == len(generated) == expected_generations
    assert generated[0][0] is prefixes[0]
    assert all(state is prefixes[1] for state, _ in generated[1:])
    assert len({id(attention) for _, attention in generated}) == expected_generations
    assert len(summaries) == expected_shadows
    assert all(row["official_score"] == 1. and row["complete_correct"] for row in predictions)
    assert all(row["eos_reached"] and not row["truncated"] for row in predictions)
    assert all(row["max_new_tokens"] == 128 and row["input_tokens"] == 3 for row in predictions)
    assert all(row["generation_without_shadow_audit_seconds"] == .9 for row in predictions)
    assert {row["config_id"] for row in summaries} == {"primary", "stress"}
    with gzip.open(tmp_path / "shadow.jsonl.gz", "rt") as stream:
        raw = [json.loads(line) for line in stream]
    assert len(raw) == expected_shadows
    assert all(row["run_id"] == "frozen" for row in raw)


def test_cost_export_contains_only_normalized_mass_for_every_group_and_metadata(tmp_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from gqa_probe.ruler_experiment import _make_attention

    attention = _make_attention("dense", cost_directory=tmp_path,
                                cost_metadata={"sample_id": "a", "task": "key"})
    attention.step = 0
    attention._audit(SimpleNamespace(layer_idx=7), torch.zeros(8, 4, 65),
                     torch.zeros(1, 8, 65, 2), torch.zeros(8, 4, 2))
    with np.load(tmp_path / "layer07_step000.npz", allow_pickle=False) as payload:
        assert set(payload.files) == {"mass", "metadata_json"}
        assert payload["mass"].shape == (8, 3, 4)
        np.testing.assert_allclose(payload["mass"][0, :, 0], [32 / 65, 32 / 65, 1 / 65])
        np.testing.assert_allclose(payload["mass"].sum(axis=1), np.ones((8, 4)))
        metadata = json.loads(str(payload["metadata_json"]))
    assert metadata["sample_id"] == "a"
    assert metadata["layer"] == 7 and metadata["step"] == 0
    assert metadata["mass_axes"] == ["kv_group", "page", "query_head"]


def test_quantizer_freeze_checks_engineering_choice_and_linked_configuration(tmp_path):
    from gqa_probe.ruler_experiment import load_quantizer_selection

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(decision="selected", selected_spec="affine32", config_id="engineering")))
    (tmp_path / "config.json").write_text(json.dumps({"config_id": "engineering"}))
    frozen = load_quantizer_selection(selection, "affine32")
    assert frozen["selected_spec"] == "affine32"
    with pytest.raises(ValueError, match="quantizer|engineering"):
        load_quantizer_selection(selection, "symmetric16")
    (tmp_path / "config.json").write_text(json.dumps({"config_id": "different"}))
    with pytest.raises(ValueError, match="config"):
        load_quantizer_selection(selection, "affine32")


def test_aggregate_completion_requires_all_prompts_and_preserves_dataset_order(tmp_path):
    from gqa_probe.ruler_experiment import RunStore, finalize_run

    rows = [dict(sample_id=value, task="key", split="test") for value in ("a", "b")]
    store = RunStore(tmp_path / "run", dict(phase="test", quantizer="affine32", expected_prompts=2))

    def commit(row):
        stage = store.begin_prompt(row)
        with gzip.open(stage / "shadow.jsonl.gz", "wt"):
            pass
        store.commit_prompt(row, stage, predictions=[dict(row, method="DenseBF16")], summaries=[])

    commit(rows[1])
    with pytest.raises(ValueError, match="complete|prompt"):
        finalize_run(store, rows)
    assert not (store.output / "complete.json").exists()
    commit(rows[0])
    complete = finalize_run(store, rows)
    assert complete["completed_prompts"] == 2
    assert complete["completed_generations"] == 2
    with gzip.open(store.output / "per_prompt_predictions.jsonl.gz", "rt") as stream:
        predictions = [json.loads(line) for line in stream]
    assert [row["sample_id"] for row in predictions] == ["a", "b"]
    assert finalize_run(store, rows) == complete


def test_resume_code_drift_error_names_explicit_snapshot_and_absolute_assets(tmp_path):
    from types import SimpleNamespace
    from gqa_probe.ruler_experiment import require_frozen_code

    args = SimpleNamespace(phase="test", output=tmp_path / "output", model=tmp_path / "model",
                           data_dir=tmp_path / "data", quantizer="affine32", resume=True,
                           selection=tmp_path / "selection.json", quantizer_selection=tmp_path / "quant.json")
    with pytest.raises(ValueError) as caught:
        require_frozen_code(args, {"gqa_probe/a.py": "new"}, {"gqa_probe/a.py": "old"}, tmp_path / "calibration")
    message = str(caught.value)
    assert str(tmp_path / "calibration/source_snapshot/resume_launcher.py") in message
    assert str(args.model.resolve()) in message and str(args.data_dir.resolve()) in message
    assert "--resume" in message and "--selection" in message and "--quantizer-selection" in message


def test_dataset_loading_checks_only_requested_split_bytes_and_rejects_hash_drift(tmp_path):
    import hashlib
    from gqa_probe.ruler_experiment import load_dataset

    row = dict(sample_id="a", task="key", split="calibration", input_ids=[1, 2],
               input_tokens=2, max_new_tokens=128, references=["value"], gold_spans=[[0, 1]])
    data = tmp_path / "calibration.jsonl"
    data.write_text(json.dumps(row) + "\n")
    manifest = dict(status="complete", configuration={"tasks": ["key"]},
                    splits={"calibration": {"path": data.name, "count": 1, "count_per_task": 1,
                                             "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, loaded_manifest, data_path = load_dataset(tmp_path, "calibration", expected_per_task=1)
    assert loaded == [row] and data_path == data
    assert loaded_manifest == manifest
    data.write_text(json.dumps(dict(row, references=["changed"])) + "\n")
    with pytest.raises(ValueError, match="checksum|hash"):
        load_dataset(tmp_path, "calibration", expected_per_task=1)


@pytest.mark.parametrize("eos", [True, False])
def test_eos_only_prefill_uses_explicit_policy_zero_and_calibration_can_finalize(tmp_path, eos):
    from types import SimpleNamespace
    from gqa_probe.ruler_experiment import ExperimentRuntime, RunStore, execute_prompt, finalize_run

    state = SimpleNamespace(input_tokens=2, elapsed_seconds=0.)
    runtime = ExperimentRuntime(
        model=None, tokenizer=SimpleNamespace(decode=lambda *a, **k: ""), eos_ids=[9],
        prefill_bf16=lambda *a, **k: state, prefill_kv4=lambda *a, **k: state,
        generate=lambda *a: SimpleNamespace(token_ids=[9 if eos else 7],
                                             attention_stats={}, elapsed_seconds=0.),
        attention_factory=lambda *a, **k: SimpleNamespace(shadow_work={}))
    row = dict(sample_id="a", task="key", split="calibration", input_ids=[1, 2],
               input_tokens=2, references=["value"], gold_spans=[[0, 1]], max_new_tokens=128)
    config = dict(phase="calibration", quantizer="affine32", data_sha256="data",
                  dataset_manifest_sha256="manifest", code_sha256={}, source_sha256={},
                  model_file_sha256={}, quantizer_engineering_selection_sha256="quant")
    store = RunStore(tmp_path / "run", config)
    stage = store.begin_prompt(row)
    if not eos:
        with pytest.raises(ValueError, match="EOS|eos|decode"):
            execute_prompt(runtime, row, phase="calibration", quantizer="affine32", stage=stage, run_id=store.run_id)
        return
    predictions, summaries = execute_prompt(runtime, row, phase="calibration", quantizer="affine32",
                                           stage=stage, run_id=store.run_id)
    assert len(summaries) == 10
    assert all(row["no_decode_observations"] and row["n_records"] == row["n_heads"] == 0 for row in summaries)
    assert all(row["absolute_error_p99"] == row["relative_error_p99"] == 0. for row in summaries)
    store.commit_prompt(row, stage, predictions=predictions, summaries=summaries)
    complete = finalize_run(store, [row])
    assert complete["no_decode_prompts"] == 1
    assert complete["raw_shadow_records"] == 0
    selection = json.loads((store.output / "selection.json").read_text())
    assert selection["status"] == "prespecified_falsification_candidate"
