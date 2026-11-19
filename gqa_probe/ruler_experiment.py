"""Frozen Round-2 calibration/test orchestration with atomic prompt checkpoints.

Run as ``python -m gqa_probe.ruler_experiment --phase calibration|test ...``.
Inference is an Oracle numeric reference, with dense prefill and grouped FP32
decode; timing is instrumented and makes no sparse-kernel speed claim.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import uuid

import numpy as np

from .paired_statistics import score_prediction, summarize_shadow


SAMPLE_LAYERS = [0, 7, 14, 21, 28, 35]
SAMPLE_STEPS = [0, 1, 3, 7, 15, 31, 63, 127]
FOCUS = {"primary": (.8, .2), "stress": (.9, .1)}
CANDIDATES = [(0., 1), (.01, 1), (.02, 1), (.01, 2)]
TASKS = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value, *, replace=True):
    """Fsync then publish atomically; exclusive writes never replace a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def shadow_specs(phase, selection=None):
    """Ten calibration shadows, or four test shadows using the frozen swap."""
    if phase not in {"calibration", "test"}:
        raise ValueError("phase must be calibration or test")
    if phase == "test" and selection is None:
        raise ValueError("test requires a frozen selection")
    candidates = CANDIDATES if phase == "calibration" else [(
        selection["selected"]["delta"], selection["selected"]["max_swaps"])]
    specs = []
    for config_id, (target, fraction) in FOCUS.items():
        base = dict(config_id=config_id, budget_fraction=fraction, target=target)
        specs.append(dict(base, name=f"{config_id}:A1", algorithm="A1", delta=.01, max_swaps=1))
        specs.extend(dict(base, name=f"{config_id}:SWAP:d{delta:g}:s{swaps}", algorithm="SWAP",
                          delta=delta, max_swaps=swaps) for delta, swaps in candidates)
    return specs


def freeze_candidate(summaries, *, expected_ids, provenance, output=None):
    """Freeze one candidate from complete calibration prompt summaries only.

    Eligibility and ranking use mean *prompt* P99 under the primary setting:
    absolute <= .995*A1 and relative <= 1.01*A1. Stress summaries are retained
    for transparency and never enter selection. Zero baseline absolute error
    cannot establish a fractional improvement. Ties prefer fewer swaps, then
    smaller delta. Every expected prompt/spec must appear exactly once.
    """
    expected_ids = set(expected_ids)
    if not expected_ids:
        raise ValueError("calibration identity set must be nonempty")
    specs = shadow_specs("calibration")
    by_name = {spec["name"]: [] for spec in specs}
    seen = set()
    for row in summaries:
        identity = (row["task"], row["sample_id"])
        name = row["spec"]
        if row.get("split") != "calibration" or identity not in expected_ids or name not in by_name:
            raise ValueError("selection accepts only the exact frozen calibration cohort/specs")
        key = (identity, name)
        if key in seen:
            raise ValueError("duplicate calibration prompt/spec")
        seen.add(key)
        for field in ("absolute_error_p99", "relative_error_p99"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("calibration tail metrics must be finite and nonnegative")
        by_name[name].append(row)
    if len(seen) != len(expected_ids) * len(specs):
        raise ValueError("calibration prompt/spec set is incomplete")
    metrics = []
    for spec in specs:
        rows = by_name[spec["name"]]
        metrics.append(dict(spec=spec["name"], config_id=spec["config_id"],
                            algorithm=spec["algorithm"], delta=spec["delta"], max_swaps=spec["max_swaps"],
                            n_prompts=len(rows),
                            mean_absolute_error_p99=float(np.mean([r["absolute_error_p99"] for r in rows])),
                            mean_relative_error_p99=float(np.mean([r["relative_error_p99"] for r in rows]))))
    baseline = next(row for row in metrics if row["spec"] == "primary:A1")
    candidates = []
    for row in metrics:
        if row["config_id"] != "primary" or row["algorithm"] != "SWAP":
            continue
        row = dict(row)
        absolute = row["mean_absolute_error_p99"]
        relative = row["mean_relative_error_p99"]
        base_abs, base_rel = baseline["mean_absolute_error_p99"], baseline["mean_relative_error_p99"]
        row["absolute_improvement_fraction"] = 1 - absolute / base_abs if base_abs else None
        row["relative_error_change_fraction"] = relative / base_rel - 1 if base_rel else None
        row["absolute_eligible"] = bool(base_abs > 0 and absolute <= .995 * base_abs)
        row["relative_eligible"] = bool(relative <= 1.01 * base_rel)
        row["eligible"] = row["absolute_eligible"] and row["relative_eligible"]
        candidates.append(row)
    eligible = sorted((row for row in candidates if row["eligible"]), key=lambda row: (
        row["mean_absolute_error_p99"], row["max_swaps"], row["delta"]))
    chosen = eligible[0] if eligible else dict(delta=.01, max_swaps=1)
    result = dict(
        schema_version=1, phase="calibration", created_utc=_utc(),
        quantizer=provenance.get("quantizer"), provenance=dict(provenance),
        selected=dict(delta=chosen["delta"], max_swaps=chosen["max_swaps"]),
        status="eligible_candidate" if eligible else "prespecified_falsification_candidate",
        reason=("Lowest mean prompt absolute P99 among eligible primary-setting candidates."
                if eligible else "No eligible candidate; freeze the prespecified delta=.01/one-swap falsification candidate, not a claimed winner."),
        recipe=dict(selection_setting="primary", absolute_multiplier_max=.995,
                    relative_multiplier_max=1.01, aggregation="mean_of_prompt_p99",
                    ranking=["mean_absolute_error_p99", "max_swaps", "delta"]),
        calibration_prompt_count=len(expected_ids), calibration_metrics=metrics,
        candidate_metrics=candidates, calibration_prompt_metrics=list(summaries),
        calibration_prompt_metrics_sha256=_digest(summaries))
    if output is not None:
        _write_json(output, result, replace=False)
    return result


def _core_config(config):
    return {key: value for key, value in config.items() if key not in {"run_id", "created_utc"}}


class RunStore:
    """One immutable, checksummed directory per fully computed prompt."""

    def __init__(self, output, config, *, resume=False):
        self.output = Path(output).resolve()
        core = _core_config(config)
        self.run_id = _digest(core)
        config_path = self.output / "config.json"
        if config_path.exists():
            if not resume:
                raise FileExistsError("config.json exists; explicit --resume is required")
            previous = json.loads(config_path.read_text())
            if _core_config(previous) != core or previous["run_id"] != self.run_id:
                raise ValueError("resume configuration mismatch")
            self.config = previous
        else:
            if resume:
                raise ValueError("resume requires an existing frozen config.json")
            if self.output.exists() and any(self.output.iterdir()):
                raise FileExistsError("new run output directory must be empty")
            self.output.mkdir(parents=True, exist_ok=True)
            self.config = dict(core, run_id=self.run_id, created_utc=_utc())
            _write_json(config_path, self.config, replace=False)
        (self.output / "prompts").mkdir(exist_ok=True)
        (self.output / "incomplete").mkdir(exist_ok=True)
        self._committed = {}
        for directory in sorted((self.output / "prompts").iterdir()):
            if not directory.is_dir():
                raise ValueError("unexpected entry in committed prompt directory")
            bundle = json.loads((directory / "bundle.json").read_text())
            if bundle["run_id"] != self.run_id or directory.name != self.prompt_key(bundle):
                raise ValueError("committed prompt configuration/identity mismatch")
            for name, expected in bundle["files_sha256"].items():
                file = directory / name
                if not file.is_file() or _sha(file) != expected:
                    raise ValueError(f"prompt payload checksum mismatch: {name}")
            self._committed[directory.name] = bundle

    @staticmethod
    def prompt_key(row):
        return _digest([row["task"], row["sample_id"]])[:24]

    def is_complete(self, row):
        key = self.prompt_key(row)
        if key not in self._committed:
            return False
        saved = self._committed[key]
        if any(saved.get(name) != row.get(name) for name in ("sample_id", "task", "split")):
            raise ValueError("prompt identity collision or split mismatch")
        return True

    def begin_prompt(self, row):
        if self.is_complete(row):
            return None
        stage = self.output / "incomplete" / self.prompt_key(row)
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        return stage

    def commit_prompt(self, row, stage, *, predictions, summaries):
        key = self.prompt_key(row)
        final = self.output / "prompts" / key
        if key in self._committed or final.exists():
            raise FileExistsError("prompt is already committed")
        stage = Path(stage)
        if stage != self.output / "incomplete" / key or not stage.is_dir():
            raise ValueError("invalid prompt staging directory")
        if "predictions_per_prompt" in self.config and len(predictions) != self.config["predictions_per_prompt"]:
            raise ValueError("incomplete prompt prediction set")
        _write_json(stage / "predictions.json", predictions, replace=False)
        _write_json(stage / "summaries.json", summaries, replace=False)
        files = {str(path.relative_to(stage)): _sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        bundle = dict(schema_version=1, run_id=self.run_id,
                      **{name: row[name] for name in ("sample_id", "task", "split")},
                      n_predictions=len(predictions), n_summaries=len(summaries), files_sha256=files)
        _write_json(stage / "bundle.json", bundle, replace=False)
        os.rename(stage, final)
        self._committed[key] = bundle
        self.progress("running")

    def progress(self, status, **extra):
        value = dict(status=status, run_id=self.run_id, updated_utc=_utc(),
                     completed_prompts=len(self._committed),
                     completed_generations=sum(row["n_predictions"] for row in self._committed.values()),
                     expected_prompts=self.config.get("expected_prompts"),
                     expected_generations=self.config.get("expected_generations"), **extra)
        _write_json(self.output / "progress.json", value)
        return value


_LAUNCHER = ("from pathlib import Path\nimport runpy, sys\n"
             "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
             "runpy.run_module('gqa_probe.ruler_experiment', run_name='__main__')\n")


def snapshot_sources(output, payloads):
    """Publish source bytes once; resume uses this package rather than live code."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    if snapshot.exists():
        raise FileExistsError("source snapshot already exists")
    temporary = output / f".snapshot-{uuid.uuid4().hex}"
    temporary.mkdir()
    payloads = dict(payloads, **{"resume_launcher.py": _LAUNCHER.encode()})
    hashes = {}
    for name, data in payloads.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid relative snapshot source path")
        file = temporary / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    os.rename(temporary, snapshot)
    return hashes


def verify_snapshot(output, hashes):
    snapshot = Path(output).resolve() / "source_snapshot"
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid snapshot source path")
        file = snapshot / relative
        if not file.is_file() or _sha(file) != expected:
            raise ValueError(f"frozen source snapshot checksum mismatch: {name}")
    return snapshot


def validate_selection(selection, quantizer, *, manifest=None, code_sha256=None, model_sha256=None):
    if (selection.get("phase") != "calibration" or selection.get("status") not in {
            "eligible_candidate", "prespecified_falsification_candidate"} or
            selection.get("selected") not in [dict(delta=delta, max_swaps=swaps) for delta, swaps in CANDIDATES]):
        raise ValueError("invalid frozen calibration selection")
    if selection.get("quantizer") != quantizer:
        raise ValueError("test quantizer differs from frozen calibration selection")
    provenance = selection.get("provenance", {})
    if manifest is not None:
        calibration = manifest["splits"]["calibration"]
        if (provenance.get("calibration_data_sha256") != calibration["sha256"] or
                selection.get("calibration_prompt_count") != calibration["count"]):
            raise ValueError("frozen selection calibration dataset mismatch")
    if code_sha256 is not None and provenance.get("code_sha256") != code_sha256:
        raise ValueError("frozen selection code hash mismatch")
    if model_sha256 is not None and provenance.get("model_file_sha256") != model_sha256:
        raise ValueError("frozen selection model hash mismatch")


@dataclass
class ExperimentRuntime:
    model: object
    tokenizer: object
    eos_ids: list[int]
    prefill_bf16: object
    prefill_kv4: object
    generate: object
    attention_factory: object = None


def _make_attention(algorithm="dense", *, cost_directory=None, cost_metadata=None, **kwargs):
    # No GPU or quantizer dependencies are imported by pure storage/selection
    # helpers. The quantizer implementation is resolved only by actual runs.
    from .oracle_attention import OracleAttention, torch_page_mass
    from .selection import protected_pages

    if cost_directory is None:
        return OracleAttention(algorithm, **kwargs)

    class CostSamplingOracleAttention(OracleAttention):
        def _audit(self, module, logits, value, dense):
            if self.step == 0:
                directory = Path(cost_directory)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"layer{module.layer_idx:02d}_step000.npz"
                metadata = dict(cost_metadata or {}, layer=module.layer_idx, step=0,
                                valid_tokens=int(logits.shape[-1]), page_size=32,
                                protected=protected_pages(int(logits.shape[-1])),
                                mass_axes=["kv_group", "page", "query_head"])
                # This exports normalized mass only. The superclass caller has
                # already started its shadow-audit timer around this method.
                mass = torch_page_mass(logits).cpu().numpy()
                with path.open("xb") as stream:
                    np.savez_compressed(stream, mass=mass, metadata_json=json.dumps(metadata, sort_keys=True))
                self.shadow_work["cost_mass_exports"] = self.shadow_work.get("cost_mass_exports", 0) + 1
            return super()._audit(module, logits, value, dense)

    return CostSamplingOracleAttention(algorithm, **kwargs)


def _empty_summaries(row, specs, run_id, quantizer):
    """An EOS-only prefill has no sparse decode perturbation, hence zero tails."""
    result = []
    for spec in specs:
        result.append(dict(
            sample_id=row["sample_id"], task=row["task"], split=row["split"],
            spec=spec["name"], config_id=spec["config_id"], run_id=run_id, kv_quantizer=quantizer,
            algorithm=spec["algorithm"], budget_fraction=spec["budget_fraction"], target=spec["target"],
            delta=spec["delta"], max_swaps=spec["max_swaps"], absolute_error_p95=0., absolute_error_p99=0.,
            relative_error_p95=0., relative_error_p99=0., coverage_p05=None, mean_deficit=None,
            mean_pages=None, gate_hit_rate=None, sum_swaps=0, sum_candidate_pairs=0,
            sum_executed_candidate_pairs=0, mean_gold_span_recall=None, mean_gold_token_recall=None,
            mean_gold_page_recall=None, n_records=0, n_heads=0, sum_work={}, no_decode_observations=True))
    return result


def execute_prompt(runtime, row, *, phase, quantizer, stage, run_id, selection=None, cost_sample=False):
    """Compute a complete prompt bundle without mutating shared prefill states."""
    stage = Path(stage)
    specs = shadow_specs(phase, selection)
    by_name = {spec["name"]: spec for spec in specs}
    records, predictions = [], []
    factory = runtime.attention_factory or _make_attention
    eos_ids = set(runtime.eos_ids)

    with gzip.open(stage / "shadow.jsonl.gz", "xt", encoding="utf-8") as stream:
        def callback(record):
            spec = by_name[record["spec"]]
            enriched = dict(record, sample_id=row["sample_id"], task=row["task"], split=row["split"],
                            config_id=spec["config_id"], run_id=run_id, kv_quantizer=quantizer)
            stream.write(_json_bytes(enriched).decode() + "\n")
            records.append(enriched)

        def predict(state, attention, method, config_id, *, reused=False, spec=None):
            generated = runtime.generate(runtime.model, state, 128, runtime.eos_ids, attention)
            tokens = [int(token) for token in generated.token_ids]
            text = runtime.tokenizer.decode(tokens, skip_special_tokens=True)
            audit_seconds = float(generated.attention_stats.get("shadow_audit_seconds", 0.))
            elapsed = float(generated.elapsed_seconds)
            if audit_seconds > elapsed + 1e-6:
                raise ValueError("shadow audit time exceeds generation time")
            eos_reached = bool(tokens and tokens[-1] in eos_ids)
            prediction = dict(
                sample_id=row["sample_id"], task=row["task"], split=row["split"],
                method=method, config_id=config_id, run_id=run_id,
                kv_quantizer="bf16" if method == "DenseBF16" else quantizer,
                token_ids=tokens, text=text, references=row["references"], **score_prediction(text, row["references"]),
                eos_reached=eos_reached, truncated=not eos_reached and len(tokens) == 128,
                generated_tokens=len(tokens), input_tokens=state.input_tokens, max_new_tokens=128,
                prefill_seconds=float(state.elapsed_seconds), prefill_reused=reused,
                generation_seconds=elapsed, audit_seconds=audit_seconds,
                generation_without_shadow_audit_seconds=elapsed - audit_seconds,
                attention_stats=generated.attention_stats, shadow_work=dict(attention.shadow_work))
            if spec is not None:
                prediction.update(spec=spec["name"], budget_fraction=spec["budget_fraction"],
                                  target=spec["target"], delta=spec["delta"], max_swaps=spec["max_swaps"])
            predictions.append(prediction)

        state_bf16 = runtime.prefill_bf16(runtime.model, row["input_ids"], kv_bits=16)
        predict(state_bf16, factory("dense", gold_spans=row["gold_spans"]), "DenseBF16", "dense")
        del state_bf16
        state_kv4 = runtime.prefill_kv4(runtime.model, row["input_ids"], spec=quantizer)
        dense_attention = factory(
            "dense", shadow_specs=specs, sample_layers=SAMPLE_LAYERS, sample_steps=SAMPLE_STEPS,
            callback=callback, gold_spans=row["gold_spans"],
            cost_directory=stage / "cost_mass" if cost_sample else None,
            cost_metadata=dict(sample_id=row["sample_id"], task=row["task"], split=row["split"], run_id=run_id))
        predict(state_kv4, dense_attention, "DenseKV4", "dense")
        if phase == "test":
            for spec in specs:
                attention = factory(spec["algorithm"], budget_fraction=spec["budget_fraction"],
                                    target=spec["target"], delta=spec["delta"], max_swaps=spec["max_swaps"],
                                    gold_spans=row["gold_spans"])
                predict(state_kv4, attention, spec["algorithm"], spec["config_id"], reused=True, spec=spec)
    if not records:
        dense = next(record for record in predictions if record["method"] == "DenseKV4")
        if dense["generated_tokens"] != 1 or not dense["eos_reached"]:
            raise ValueError("only a first-token EOS may have zero decode shadow observations")
        summaries = _empty_summaries(row, specs, run_id, quantizer)
    else:
        summaries = summarize_shadow(records)
        if {summary["spec"] for summary in summaries} != set(by_name):
            raise ValueError("prompt shadow specifications are incomplete")
    return predictions, summaries


def load_quantizer_selection(path, quantizer):
    path = Path(path)
    selection = json.loads(path.read_text())
    if selection.get("decision") != "selected" or selection.get("selected_spec") != quantizer:
        raise ValueError("requested quantizer differs from the frozen engineering selection")
    config = json.loads((path.parent / "config.json").read_text())
    if config.get("config_id") != selection.get("config_id"):
        raise ValueError("quantizer engineering selection/config mismatch")
    return selection


def require_frozen_code(args, actual, expected, snapshot_output):
    if actual == expected:
        return
    launcher = Path(snapshot_output).resolve() / "source_snapshot/resume_launcher.py"
    command = [sys.executable, str(launcher), "--phase", args.phase, "--output", str(args.output.resolve()),
               "--quantizer", args.quantizer, "--model", str(args.model.resolve()),
               "--data-dir", str(args.data_dir.resolve()),
               "--quantizer-selection", str(args.quantizer_selection.resolve())]
    if args.resume:
        command.append("--resume")
    if args.selection is not None:
        command.extend(["--selection", str(args.selection.resolve())])
    raise ValueError("current code differs from frozen source hashes. Review the preserved snapshot, "
                     "then explicitly run its launcher with absolute assets: " + shlex.join(command))


def load_dataset(data_dir, phase, *, expected_per_task=None):
    """Read and checksum only the requested split, never another split's rows."""
    if phase not in {"calibration", "test"}:
        raise ValueError("formal dataset phase must be calibration or test")
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("dataset manifest is incomplete")
    metadata = manifest["splits"][phase]
    path = data_dir / metadata["path"]
    if _sha(path) != metadata["sha256"]:
        raise ValueError("requested dataset split checksum mismatch")
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    tasks = manifest["configuration"]["tasks"]
    count = expected_per_task if expected_per_task is not None else (32 if phase == "calibration" else 128)
    if len(rows) != metadata["count"] or Counter(row["task"] for row in rows) != Counter({task: count for task in tasks}):
        raise ValueError("dataset prompt/task counts differ from the prespecified cohort")
    identities = set()
    for row in rows:
        identity = (row["task"], row["sample_id"])
        if identity in identities or row["split"] != phase:
            raise ValueError("dataset has duplicate identities or the wrong split")
        identities.add(identity)
        length = len(row["input_ids"])
        if (row["input_tokens"] != length or row["max_new_tokens"] != 128 or length < 1 or
                length + 128 > 16384 or not row["references"]):
            raise ValueError("invalid final-token dataset length/output/reference policy")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in row["input_ids"]):
            raise ValueError("dataset input IDs must be nonnegative integers")
        if any(not 0 <= start < end <= length for start, end in row["gold_spans"]):
            raise ValueError("dataset gold span exceeds final chat tokens")
    return rows, manifest, path


def finalize_run(store, rows):
    """Publish aggregates and calibration selection only after every prompt commits."""
    expected = {store.prompt_key(row) for row in rows}
    if len(expected) != len(rows) or expected != store._committed.keys():
        raise ValueError("cannot finalize before the complete exact prompt cohort is committed")
    complete_path = store.output / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["run_id"] != store.run_id:
            raise ValueError("completion configuration mismatch")
        for name, digest in complete["files_sha256"].items():
            if _sha(store.output / name) != digest:
                raise ValueError("completed aggregate checksum mismatch")
        return complete
    summaries = []
    predictions_count, cost_files, cost_matrices = 0, 0, 0
    temporary = store.output / f".predictions-{uuid.uuid4().hex}.jsonl.gz"
    with gzip.open(temporary, "xt", encoding="utf-8") as stream:
        for row in rows:
            directory = store.output / "prompts" / store.prompt_key(row)
            predictions = json.loads((directory / "predictions.json").read_text())
            summaries.extend(json.loads((directory / "summaries.json").read_text()))
            for prediction in predictions:
                stream.write(_json_bytes(prediction).decode() + "\n")
            predictions_count += len(predictions)
            for path in sorted((directory / "cost_mass").glob("*.npz")):
                with np.load(path, allow_pickle=False) as payload:
                    cost_matrices += int(payload["mass"].shape[0])
                cost_files += 1
    os.replace(temporary, store.output / "per_prompt_predictions.jsonl.gz")
    _write_json(store.output / "per_prompt_shadow.json", summaries)
    if store.config["phase"] == "calibration":
        provenance = dict(quantizer=store.config["quantizer"], calibration_run_id=store.run_id,
                          calibration_config_sha256=_sha(store.output / "config.json"),
                          calibration_data_sha256=store.config["data_sha256"],
                          dataset_manifest_sha256=store.config["dataset_manifest_sha256"],
                          code_sha256=store.config["code_sha256"], source_sha256=store.config["source_sha256"],
                          model_file_sha256=store.config["model_file_sha256"],
                          quantizer_engineering_selection_sha256=store.config["quantizer_engineering_selection_sha256"])
        selection_path = store.output / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if (selection.get("provenance") != provenance or
                    selection.get("calibration_prompt_metrics_sha256") != _digest(summaries)):
                raise ValueError("existing candidate selection differs from this calibration")
        else:
            freeze_candidate(summaries, expected_ids={(row["task"], row["sample_id"]) for row in rows},
                             provenance=provenance, output=selection_path)
    artifacts = ["config.json", "per_prompt_predictions.jsonl.gz", "per_prompt_shadow.json"]
    if store.config["phase"] == "calibration":
        artifacts.append("selection.json")
    no_decode = {(row["task"], row["sample_id"]) for row in summaries if row.get("no_decode_observations")}
    complete = dict(status="complete", run_id=store.run_id, completed_utc=_utc(),
                    phase=store.config["phase"], completed_prompts=len(rows),
                    completed_generations=predictions_count, shadow_prompt_summaries=len(summaries),
                    raw_shadow_records=sum(row.get("n_records", 0) for row in summaries),
                    no_decode_prompts=len(no_decode),
                    no_decode_convention="policy-level zero perturbation; no observed head quantile",
                    cost_npz_files=cost_files, cost_group_matrices=cost_matrices,
                    files_sha256={name: _sha(store.output / name) for name in artifacts},
                    prompt_bundle_sha256={key: _sha(store.output / "prompts" / key / "bundle.json")
                                          for key in sorted(store._committed)})
    _write_json(complete_path, complete, replace=False)
    store.progress("complete")
    return complete


def _verified_model(model_path, dataset_manifest):
    source = json.loads((model_path / "source_manifest.json").read_text())
    if not source.get("verified"):
        raise ValueError("local model source manifest is not verified")
    hashes = {}
    for entry in source["files"]:
        actual = _sha(model_path / entry["Path"])
        if actual != entry["Sha256"]:
            raise ValueError(f"local model hash mismatch: {entry['Path']}")
        hashes[entry["Path"]] = actual
    if any(hashes.get(name) != digest for name, digest in dataset_manifest["tokenizer"].items()):
        raise ValueError("dataset tokenizer and local model hashes differ")
    architecture = json.loads((model_path / "config.json").read_text())
    if architecture["num_hidden_layers"] != 36 or architecture["num_key_value_heads"] != 8:
        raise ValueError("formal Round-2 model must have 36 layers and eight KV groups")
    return source, hashes


def _run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ("output", "model", "data_dir", "quantizer_selection"):
        setattr(args, name, getattr(args, name).resolve())
    if args.selection is not None:
        args.selection = args.selection.resolve()
    code_names = ["__init__.py", "ruler_experiment.py", "ruler_inference.py", "paired_statistics.py",
                  "oracle_attention.py", "torch_selection.py", "quantization_round2.py", "capture.py",
                  "attention.py", "selection.py"]
    package = Path(__file__).resolve().parent
    payloads = {f"gqa_probe/{name}": (package / name).read_bytes() for name in code_names}
    code_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    source_hashes = dict(code_hashes, **{"resume_launcher.py": hashlib.sha256(_LAUNCHER.encode()).hexdigest()})
    old_config_path = args.output / "config.json"
    if old_config_path.exists():
        if not args.resume:
            raise FileExistsError("config.json exists; explicit --resume is required")
        old = json.loads(old_config_path.read_text())
        require_frozen_code(args, code_hashes, old.get("code_sha256"), args.output)
        verify_snapshot(args.output, old["source_sha256"])
    selection = None
    if args.phase == "test":
        selection = json.loads(args.selection.read_text())
        validate_selection(selection, args.quantizer)
        provenance = selection.get("provenance", {})
        require_frozen_code(args, code_hashes, provenance.get("code_sha256"), args.selection.parent)
        verify_snapshot(args.selection.parent, provenance["source_sha256"])
    load_quantizer_selection(args.quantizer_selection, args.quantizer)
    quantizer_selection_sha = _sha(args.quantizer_selection)
    if selection is not None and selection["provenance"].get("quantizer_engineering_selection_sha256") != quantizer_selection_sha:
        raise ValueError("frozen quantizer engineering selection hash differs")
    # Validate selection against manifest metadata before opening held-out rows.
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if selection is not None:
        validate_selection(selection, args.quantizer, manifest=manifest)
        if selection["provenance"].get("dataset_manifest_sha256") != _sha(manifest_path):
            raise ValueError("selection dataset manifest checksum differs")
    rows, manifest, data_path = load_dataset(args.data_dir, args.phase)
    if manifest["configuration"]["tasks"] != TASKS:
        raise ValueError("formal dataset must contain exactly the four frozen tasks")
    logging.info("Verifying local model files before one model load")
    model_source, model_hashes = _verified_model(args.model, manifest)
    if selection is not None:
        validate_selection(selection, args.quantizer, code_sha256=code_hashes, model_sha256=model_hashes)
    import torch
    import transformers
    from .quantization_round2 import quantization_spec, prefill_for_quantization
    from .ruler_inference import prefill_model, generate_from_prefill

    cohort, visited = [], Counter()
    if args.phase == "test":
        for row in rows:
            if visited[row["task"]] < 16:
                cohort.append(dict(task=row["task"], sample_id=row["sample_id"]))
            visited[row["task"]] += 1
    configuration = dict(
        schema_version=1, phase=args.phase, quantizer=args.quantizer, quantization=quantization_spec(args.quantizer),
        quantizer_engineering_selection_sha256=quantizer_selection_sha,
        quantizer_engineering_config_sha256=_sha(args.quantizer_selection.parent / "config.json"),
        quantizer_engineering_selection_path=str(args.quantizer_selection),
        dataset_path=str(data_path), data_sha256=manifest["splits"][args.phase]["sha256"],
        dataset_manifest_sha256=_sha(manifest_path), generator=manifest["generator"], corpus=manifest["corpus"],
        model=str(args.model), model_source=model_source, model_file_sha256=model_hashes,
        model_source_manifest_sha256=_sha(args.model / "source_manifest.json"),
        code_sha256=code_hashes, source_sha256=source_hashes,
        model_dtype="bfloat16", prefill="dense_sdpa", decode="grouped_fp32_qk_softmax_av",
        all_layers=36, all_kv_groups=8, page_size=32, sink_pages=1, local_tokens=128,
        max_new_tokens=128, generation="greedy_eos_aware_including_prefill_first_token",
        sample_layers=SAMPLE_LAYERS, sample_steps=SAMPLE_STEPS,
        shadows=shadow_specs(args.phase, selection), cost_cohort=cohort, cost_step=0,
        cost_mass_axes=["kv_group", "page", "query_head"],
        tf32=False, torch_threads=8, seed=20261118, python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__,
        expected_prompts=len(rows), predictions_per_prompt=2 if args.phase == "calibration" else 6,
        expected_generations=len(rows) * (2 if args.phase == "calibration" else 6),
        selection_sha256=_sha(args.selection) if args.selection is not None else None,
        selection_path=str(args.selection) if args.selection is not None else None,
        prefill_timing_policy="one BF16 prefill and one shared KV4 prefill per prompt; reused timing fields must not be summed",
        timing_interpretation="instrumented Oracle numeric reference; audit time includes cost mass export; no system speed claim")
    store = RunStore(args.output, configuration, resume=args.resume)
    if not args.resume:
        if snapshot_sources(store.output, payloads) != source_hashes:
            raise ValueError("source changed while freezing the run")
    verify_snapshot(store.output, source_hashes)
    if (store.output / "complete.json").exists():
        return finalize_run(store, rows)
    if not torch.cuda.is_available():
        raise RuntimeError("formal RULER experiment requires CUDA")
    store.progress("loading_model")
    torch.set_num_threads(8)
    torch.manual_seed(20261118)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        low_cpu_mem_usage=True, local_files_only=True).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    runtime = ExperimentRuntime(model, tokenizer, eos_ids, prefill_model, prefill_for_quantization, generate_from_prefill)
    cost_ids = {(row["task"], row["sample_id"]) for row in cohort}
    try:
        for row in rows:
            stage = store.begin_prompt(row)
            if stage is None:
                continue
            predictions, summaries = execute_prompt(
                runtime, row, phase=args.phase, quantizer=args.quantizer, stage=stage,
                run_id=store.run_id, selection=selection, cost_sample=(row["task"], row["sample_id"]) in cost_ids)
            store.commit_prompt(row, stage, predictions=predictions, summaries=summaries)
            progress = store.progress("running")
            logging.info("Completed prompts %s/%s; generations %s/%s", progress["completed_prompts"],
                         len(rows), progress["completed_generations"], configuration["expected_generations"])
        complete = finalize_run(store, rows)
        logging.info("Completed %s: prompts=%s generations=%s", args.phase,
                     complete["completed_prompts"], complete["completed_generations"])
        return complete
    except BaseException as exc:
        store.progress("interrupted", error_type=type(exc).__name__, error=str(exc))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantizer", choices=("affine32", "symmetric16"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--data-dir", type=Path, default=Path("results/ruler_round2_data"))
    parser.add_argument("--quantizer-selection", type=Path,
                        default=Path("results/ruler_round2_quantization_engineering/selection.json"))
    args = parser.parse_args(argv)
    if args.phase == "test" and (args.selection is None or not args.selection.is_file()):
        raise ValueError("test phase requires an existing frozen --selection JSON")
    return _run(args)


if __name__ == "__main__":
    main()
