"""Restore the Round 10 ranking and test narrower same-K summaries in Round 12."""
import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from gqa_probe.compact_summary import ChannelCalibration, records_as_bytes, score_inputs, scored_channels
from gqa_probe.paired_statistics import paired_bootstrap
from gqa_probe.round7_attention import Attention
from gqa_probe.round11_trace import CommonAttention
from gqa_probe.round11_workpoints import A, B, C, M
from gqa_probe.round12_narrow import CANDIDATES, OPTS, same_k
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.scorer_int import integer_scores
from scripts.prepare_round11 import HISTORY, read_split
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DENSE
from scripts.run_round10 import ROOT as R10
from scripts.run_round10 import FitAttention, answer_spans, init_model, prefill, prompt_bundle, read_bundle, seal
from scripts.run_round11 import ROOT as R11
from scripts.run_round11 import group_result, gold_spans, read_json, source_hashes, split_inventory, summarize, trace_summary
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round12_frontend")
BASE = "10b287bbc24f83463ffacdd47164153778309c34"
DIAG_GATES = dict(v_evidence_loss_max=.01, p99_ratio_max=1.05)
TEST_GATES = dict(quality_ci_low_min=-.01, read_ratio_max=.95, zero_overruns=True)


def shared_config(root):
    paths = ("scripts/run_round12.py", "gqa_probe/round12_narrow.py")
    return dict(round11_base=BASE, sources={**source_hashes(), **{p: _sha(Path(p)) for p in paths}},
                channels_sha256=_sha(root / "channels.json"), prefill_chunk=2048)


def fit(root):
    output = root / "channels.json"
    source = R10 / "data" / "32k"
    baseline = R10 / "channels.json"
    frozen = read_json(baseline)
    manifest, _, rows = load_rows(source, "calibration")
    by_id = {r["sample_id"]: r for r in rows}
    model, _, eos, model_sha = init_model(manifest)
    cal = ChannelCalibration()
    for ident in frozen["ids"]:
        row = by_id[ident]
        state = prefill(model, row)
        generate_from_prefill(model, state, min(16, row["max_new_tokens"]), eos, FitAttention(cal))
        del state
    matching = {rule: {str(layer): ids.tolist() for layer, ids in by_layer.items()}
                for rule, by_layer in cal.freeze(96).items()}
    if matching != frozen["channels"]:
        raise ValueError("restored Top-96 differs from Round 10")
    widths = {f"mr{w}_score": {str(layer): ids.tolist() for layer, ids in cal.freeze(w)["score"].items()}
              for w in (96, 80, 64)}
    ranking = {}
    contribution = {}
    coverage = {}
    for layer, stats in cal.statistics.items():
        values = stats.score_abs
        order = values.argsort(dim=-1, descending=True, stable=True)
        ranking[str(layer)] = order.tolist()
        contribution[str(layer)] = values.tolist()
        sorted_values = values.gather(-1, order)
        coverage[str(layer)] = {str(w): (sorted_values[:, :w].sum(-1) / values.sum(-1).clamp_min(1e-12)).tolist()
                                for w in (96, 80, 64, 48)}
    digest = {str(w): [float(q) for q in np.quantile([x for row in coverage.values() for x in row[str(w)]], (0, .05, .5, 1))]
              for w in (96, 80, 64, 48)}
    seal(output, dict(frozen_top96_exact=True, local_round10_fit_ids=frozen["ids"],
                      original_mask_sha256=_sha(baseline), fit_manifest_sha256=_sha(source / "manifest.json"),
                      model_sha256=model_sha, ranked_ids=ranking, score_abs=contribution, masks=widths,
                      coverage_by_layer_group=coverage, coverage_min_p05_median_max=digest,
                      per_layer_observations={str(layer): stats.observations for layer, stats in cal.statistics.items()}))
    print("Round 10 score and key Top-96 masks match the refitted ranking.", flush=True)


def masks(root):
    return {m: {int(layer): torch.tensor(ids, device="cuda") for layer, ids in layers.items()}
            for m, layers in read_json(root / "channels.json")["masks"].items()}


def fixture_writer(root, sample_id, selected_step):
    loc = root / "local_inputs"
    loc.mkdir(parents=True, exist_ok=True)

    def export(layer, step, query, records, channels, scale):
        if layer not in (7, 21, 35) or step != selected_step:
            return
        group = layer % query.shape[0]
        q = query[group:group+1]
        g64 = ((q.abs().amax(-1) / 127).double() * scale)[0].cpu().numpy()
        for method, record in records.items():
            dim = scored_channels(method)
            take = [*range(16), record[0].shape[1]-1]
            rec = tuple(t[group:group+1, take] for t in record)
            dims = channels.get(method)
            data = score_inputs(q, rec, method, scale, channels=None if dims is None else dims[group:group+1])
            packed = records_as_bytes(rec, method)[0]
            np.savez_compressed(loc / f"d{dim:03d}_{sample_id.rsplit('/',1)[-1]}_l{layer:02d}_s{step:03d}.npz",
                                q8=data.query[0].cpu().numpy(), g64=g64, records=packed.cpu().numpy(),
                                scores=integer_scores(data)[0].T.cpu().numpy(),
                                coeff_a=data.coeff_a[0].T.cpu().numpy(), coeff_b=data.coeff_b[0].T.cpu().numpy(),
                                bias=data.bias[0,0].cpu().numpy(), page_ids=np.asarray(take))
    return export


def capture(root, cohort):
    data = R11 / "data" / cohort
    manifest, _, all_rows = load_rows(data, "calibration")
    policies = (A, B, M, *CANDIDATES) if cohort == "32k" else (A, B, M)
    if cohort == "64k":
        candidate = read_json(root / "selection.json")["selected"]
        if candidate:
            policies = (*policies, OPTS[candidate])
    rows = all_rows[3:15] if cohort == "32k" else all_rows[3:9]
    model, tokenizer, eos, model_sha = init_model(manifest)
    channel_masks = masks(root)
    output = root / cohort / "extended"
    output.mkdir(parents=True, exist_ok=True)
    config = dict(**shared_config(root), dataset_sha256=_sha(data / "manifest.json"), model_sha256=model_sha,
                  sample_ids=[row["sample_id"] for row in rows], policies=list(map(asdict, policies)),
                  full_steps=5 if cohort == "32k" else 3)
    seal(output / "config.json", config)
    metrics, rows_by_prompt, traces, receipts = [], [], [], {}
    for i, row in enumerate(rows):
        target = output / "bundles" / f"{i:04d}.json.gz"
        if not target.exists():
            state = prefill(model, row)
            args = model, state, row["max_new_tokens"], eos
            initial = generate_from_prefill(*args, Attention(DENSE, None, row["gold_spans"]))
            starts = answer_spans(tokenizer, initial.token_ids, row)
            last = len(initial.token_ids)-2
            beginnings = sorted(starts)
            full_opts = [0, next(iter(beginnings), 0), last, min(31, last), beginnings[-1] if beginnings else min(7, last)]
            whole = full_opts[i] if i < config["full_steps"] else None
            observer = fixture_writer(root, row["sample_id"], whole) if cohort == "32k" and i < 2 else None
            shared = CommonAttention(DENSE, policies, channel_masks, gold_spans(row), starts, row["sample_id"],
                                     cohort, i, whole, allocate=same_k,
                                     matched_pairs=(("A","B"),*(("M",p.name) for p in policies if p in CANDIDATES)),
                                     observer=observer)
            same = generate_from_prefill(*args, shared)
            if same.token_ids != initial.token_ids:
                raise ValueError("common-state capture changed Dense continuation")
            prompt_bundle(target, dict(sample_id=row["sample_id"], metrics=shared.metrics, traces=shared.traces,
                                       whole_step=whole))
            del state, args, shared
        obj = read_bundle(target)
        metrics.extend(obj["metrics"])
        traces.extend(obj["traces"])
        rows_by_prompt.extend(dict(sample_id=obj["sample_id"], **r) for r in trace_summary(obj["metrics"]))
        receipts[target.name] = _sha(target)
        print(f"{cohort} states {i+1}/{len(rows)}", flush=True)
    output_rows = trace_summary(metrics)
    path = output / "states.jsonl.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", compresslevel=4) as stream:
        for record in traces:
            stream.write(json.dumps(record, separators=(",",":"), allow_nan=False) + "\n")
    tmp.replace(path)
    write_csv(output / "same_state.csv", output_rows)
    write_csv(output / "per_prompt.csv", rows_by_prompt)
    _write_json(output / "complete.json", dict(config_sha256=_sha(output / "config.json"), rows=len(traces),
                                              state_sha256=_sha(path), bundle_receipts=receipts))
    if cohort == "32k":
        _write_json(root / "real_fixtures.json", dict(
            encoding="NPZ of real Q8, raw records, binary64 g, Q24 coefficients, bias and signed integer scores.",
            captures={p.name:_sha(p) for p in sorted((root / "local_inputs").glob("*.npz"))},
            split="Round 11 32K calibration, indices 3 and 4", config_sha256=_sha(output / "config.json")))
        vals = {r["policy"]: r for r in output_rows if r["scope"]=="value_start"}
        ref = vals["M"]
        evaluations = []
        for p in CANDIDATES:
            trial = vals[p.name]
            recall = trial["v_evidence_token_recall"] - ref["v_evidence_token_recall"]
            p99 = trial["absolute_p99"] / ref["absolute_p99"]
            evaluations.append(dict(policy=p.name, v_token_delta=recall, abs_p99_ratio=p99,
                                    same_k=trial["k_records_per_group"] == ref["k_records_per_group"],
                                    passes=recall >= -DIAG_GATES["v_evidence_loss_max"] and
                                    p99 <= DIAG_GATES["p99_ratio_max"] and not trial["overrun_groups"]))
        _write_json(root / "screen.json", dict(gates=DIAG_GATES, results=evaluations))


def phase(root, cohort, label, policies, *, independent=False):
    data = (root if independent else R11) / "data" / cohort
    split = "test" if independent else "calibration"
    manifest, _, _ = load_rows(data, split)
    inventory, rows = split_inventory(data, split)
    model, tokenizer, eos, model_sha = init_model(manifest)
    channel_masks = masks(root)
    output = root / cohort / label
    output.mkdir(parents=True, exist_ok=True)
    conf = dict(**shared_config(root), dataset=inventory, model_sha256=model_sha,
                policies=list(map(asdict, policies)), split=split)
    if independent:
        seal(root / cohort / "frozen_test.json",
             dict(config=conf, gates=TEST_GATES, selection_sha256=_sha(root / "selection.json"),
                  calibration_sha256=_sha(root / cohort / "calibration" / "complete.json")))
    seal(output / "config.json", conf)
    predictions, receipts = [], {}
    for i, row in enumerate(rows):
        p = output / "bundles" / f"{i:04d}.json.gz"
        if not p.exists():
            state = prefill(model, row)
            answers = [group_result(policy, model, state, channel_masks, tokenizer, eos, row, allocate=same_k)
                       for policy in policies]
            prompt_bundle(p, answers)
            del state
        answers = read_bundle(p)
        if [a["policy"] for a in answers] != [policy.name for policy in policies] or any(
                a["sample_id"] != row["sample_id"] for a in answers):
            raise ValueError("prompt results differ from config")
        predictions.extend(answers)
        receipts[p.name] = _sha(p)
        print(f"{cohort} {label} {i+1}/{len(rows)}", flush=True)
    results = summarize(predictions)
    by = {r["policy"]: r for r in results}
    base = policies[0]
    pairs = []
    for p in policies[1:]:
        intervals = {m: paired_bootstrap(predictions, base.name, p.name, m, method_key="policy", confidence=.95)
                     for m in ("official_score","complete_correct","read_per_decode","read","qk_mac","pv_mac","fused_updates")}
        score = intervals["official_score"]
        read = by[p.name]["read_per_decode"] / by[base.name]["read_per_decode"]
        budget_ok = not (by[p.name]["overrun_groups"] or by[base.name]["overrun_groups"])
        old = {x["sample_id"]:x for x in predictions if x["policy"]==base.name}
        deltas = [dict(sample_id=x["sample_id"], score_delta=x["official_score"]-old[x["sample_id"]]["official_score"],
                       read_delta=x["read"]-old[x["sample_id"]]["read"],
                       complete_reference=old[x["sample_id"]]["complete_correct"],complete_candidate=x["complete_correct"])
                  for x in predictions if x["policy"]==p.name]
        pairs.append(dict(reference=base.name, candidate=p.name, intervals=intervals, read_ratio=read,
                          zero_overruns=budget_ok, test_passes=budget_ok and read<=.95 and score["ci_low"]>=-.01,
                          calibration_eligible=budget_ok and read<=.95 and score["point_delta"]>=-.01,
                          prompt_deltas=deltas))
    write_csv(output / "quality_cost.csv", results)
    _write_json(output / "predictions.json", predictions)
    _write_json(output / "paired.json", pairs)
    _write_json(output / "complete.json", dict(config_sha256=_sha(output / "config.json"),
                                              predictions=len(predictions), bundles=receipts))
    return pairs


def narrow_calibration(root):
    opts = tuple(OPTS[r["policy"]] for r in read_json(root / "screen.json")["results"] if r["passes"])
    selected = None
    if opts:
        pairs = phase(root, "32k", "calibration", (M, *opts))
        eligible = [x for x in pairs if x["calibration_eligible"]]
        if eligible:
            selected = max(eligible, key=lambda x: (x["intervals"]["official_score"]["point_delta"],
                                                     x["candidate"] == "N80"))["candidate"]
    _write_json(root / "selection.json", dict(selected=selected, filter="Calibration noninferiority point and 5% saving; prefer 80 channels on score tie.",
                                             candidate_order=[p.name for p in opts], screen_sha256=_sha(root / "screen.json")))


def validate_independence(root):
    inputs = [Path("results") / x for x in HISTORY]
    inputs += [R11 / "data" / x for x in ("32k","64k","32k_vt","16k_fwe")]
    seen = set()
    for data in inputs:
        meta = read_json(data / "manifest.json")
        for split in meta["splits"]:
            _, rows, _ = read_split(data, split)
            seen.update(r["source_metadata"]["instance_sha256"] for r in rows)
    receipt = {}
    for cohort, n in (("32k",128),("64k",64)):
        inventory, rows = split_inventory(root / "data" / cohort, "test")
        identities = {r["source_metadata"]["instance_sha256"] for r in rows}
        if len(identities) != n or identities & seen:
            raise ValueError("Round 12 test overlaps known samples")
        seen.update(identities)
        receipt[cohort] = inventory
    _write_json(root / "data_receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit","capture32","cal32","data","cal64","cal_c","capture64","test32","test64"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root, stage = args.root, args.stage
    with torch.inference_mode():
        if stage == "fit":
            fit(root)
        elif stage.startswith("capture"):
            capture(root, stage[-2:]+"k")
        elif stage == "cal32":
            narrow_calibration(root)
        elif stage == "data":
            validate_independence(root)
        else:
            selected = read_json(root / "selection.json")["selected"]
            if not selected:
                return
            narrow = OPTS[selected]
            if stage == "cal64":
                phase(root, "64k", "calibration", (M,narrow))
            elif stage == "cal_c":
                phase(root, "32k", "calibration_c", (C,replace(narrow,name=f"C_{selected}",extra_pages=None)))
            elif stage == "test32" or read_json(root / "64k" / "calibration" / "paired.json")[0]["calibration_eligible"]:
                phase(root, stage[-2:]+"k", "independent", (M,narrow), independent=True)


if __name__ == "__main__":
    main()
