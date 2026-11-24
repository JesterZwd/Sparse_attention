"""Run the three Round 11 work points on common prefixes and independent prompts."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re

import numpy as np
import torch
import transformers

from gqa_probe.compact_summary import CompactOnline, SUMMARY_BYTES, weights
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.paired_statistics import paired_bootstrap, score_prediction
from gqa_probe.round7_attention import Attention
from gqa_probe.round11_trace import CommonAttention, SAMPLE_LAYERS, SAMPLE_STEPS
from gqa_probe.round11_workpoints import A, B, C, M, BASELINE, GATES, Workpoint, allocation, verdict
from gqa_probe.ruler_experiment import _sha, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.summary_decode import screen
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DENSE, make_answer
from scripts.run_round10 import (ROOT as ROUND10, answer_spans, init_model, prefill, prefix_layers,
                                 prompt_bundle, read_bundle, read_masks, seal, sources, summarize_predictions)
from scripts.run_ruler_round3 import load_rows


ROOT = Path("results/round11_workpoints")
BASE = "a52d1205d895e6c9d083dca4203f4bda263c6c38"
CALIBRATION = (A, B, C)
ARITHMETIC = ("fused_updates", "qk_mac", "pv_mac")
MID_QK_PV_LIMIT = 1.10
RECORDS = ("scanned_pages", "k_pages", "prefix_pages", "k_records", "v_records",
           "fallback_prefix_records", "summary_append_bytes", "summary_prefill_bytes")
PHYSICAL = ("k_code_bytes", "k_scale_bytes", "k_zero_point_bytes", "k_alignment_bytes", "v_code_bytes",
            "v_scale_bytes", "v_zero_point_bytes", "v_alignment_bytes", "summary_alignment_bytes", "index_bytes")
METRICS = ("official_score", "complete_correct", "generated_tokens", "truncated", "read_per_decode", "read",
           "qk_mac", "pv_mac", "fused_updates")


def source_hashes():
    paths = ("scripts/run_round11.py", "gqa_probe/round11_workpoints.py", "gqa_probe/round11_trace.py")
    return {**sources(), **{p: _sha(Path(p)) for p in paths}}


def gold_spans(row):
    """Use value-specific spans on Multi-Value, if the task has them."""
    return [item["value_span"] for item in row["source_metadata"].get("gold_evidence", ())
            if "value_span" in item]


class WorkpointAttention(Attention):
    def __init__(self, policy, prefixes, masks, gold, *, allocate=allocation):
        super().__init__(DENSE, None, gold)
        self.workpoint, self.prefixes, self.masks = policy, prefixes, masks
        self.allocate = allocate

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        policy = self.workpoint
        groups, tokens, dim = key.shape[1:]
        q = query.reshape(groups, -1, dim).float()
        heads = q.shape[1]
        layer = module.layer_idx
        scale = kwargs.get("scaling") or dim ** -.5
        history = page_history(tokens, policy.page)
        if layer not in self.online:
            self.online[layer] = CompactOnline(self.prefixes[layer])
        record = self.online[layer].append(key[0, :, -1], tokens)[policy.method]
        channel_layers = self.masks.get(policy.method)
        channels = None if channel_layers is None else channel_layers[layer]
        score = weights(q, record, policy.method, scale, channels=channels)
        alloc = self.allocate(policy, ranked_pages(score, history), history)
        out = screen(alloc, q, key, value, scale, gold_spans=self.gold_spans)
        for name, field in out.ledger.items():
            self._add("read" if name == "total" else name, field)
        self._add("decode_layer_calls", 1)
        self._add("kv_group_calls", groups)
        self._add("dense_comparator_bytes", groups * alloc.dense_bytes)
        self._add("cap_bytes", groups * alloc.cap)
        self._add("over_budget_group_steps", out.ledger["over_bytes"] > 0)
        self._add("fallback_group_steps", out.fallback)
        self._add("fallback_prefix_records", (alloc.full_mask(out.prefix).sum((1, 2)) * out.fallback))
        self._add("k_pages", alloc.count)
        self._add("prefix_pages", out.prefix)
        self._add("qk_mac", out.ledger["k_records"] * heads * dim)
        self._add("pv_mac", out.final_mask.sum() * dim)
        self._add("scanned_pages", groups * history.pages)
        updates = groups * heads * history.pages * (dim if channels is None else channels.shape[-1])
        self._add("scorer_terms", 2 * updates)
        self._add("fused_updates", updates)
        self._add("summary_append_bytes", groups * SUMMARY_BYTES[policy.method])
        if self.step == 0:
            self._add("summary_prefill_bytes", groups * ((tokens - 1) // policy.page) * SUMMARY_BYTES[policy.method])
        self._add("evidence_page_total", groups * out.evidence["gold_pages"])
        self._add("evidence_token_total", groups * out.evidence["gold_tokens"])
        for name, field in out.evidence.items():
            if isinstance(field, torch.Tensor):
                self._add(f"evidence_{name}", field)
        peak = alloc.count.max()
        self.peak_pages = peak if self.peak_pages is None else torch.maximum(self.peak_pages, peak)
        for name, loss in (("lost_all", out.loss_read_k), ("lost_prefix", out.loss_prefix)):
            self._add(f"{name}_sum", loss)
            high = loss.max()
            self.peak_loss[name] = high if name not in self.peak_loss else torch.maximum(self.peak_loss[name], high)
        self._add("query_head_calls", groups * heads)
        return self.output(query, out.output)


def group_result(policy, model, state, masks, tokenizer, eos, row, *, allocate=allocation):
    attention = WorkpointAttention(policy, prefix_layers(state, policy, masks), masks, gold_spans(row),
                                   allocate=allocate)
    run = generate_from_prefill(model, state, row["max_new_tokens"], eos, attention)
    answer = make_answer(policy, run, tokenizer, row, eos)
    raw = tokenizer.decode(run.token_ids, skip_special_tokens=True)
    clean = re.sub(r"[\x00-\x1f]", "\n", raw.strip()).strip()
    answer.update(score_prediction(clean, row["references"]))
    answer.update(task=row["task"], sample_id=row["sample_id"], generated_tokens=len(run.token_ids))
    if answer["kv_group_calls"] != answer["decode_tokens"] * model.config.num_hidden_layers * model.config.num_key_value_heads:
        raise ValueError("incomplete group-step accounting")
    if answer.get("over_budget_group_steps", 0) or answer["read"] > answer.get("cap_bytes", 0):
        raise ValueError("closed-loop route exceeded cap")
    if answer.setdefault("fused_updates", 0) * 2 != answer.get("scorer_terms", 0):
        raise ValueError("scorer work arithmetic mismatch")
    return answer


def summarize(predictions):
    output = summarize_predictions(predictions)
    for item in output:
        rows = [r for r in predictions if r["policy"] == item["policy"]]
        def total(key):
            return sum(r.get(key, 0) for r in rows)
        n = len(rows)
        steps = total("decode_tokens")
        heads = total("query_head_calls")
        item.update(generated_tokens_per_answer=total("generated_tokens") / n,
                    truncated_rate=total("truncated") / n, decode_tokens_per_answer=steps / n,
                    fused_updates_per_decode=total("fused_updates") / steps,
                    arithmetic_per_decode=sum(total(m) for m in ARITHMETIC) / steps,
                    qk_pv_per_decode=(total("qk_mac") + total("pv_mac")) / steps,
                    lost_read_k_mean=total("lost_all_sum") / heads,
                    lost_prefix_mean=total("lost_prefix_sum") / heads,
                    lost_read_k_max=max(r["lost_all_max"] for r in rows),
                    lost_prefix_max=max(r["lost_prefix_max"] for r in rows),
                    max_k_pages=max(r["peak_k_pages"] for r in rows),
                    group_steps_with_fallback=total("fallback_group_steps"))
        item.update({f"{name}_per_decode": total(name) / steps for name in (*PHYSICAL, *RECORDS)})
    return output


def pair_reports(output, predictions, policies, summary):
    by = {r["policy"]: r for r in summary}
    comparisons = []
    for new in policies:
        if new.name == "A":
            continue
        ref = A
        refs = [A] + ([B] if new.name == "C" and B in policies else [])
        for ref in refs:
            a, b = by[ref.name], by[new.name]
            paired = {m: paired_bootstrap(predictions, ref.name, new.name, m, method_key="policy",
                                          confidence=.95) for m in METRICS}
            pairs = {p["sample_id"]: p for p in predictions if p["policy"] == ref.name}
            deltas = []
            for candidate in (p for p in predictions if p["policy"] == new.name):
                prior = pairs[candidate["sample_id"]]
                delta = {m: candidate[m] - prior[m] for m in METRICS}
                deltas.append(dict(sample_id=candidate["sample_id"], deltas=delta))
            score = paired["official_score"]
            gate = verdict(new, a, b, score["point_delta"], score["ci_low"]) if ref == A else None
            comparisons.append(dict(reference=ref.name, candidate=new.name,
                                    algorithm_gate=gate, intervals=paired, prompt_deltas=deltas))
    _write_json(output / "paired.json", comparisons)
    return comparisons


def model_info(root, cohort, split="calibration"):
    path = root / "data" / cohort
    manifest, _, _ = load_rows(path, split)
    return init_model(manifest)


def split_inventory(path, split, count=None):
    manifest, length, rows = load_rows(path, split)
    rows = rows[:count]
    return dict(length=length, manifest_sha256=_sha(path / "manifest.json"),
                dataset_sha256=manifest["splits"][split]["sha256"],
                sample_ids=[r["sample_id"] for r in rows],
                instance_sha256=[r["source_metadata"]["instance_sha256"] for r in rows]), rows


def phase_config(data, split, rows_count, model_sha, policies):
    inventory, rows = split_inventory(data, split, rows_count)
    config = dict(dataset=inventory, split=split, source_sha256=source_hashes(), base=BASE,
                  model_sha256=model_sha, frozen_mask_sha256=_sha(ROUND10 / "channels.json"),
                  policies=list(map(asdict, policies)), prefill_chunk=2048,
                  torch=torch.__version__, transformers=transformers.__version__, device=torch.cuda.get_device_name())
    return config, rows


def run_phase(data, output, policies, runtime, *, split="calibration", count=None, freeze=None):
    model, tokenizer, eos, model_sha = runtime
    config, rows = phase_config(data, split, count, model_sha, policies)
    if split == "test" and freeze != config:
        raise ValueError("held-out execution differs from freeze")
    output.mkdir(parents=True, exist_ok=True)
    cp = output / "config.json"
    seal(cp, config)
    masks = read_masks(ROUND10, "cuda")
    predictions = []
    receipts = {}
    for index, row in enumerate(rows):
        path = output / "bundles" / f"{index:04d}.json.gz"
        if not path.exists():
            state = prefill(model, row)
            entries = [group_result(p, model, state, masks, tokenizer, eos, row) for p in policies]
            prompt_bundle(path, entries)
            del state
        bundle = read_bundle(path)
        if [p["policy"] for p in bundle] != [p.name for p in policies] or any(
                r["sample_id"] != row["sample_id"] for r in bundle):
            raise ValueError("prompt bundle configuration mismatch")
        predictions.extend(bundle)
        receipts[path.name] = _sha(path)
        print(f"{output.parent.name} {output.name} {index + 1}/{len(rows)}", flush=True)
    result = summarize(predictions)
    if A in policies:
        pair_reports(output, predictions, policies, result)
    write_csv(output / "quality_cost.csv", result)
    _write_json(output / "predictions.json", predictions)
    _write_json(output / "complete.json", dict(config_sha256=_sha(cp), predictions=len(predictions), bundles=receipts))
    return predictions, result


def read_json(path):
    return json.loads(path.read_text())


def calibration32(root):
    data = root / "data" / "32k"
    output = root / "32k" / "calibration"
    runtime = model_info(root, "32k")
    preds, rows = run_phase(data, output, CALIBRATION, runtime)
    by = {r["policy"]: r for r in rows}
    preliminary = read_json(output / "paired.json")
    b = next(x for x in preliminary if x["candidate"] == "B" and x["reference"] == "A")
    c = by["C"]["qk_pv_per_decode"] / by["A"]["qk_pv_per_decode"]
    quality_touch = b["intervals"]["official_score"]["ci_low"] < GATES["quality_margin"]
    run_mid = quality_touch or c > MID_QK_PV_LIMIT
    extras = []
    m = None
    if run_mid:
        more, _ = run_phase(data, root / "32k" / "middle_calibration", (M,), runtime)
        preds += more
        all_rows = summarize(preds)
        by = {r["policy"]: r for r in all_rows}
        dest = root / "32k" / "pooled_calibration"
        dest.mkdir(exist_ok=True)
        write_csv(dest / "quality_cost.csv", all_rows)
        _write_json(dest / "predictions.json", preds)
        comparisons = pair_reports(dest, preds, (*CALIBRATION, M), all_rows)
        m = next(x for x in comparisons if x["candidate"] == "M")
        if (m["intervals"]["official_score"]["point_delta"] >= GATES["min_quality_gain"] and
                by["M"]["qk_pv_per_decode"] < by["C"]["qk_pv_per_decode"]):
            extras.append(M.name)
    _write_json(root / "32k" / "calibration_decision.json",
                dict(b_quality_touch=quality_touch, c_qk_pv_ratio=c, c_qk_pv_limit=MID_QK_PV_LIMIT,
                     middle_calibrated=run_mid, middle_score_delta=m["intervals"]["official_score"]["point_delta"]
                     if m else None, common_policies=["A", "B", "C"] + (["M"] if run_mid else []),
                     independent_policies=["A", "B", "C", *extras]))


def trace_summary(metrics):
    by = defaultdict(list)
    for row in metrics:
        for scope in ("all", "value_start") if row["answer_start"] else ("all",):
            by[row["policy"], scope].append(row)
    rows = []
    for (policy, scope), obs in sorted(by.items()):
        groups = sum(len(r["pages"]) for r in obs)
        def total(field):
            return sum(sum(o[field]) for o in obs)
        def frac(metric, denom):
            d = sum(len(o["pages"]) * o[denom] for o in obs)
            return total(metric) / d if d else None
        def percentile(metric, quantile):
            return float(np.quantile(np.concatenate([np.ravel(o[metric]) for o in obs]), quantile))
        rows.append(dict(policy=policy, scope=scope, groups=groups,
                         k_page_recall=frac("k_page_hits", "gold_pages"),
                         final_k_page_recall=frac("final_k_page_hits", "gold_pages"),
                         v_evidence_token_recall=frac("v_gold_token_hits", "gold_tokens"),
                         initial_v_evidence_token_recall=frac("initial_v_gold_token_hits", "gold_tokens"),
                         absolute_p99=percentile("abs_error", .99), relative_p99=percentile("rel_error", .99),
                         read_k_loss_p99=percentile("read_k_loss", .99),
                         final_prefix_loss_p99=percentile("final_prefix_loss", .99),
                         final_prefix_loss_max=percentile("final_prefix_loss", 1),
                         overrun_groups=sum(sum(x > 0 for x in o["over_bytes"]) for o in obs),
                         **{f"{k}_per_group": total(k) / groups for k in
                            ("total", "summary_bytes", "k_bytes", "v_bytes", "k_records", "v_records",
                             "qk_mac", "pv_mac", "pages", "prefix", "fallback")}))
    return rows


def common_states(root, cohort, names, count, tag="common"):
    policy = tuple(BASELINE[p] for p in names)
    data = root / "data" / cohort
    model, tokenizer, eos, model_sha = model_info(root, cohort)
    config, rows = phase_config(data, "calibration", count, model_sha, policy)
    out = root / cohort / tag
    out.mkdir(parents=True, exist_ok=True)
    config.update(sample_layers=list(SAMPLE_LAYERS), sample_steps=list(SAMPLE_STEPS), full_sample_rows=3)
    seal(out / "config.json", config)
    masks = read_masks(ROUND10, "cuda")
    diag = []
    per_prompt = []
    traces = []
    bundles = {}
    for index, row in enumerate(rows):
        path = out / "bundles" / f"{index:04d}.json.gz"
        if not path.exists():
            state = prefill(model, row)
            inputs = (model, state, row["max_new_tokens"], eos)
            dense = generate_from_prefill(*inputs, Attention(DENSE, None, row["gold_spans"]))
            starts = answer_spans(tokenizer, dense.token_ids, row)
            full_step = None
            if index < 3:
                candidates = [0, *sorted(starts)]
                full_step = candidates[min(index, len(candidates) - 1)]
            attn = CommonAttention(DENSE, policy, masks, gold_spans(row), starts, row["sample_id"],
                                   cohort, index, full_step)
            replay = generate_from_prefill(*inputs, attn)
            if dense.token_ids != replay.token_ids:
                raise ValueError("common-state observation changed dense continuation")
            prompt_bundle(path, dict(sample_id=row["sample_id"], full_step=full_step,
                                     metrics=attn.metrics, traces=attn.traces))
            del state, inputs, attn
        bundle = read_bundle(path)
        diag.extend(bundle["metrics"])
        traces.extend(bundle["traces"])
        per_prompt.extend(dict(sample_id=row["sample_id"], **x) for x in trace_summary(bundle["metrics"]))
        bundles[path.name] = _sha(path)
        print(f"{cohort} common {index + 1}/{len(rows)}", flush=True)
    if "B" in names:
        ab = [x for x in traces if x["policy"] in ("A", "B")]
        by = defaultdict(dict)
        for x in ab:
            by[x["sample_id"], x["step"], x["layer"], x["group"]][x["policy"]] = x
        for a, b in ((p["A"], p["B"]) for p in by.values()):
            assert a["initial_prefix"] == b["initial_prefix"] and a["qk_mac"] == b["qk_mac"]
    path = out / "states.jsonl.gz"
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt", compresslevel=4) as handle:
        for record in traces:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)
    write_csv(out / "same_state.csv", trace_summary(diag))
    write_csv(out / "per_prompt.csv", per_prompt)
    _write_json(out / "complete.json", dict(config_sha256=_sha(out / "config.json"), metric_rows=len(diag),
                                            trace_rows=len(traces), states_sha256=_sha(path), bundles=bundles))


def freeze(root, cohort, policies, model_sha, hardware=None):
    output = root / cohort
    data = root / "data" / cohort
    config, _ = phase_config(data, "test", None, model_sha, policies)
    cal = output / "calibration" / "complete.json"
    common = output / "common" / "complete.json"
    plan = dict(test=config, gates=GATES, prefill_chunk=2048, calibration_sha256=_sha(cal),
                common_sha256=_sha(common), hardware_screen=hardware,
                comparisons=["B-A", "C-A", "C-B", "M-A"])
    seal(output / "frozen_test.json", plan)
    return plan


def independent(root, cohort):
    output = root / cohort
    data = root / "data" / cohort
    if cohort == "32k":
        runtime = model_info(root, cohort)
        names = read_json(output / "calibration_decision.json")["independent_policies"]
        opts = tuple(BASELINE[name] for name in names)
        plan = freeze(root, cohort, opts, runtime[3])
    else:
        frozen = output / "frozen_test.json"
        if not frozen.is_file():
            return
        plan = read_json(frozen)
        opts = tuple(Workpoint(**p) for p in plan["test"]["policies"])
        runtime = model_info(root, cohort, "test")
    _, results = run_phase(data, output / "independent", opts, runtime,
                           split="test", freeze=plan["test"])
    pairs = read_json(output / "independent" / "paired.json")
    _write_json(output / "decision.json", dict(policies=[p.name for p in opts],
                candidates={r["candidate"]: r["algorithm_gate"] for r in pairs if r["reference"] == "A"},
                sparse_group_steps=sum(x["group_steps"] for x in results),
                overrun_groups=sum(x["overrun_groups"] for x in results)))


def admitted(root):
    outcomes = read_json(root / "32k" / "decision.json")["candidates"]
    names = ["B"] if outcomes["B"]["passes"] else []
    # Prefer the lighter, qualified +4 quality point if one was frozen.
    if outcomes.get("M", {}).get("passes"):
        names.append("M")
    elif outcomes["C"]["passes"]:
        names.append("C")
    return names


def calibration64(root):
    output = root / "64k"
    output.mkdir(exist_ok=True)
    names = admitted(root)
    if not names:
        _write_json(output / "entry.json", dict(eligible=[], reason="32K quality and physical gates"))
        return
    data = root / "data" / "64k"
    opts = (A, *(BASELINE[name] for name in names))
    runtime = model_info(root, "64k")
    _, table = run_phase(data, output / "calibration", opts, runtime)
    _write_json(output / "entry.json", dict(eligible=names, model_sha256=runtime[3], table=table))
    del runtime
    torch.cuda.empty_cache()
    common64(root)


def common64(root):
    output = root / "64k"
    entry = read_json(output / "entry.json")
    if not entry["eligible"]:
        return
    pairs = read_json(output / "calibration" / "paired.json")
    quality = {}
    for pair in pairs:
        if pair["reference"] != "A":
            continue
        delta = pair["intervals"]["official_score"]["point_delta"]
        name = pair["candidate"]
        gate = pair["algorithm_gate"]
        read_limit = 1 - GATES["min_read_saving"] if name == "B" else GATES["max_quality_read_ratio"]
        # The small transfer calibration checks usable quality. The held-out
        # test applies the unchanged noninferiority or quality-gain CI gate.
        quality[name] = (gate["zero_overruns"] and delta >= GATES["quality_margin"] and
                         gate["read_ratio"] <= read_limit)
    entry.update(calibrated=quality, calibration_point_margin=GATES["quality_margin"],
                 calibration_paired_sha256=_sha(output / "calibration" / "paired.json"))
    _write_json(output / "entry.json", entry)
    common_states(root, "64k", ["A", *entry["eligible"]], count=8)


def cross_task(root):
    names = admitted(root)
    if not names:
        # Evaluate the cap-matched probe for transfer diagnostics even if 32K
        # does not support a claim. Neither task is used to fit the channel mask.
        names = ["B"]
    policies = (A, *(BASELINE[name] for name in names))
    cohort = "32k_vt"
    output = root / "cross_task"
    output.mkdir(exist_ok=True)
    runtime = model_info(root, cohort)
    _, rows = run_phase(root / "data" / cohort, output / cohort, policies, runtime)
    saturated = min(r["official_score"] for r in rows) >= .98
    if saturated:
        _, fwe = run_phase(root / "data" / "16k_fwe", output / "16k_fwe", policies, runtime)
    else:
        fwe = None
    _write_json(output / "decision.json", dict(vt_saturated=saturated, vt=rows, fwe=fwe, policies=[p.name for p in policies]))


def regression(root):
    source = ROUND10 / "32k" / "independent" / "predictions.json"
    data = ROUND10 / "data" / "32k"
    model = init_model(load_rows(data, "calibration")[0])
    # Already executed in Round 10. Switch split to a dedicated audited
    # regression path; these IDs never enter Round 11 test or calibration.
    inventory, rows = split_inventory(data, "test", 4)
    out = root / "regression"
    out.mkdir(exist_ok=True)
    masks = read_masks(ROUND10, "cuda")
    net, tok, eos, model_sha = model
    old = {p["sample_id"]: p for p in read_json(source) if p["policy"] == "mr96_score_112"}
    keys = ("official_score", "complete_correct", "read", "qk_mac", "pv_mac", "scorer_terms", "decode_tokens",
            "fallback_group_steps")
    matches = []
    for row in rows:
        state = prefill(net, row)
        now = group_result(C, net, state, masks, tok, eos, row)
        prior = old[row["sample_id"]]
        if any(now[k] != prior[k] for k in keys):
            raise ValueError(f"Round 10 C regression changed {row['sample_id']}")
        matches.append(dict(sample_id=row["sample_id"], **{k: now[k] for k in keys}))
        del state
    _write_json(out / "complete.json", dict(prior_sha256=_sha(source), inventory=inventory,
                                           model_sha256=model_sha, identical_fields=keys, prompts=matches))


def main():
    options = ("check32", "cal32", "common32", "test32", "cal64", "common64", "test64", "cross", "regression")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=options)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    with torch.inference_mode():
        if args.phase == "check32":
            common_states(root, "32k", ["A", "B", "C"], count=2, tag="precheck")
        elif args.phase == "cal32":
            calibration32(root)
        elif args.phase == "common32":
            dec = read_json(root / "32k" / "calibration_decision.json")
            common_states(root, "32k", dec["common_policies"], count=12)
        elif args.phase == "test32":
            independent(root, "32k")
        elif args.phase == "cal64":
            calibration64(root)
        elif args.phase == "common64":
            common64(root)
        elif args.phase == "test64":
            independent(root, "64k")
        elif args.phase == "regression":
            regression(root)
        else:
            cross_task(root)


if __name__ == "__main__":
    main()
