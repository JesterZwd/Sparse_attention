"""Run compact-summary RULER experiments with separate fitting and held-out cohorts."""
import argparse
from dataclasses import asdict, dataclass, replace
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from gqa_probe.chunked_prefill import prefill_chunked
from gqa_probe.compact_summary import ChannelCalibration, CompactOnline, CompactPrefix, SUMMARY_BYTES, weights
from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.independent_gate import independent_decision
from gqa_probe.oracle_attention import grouped_logits, grouped_output, torch_page_mass
from gqa_probe.page_read_budget import page_history
from gqa_probe.round7_attention import Attention
from gqa_probe.ruler_experiment import _sha, _verified_model, _write_json
from gqa_probe.ruler_inference import generate_from_prefill
from gqa_probe.summary_decode import plan, screen
from scripts.report_ruler_round3 import write_csv
from scripts.round7_run import DENSE, make_answer
from scripts.run_pagepack import pairing
from scripts.run_ruler_round3 import MODEL, load_rows


ROOT = Path("results/round10_summary_64k")
BASE = "945084ae1d20d0d85ade22ad69f8e2fb64c073db"
FIT = 4
LAYERS = (0, 7, 14, 21, 28, 35)
STEPS = (0, 1, 7, 15, 31, 63)
TOL = .01


@dataclass(frozen=True)
class Policy:
    name: str
    method: str
    page: int = 32
    budget: float | None = 5
    reserve: float = .80
    pages: int | None = None


Q4 = Policy("quest4_5_r65", "q4", reserve=.65)
MR = Policy("mr25_5_r80", "mr25")
STRUCTURES = (
    Policy("mean4_radius2_112", "mr4"),
    Policy("mr96_k_112", "mr96_k"),
    Policy("mr96_score_112", "mr96_score"),
    Policy("mr25_p64_144", "mr25", 64),
)


def seal(path, obj):
    if path.exists() and json.loads(path.read_text()) != obj:
        raise ValueError(f"Frozen input changed at {path}")
    _write_json(path, obj)


def sources():
    names = ["chunked_prefill", "compact_summary", "summary_decode", "scorer_int", "ruler_inference",
             "quantization_round2", "round7_gates", "page_read_budget", "fixed_k_v", "vskip_summary",
             "oracle_attention", "round7_attention", "paired_statistics", "independent_gate"]
    files = [Path("gqa_probe") / (name + ".py") for name in names]
    files += list(map(Path, ("scripts/run_round10.py", "scripts/run_ruler_round3.py", "scripts/round7_run.py")))
    return {str(p): _sha(p) for p in files}


def original_spans(row):
    return [item["value_span"] for item in row["source_metadata"]["gold_evidence"]]


def answer_spans(tokenizer, tokens, row):
    prefixes = [tokenizer.decode(tokens[:i + 1], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False).lower() for i in range(len(tokens))]
    starts = {}
    for item in row["source_metadata"]["gold_evidence"]:
        offset = prefixes[-1].find(item["value"].lower())
        if offset < 0:
            continue
        debut = next(i for i, text in enumerate(prefixes) if text[:offset + 1] == prefixes[-1][:offset + 1])
        if debut:
            starts.setdefault(debut - 1, []).append(item["value_span"])
    return starts


def layer_masks(masks, layer):
    return {method: values[layer] for method, values in masks.items()}


def prefix_layers(state, policy, masks):
    return {
        layer: CompactPrefix.build(key, policy.page, (policy.method,), layer_masks(masks, layer))
        for layer, key in enumerate(state.cache.key_cache)
    }


def candidate(policy, ranked, history, q, key, value, scale, gold):
    allocation = plan(ranked, history, summary_bytes=SUMMARY_BYTES.get(policy.method, 0),
                      budget_percent=policy.budget, v_reserve=policy.reserve, max_pages=policy.pages)
    result = screen(allocation, q, key, value, scale, gold_spans=gold)
    return allocation, result


class SparseAttention(Attention):
    def __init__(self, policy, prefixes, masks, gold):
        super().__init__(DENSE, None, gold)
        self.sparse_policy = policy
        self.prefixes = prefixes
        self.masks = masks

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        policy = self.sparse_policy
        groups, tokens, dim = key.shape[1:]
        q = query.reshape(groups, -1, dim).float()
        scale = kwargs.get("scaling") or dim ** -.5
        layer = module.layer_idx
        if layer not in self.online:
            self.online[layer] = CompactOnline(self.prefixes[layer])
        record = self.online[layer].append(key[0, :, -1], tokens)[policy.method]
        history = page_history(tokens, policy.page)
        channels = self.masks.get(policy.method)
        score = weights(q, record, policy.method, scale, channels=None if channels is None else channels[layer])
        allocation, out = candidate(policy, ranked_pages(score, history), history, q, key, value, scale,
                                    self.gold_spans)
        heads = q.shape[1]
        self._add("decode_layer_calls", 1)
        self._add("kv_group_calls", groups)
        for name, field in out.ledger.items():
            self._add("read" if name == "total" else name, field)
        self._add("dense_comparator_bytes", groups * allocation.dense_bytes)
        self._add("over_budget_group_steps", out.ledger["over_bytes"] > 0)
        self._add("fallback_group_steps", out.fallback)
        self._add("k_pages", allocation.count)
        self._add("prefix_pages", out.prefix)
        self._add("qk_mac", out.ledger["k_records"] * heads * dim)
        self._add("pv_mac", out.final_mask.sum() * dim)
        retained_dimensions = dim if channels is None else channels[layer].shape[-1]
        self._add("scorer_terms", groups * heads * history.pages * retained_dimensions *
                  (1 if policy.method == "q4" else 2))
        self._add("summary_append_bytes", groups * SUMMARY_BYTES[policy.method])
        self._add("summary_prefill_bytes", groups * ((tokens - 1) // policy.page) *
                  SUMMARY_BYTES[policy.method] if self.step == 0 else 0)
        self._add("evidence_page_total", groups * out.evidence["gold_pages"])
        self._add("evidence_token_total", groups * out.evidence["gold_tokens"])
        for name, field in out.evidence.items():
            if isinstance(field, torch.Tensor):
                self._add(f"evidence_{name}", field)
        self._add("worst_loss_read_k_sum", out.loss_read_k.amax(1))
        self._add("worst_loss_prefix_sum", out.loss_prefix.amax(1))
        return self.output(query, out.output)


class FitAttention(Attention):
    def __init__(self, calibration):
        super().__init__(DENSE, None, [])
        self.calibration = calibration

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step in (0, 7, 15):
            g, tokens, dim = key.shape[1:]
            q = query.reshape(g, -1, dim).float()
            prefix = CompactPrefix.build(key[:, :, :-1], 32, ("mr25",))
            record = CompactOnline(prefix).append(key[0, :, -1], tokens)["mr25"]
            self.calibration.update(module.layer_idx, q, key, record)
        return super().attention(module, query, key, value, attention_mask, **kwargs)


class DiagnosticAttention(Attention):
    def __init__(self, masks, spans, starts):
        super().__init__(DENSE, None, spans)
        structures = [Q4, MR, *STRUCTURES]
        self.fixed = [replace(p, name=p.method if p.page == 32 else "mr25_p64",
                              budget=None, pages=768 // p.page) for p in structures]
        self.fixed += [Policy(f"oracle_{n}", "oracle", budget=None, pages=n) for n in (24, 44)]
        self.capped = structures + [replace(Q4, name="quest4_5_r80", reserve=.8),
                                    replace(MR, name="mr25_5_r65", reserve=.65)]
        self.masks = masks
        self.starts = starts
        self.rows = []

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        layer = module.layer_idx
        if layer not in LAYERS or (self.step not in STEPS and self.step not in self.starts):
            return super().attention(module, query, key, value, attention_mask, **kwargs)
        groups, tokens, dim = key.shape[1:]
        q = query.reshape(groups, -1, dim).float()
        scale = kwargs.get("scaling") or dim ** -.5
        dense_logits = grouped_logits(query, key, scale)
        dense_output = grouped_output(dense_logits, value)
        dense_weights = torch.softmax(dense_logits, dim=-1)
        gold = self.starts.get(self.step, self.gold_spans)
        used_masks = layer_masks(self.masks, layer)
        ranks = {}
        histories = {size: page_history(tokens, size) for size in (32, 64)}
        for size in histories:
            methods = tuple(sorted({p.method for p in self.capped if p.page == size}))
            prefix = CompactPrefix.build(key[:, :, :-1], size, methods, used_masks)
            records = CompactOnline(prefix).append(key[0, :, -1], tokens)
            for method in methods:
                score = weights(q, records[method], method, scale, channels=used_masks.get(method))
                ranks[size, method] = ranked_pages(score, histories[size])
        exact = torch_page_mass(dense_logits, 32).sum(-1)
        ranks[32, "oracle"] = ranked_pages(exact, histories[32])
        for phase, policies in (("fixed_k", self.fixed), ("cap5", self.capped)):
            for p in policies:
                allocation, out = candidate(p, ranks[p.page, p.method], histories[p.page],
                                            q, key, value, scale, gold)
                abs_err = (out.output - dense_output).norm(dim=-1)
                rel_err = abs_err / (dense_output.norm(dim=-1) + 1e-12)
                positions = out.positions.clamp(max=tokens - 1).flatten(1)
                gathered = dense_weights.gather(-1, positions[:, None].expand(-1, q.shape[1], -1))
                covered = (gathered * out.valid.flatten(1)[:, None]).sum(-1)
                retained = (gathered * out.final_mask.flatten(2)).sum(-1)
                self.rows.append(dict(phase=phase, policy=p.name, page=p.page, step=self.step, layer=layer,
                                      answer_start=self.step in self.starts, tokens=tokens,
                                      pages=allocation.count, prefix_pages=out.prefix, cap=allocation.cap,
                                      fallback=out.fallback, loss_read_k=out.loss_read_k,
                                      loss_prefix=out.loss_prefix, abs_err=abs_err, rel_err=rel_err,
                                      head_coverage=covered, retained_head_coverage=retained,
                                      pv_tokens=out.final_mask.sum((1, 2, 3)), **out.ledger, **out.evidence))
        return super().attention(module, query, key, value, attention_mask, **kwargs)


def read_masks(root, device):
    receipt = json.loads((root / "channels.json").read_text())
    return {f"mr96_{rule}": {int(layer): torch.tensor(ids, device=device) for layer, ids in layers.items()}
            for rule, layers in receipt["channels"].items()}


def digest(path):
    return _sha(path / "manifest.json")


def dataset(root, cohort, split):
    data = root / "data" / cohort
    manifest, context, rows = load_rows(data, split)
    assert context == int(cohort.removesuffix("k")) * 1024
    return manifest, rows


def init_model(manifest):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, model_sha = _verified_model(MODEL, manifest)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        low_cpu_mem_usage=True, local_files_only=True).eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eos = model.generation_config.eos_token_id
    return model, tok, set(eos if isinstance(eos, list) else [eos]), model_sha


def prefill(model, row):
    return prefill_chunked(model, row["input_ids"], max_new_tokens=row["max_new_tokens"])


def calibrate_channels(root, rows, model, eos):
    receipt = root / "channels.json"
    if receipt.exists():
        return read_masks(root, "cuda")
    cal = ChannelCalibration()
    for row in rows[:FIT]:
        state = prefill(model, row)
        generate_from_prefill(model, state, min(16, row["max_new_tokens"]), eos, FitAttention(cal))
        del state
    channels = cal.freeze(96)
    stats = {}
    for layer, values in cal.statistics.items():
        stats[layer] = {}
        for metric in ("key_abs", "score_abs"):
            x = getattr(values, metric)
            rule = "k" if metric == "key_abs" else "score"
            ordered = channels[rule][layer].to(x.device)
            stats[layer][metric] = (x.gather(-1, ordered).sum(-1) / x.sum(-1).clamp_min(1e-12)).tolist()
        stats[layer]["observations"] = values.observations
    if len(stats) != model.config.num_hidden_layers:
        raise ValueError("channel fitting omitted a layer")
    seal(receipt, dict(ids=[r["sample_id"] for r in rows[:FIT]], channels={
        rule: {layer: dims.tolist() for layer, dims in layers.items()} for rule, layers in channels.items()},
        top96_share=stats))
    return read_masks(root, "cuda")


def prompt_bundle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt", compresslevel=3) as stream:
        json.dump(obj, stream, allow_nan=False, separators=(",", ":"))
    temp.replace(path)


def read_bundle(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def prompt_diag(rows):
    output = []
    for phase, policy in sorted({(r["phase"], r["policy"]) for r in rows}):
        all_rows = [r for r in rows if (r["phase"], r["policy"]) == (phase, policy)]
        for part in ("all", "value_start"):
            subset = [r for r in all_rows if part == "all" or r["answer_start"]]
            if not subset:
                continue
            obs = sum(len(r["total"]) for r in subset)
            def rate(metric, denominator):
                return sum(sum(r[metric]) for r in subset) / sum(
                    len(r["total"]) * r[denominator] for r in subset)
            def avg(metric):
                return sum(sum(r[metric]) for r in subset) / obs
            def quantile(metric, p):
                return float(np.quantile(np.asarray([r[metric] for r in subset]), p))
            output.append(dict(phase=phase, policy=policy, scope=part, groups=obs,
                               k_page_recall=rate("k_page_hits", "gold_pages"),
                               prefix_page_recall=rate("final_k_page_hits", "gold_pages"),
                               v_page_recall=rate("v_page_hits", "gold_pages"),
                               v_evidence_page_recall=rate("v_evidence_page_hits", "gold_pages"),
                               k_evidence_token_recall=rate("k_gold_token_hits", "gold_tokens"),
                               v_evidence_token_recall=rate("v_gold_token_hits", "gold_tokens"),
                               initial_v_evidence_token_recall=rate("initial_v_gold_token_hits", "gold_tokens"),
                               mean_k_pages=avg("pages"), mean_prefix=avg("prefix_pages"),
                               mean_k_tokens=avg("k_records"), mean_v_tokens=avg("v_records"),
                               mean_head_pv_tokens=avg("pv_tokens"), mean_read_bytes=avg("total"),
                               mean_summary_bytes=avg("summary_bytes"), mean_k_bytes=avg("k_bytes"),
                               mean_v_bytes=avg("v_bytes"), fallback_rate=avg("fallback"),
                               overrun_groups=sum(sum(x > 0 for x in r["over_bytes"]) for r in subset),
                               abs_p95=quantile("abs_err", .95), abs_p99=quantile("abs_err", .99),
                               relative_p95=quantile("rel_err", .95), relative_p99=quantile("rel_err", .99),
                               k_head_mass_p05=quantile("head_coverage", .05),
                               retained_head_mass_p05=quantile("retained_head_coverage", .05),
                               lost_mass_read_k_p99=quantile("loss_read_k", .99),
                               lost_mass_prefix_p99=quantile("loss_prefix", .99)))
    return output


def mean_records(records, keys):
    groups = {}
    for row in records:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [{**dict(zip(keys, k)), "prompts": len(rows),
             **{m: float(np.mean([r[m] for r in rows])) for m in rows[0] if m not in (*keys, "sample_id")}}
            for k, rows in sorted(groups.items())]


def choose_structure(report):
    by = {(r["scope"], r["policy"]): r for r in report if r["phase"] == "cap5"}
    base = by["value_start", MR.name]
    absolute_base = by["all", MR.name]
    deltas = []
    for p in STRUCTURES:
        row = by["value_start", p.name]
        full = by["all", p.name]
        retrieval = row["v_evidence_token_recall"] - base["v_evidence_token_recall"]
        byte_ratio = full["mean_read_bytes"] / absolute_base["mean_read_bytes"] - 1
        tail_ratio = row["abs_p99"] / base["abs_p99"] - 1
        qualifies = (retrieval >= -.01 and tail_ratio <= .05 and full["overrun_groups"] == 0 and
                     ((retrieval >= .01 and byte_ratio <= TOL) or
                      (byte_ratio <= -.05 and full["mean_k_tokens"] <= absolute_base["mean_k_tokens"] and
                       full["mean_head_pv_tokens"] <= absolute_base["mean_head_pv_tokens"])))
        deltas.append(dict(policy=p.name, retrieval_delta=retrieval, read_delta=byte_ratio,
                           value_start_p99_delta=tail_ratio, qualifies=qualifies))
    ranked = sorted((r for r in deltas if r["qualifies"]), key=lambda x: (-x["retrieval_delta"], x["read_delta"],
                                                                         x["value_start_p99_delta"], x["policy"]))
    return dict(candidates=deltas, selected=ranked[0]["policy"] if ranked else None)


def diagnostic(root):
    manifest, rows = dataset(root, "32k", "calibration")
    model, tok, eos, model_sha = init_model(manifest)
    output = root / "diagnostic32"
    output.mkdir(exist_ok=True)
    seal(output / "config.json", dict(code=sources(), model=model_sha, base=BASE, fit=FIT,
                                     dataset_sha256=digest(root / "data" / "32k"), layers=list(LAYERS), steps=list(STEPS),
                                     projection_fit_steps=[0, 7, 15], policies=list(map(asdict, STRUCTURES))))
    masks = calibrate_channels(root, rows, model, eos)
    aggregate = []
    for index, row in enumerate(rows[FIT:], FIT):
        target = output / "bundles" / f"{index:04d}.json.gz"
        if not target.exists():
            state = prefill(model, row)
            run = generate_from_prefill(model, state, row["max_new_tokens"], eos, Attention(DENSE, None, row["gold_spans"]))
            starts = answer_spans(tok, run.token_ids, row)
            probe = DiagnosticAttention(masks, original_spans(row), starts)
            replay = generate_from_prefill(model, state, row["max_new_tokens"], eos, probe)
            if replay.token_ids != run.token_ids:
                raise ValueError("common-state replay changed the Dense continuation")
            packed = [{k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in item.items()}
                      for item in probe.rows]
            prompt_bundle(target, dict(sample_id=row["sample_id"], score=make_answer(DENSE, run, tok, row, eos),
                                       first_value_decode_steps=sorted(starts), rows=packed))
            del state, probe
        bundle = read_bundle(target)
        aggregate.extend(dict(sample_id=bundle["sample_id"], **r) for r in prompt_diag(bundle["rows"]))
        print(f"32K diagnosis {index - FIT + 1}/{len(rows) - FIT}", flush=True)
    report = mean_records(aggregate, ("phase", "policy", "scope"))
    write_csv(output / "per_prompt.csv", aggregate)
    write_csv(output / "summary.csv", report)
    decision = choose_structure(report)
    _write_json(output / "decision.json", decision)
    return decision


def summarize_predictions(predictions):
    groups = {}
    for row in predictions:
        groups.setdefault(row["policy"], []).append(row)
    output = []
    for name, rows in groups.items():
        def total(metric):
            return sum(r.get(metric, 0) for r in rows)
        def ratio(numerator, denominator):
            d = total(denominator)
            return total(numerator) / d if d else 0
        output.append(dict(policy=name, prompts=len(rows), official_score=total("official_score") / len(rows),
                           complete=total("complete_correct"), truncated=total("truncated"),
                           read_per_decode=ratio("read", "decode_tokens"), read_per_answer=total("read") / len(rows),
                           k_pages_per_group=ratio("k_pages", "kv_group_calls"),
                           k_tokens_per_group=ratio("k_records", "kv_group_calls"),
                           v_tokens_per_group=ratio("v_records", "kv_group_calls"),
                           summary_per_decode=ratio("summary_bytes", "decode_tokens"),
                           k_bytes_per_decode=ratio("k_bytes", "decode_tokens"),
                           v_bytes_per_decode=ratio("v_bytes", "decode_tokens"),
                           qk_per_decode=ratio("qk_mac", "decode_tokens"), pv_per_decode=ratio("pv_mac", "decode_tokens"),
                           scorer_terms_per_decode=ratio("scorer_terms", "decode_tokens"),
                           fallback_rate=ratio("fallback_group_steps", "kv_group_calls"),
                           group_steps=total("kv_group_calls"), overrun_groups=total("over_budget_group_steps"),
                           k_evidence=ratio("evidence_k_gold_token_hits", "evidence_token_total"),
                           v_evidence=ratio("evidence_v_gold_token_hits", "evidence_token_total"),
                           k_page_recall=ratio("evidence_k_page_hits", "evidence_page_total"),
                           v_page_recall=ratio("evidence_v_page_hits", "evidence_page_total"),
                           pruned_evidence=ratio("evidence_initial_v_gold_token_hits", "evidence_token_total") -
                                           ratio("evidence_v_gold_token_hits", "evidence_token_total")))
    return output


def run_phase(root, cohort, tag, policies, model_info, *, dense=False, count=None, freeze=None):
    split = "test" if freeze is not None else "calibration"
    _, rows = dataset(root, cohort, split)
    rows = rows[:count]
    model, tok, eos, model_sha = model_info
    output = root / cohort / tag
    mask_needed = any(p.method.startswith("mr96") for p in policies)
    masks = read_masks(root, "cuda") if mask_needed else {}
    config = dict(cohort=cohort, phase=tag, split=split, base=BASE, sources=sources(), model=model_sha,
                  manifest=digest(root / "data" / cohort), ids=[r["sample_id"] for r in rows],
                  seed_partition="new official RULER Multi-Value", dense=dense, prefill_chunk=2048,
                  policies=list(map(asdict, policies)), torch=torch.__version__, transformers=transformers.__version__,
                  device=torch.cuda.get_device_name(), channels_sha256=_sha(root / "channels.json") if masks else None)
    if freeze is not None and freeze["test"] != config:
        raise ValueError("held-out inputs differ from the frozen plan")
    output.mkdir(parents=True, exist_ok=True)
    seal(output / "config.json", config)
    predictions = []
    for index, row in enumerate(rows):
        target = output / "bundles" / f"{index:04d}.json.gz"
        if not target.exists():
            state = prefill(model, row)
            results = []
            for p in ([DENSE] if dense else []) + list(policies):
                if p == DENSE:
                    attention = Attention(DENSE, None, row["gold_spans"])
                else:
                    attention = SparseAttention(p, prefix_layers(state, p, masks), masks, original_spans(row))
                run = generate_from_prefill(model, state, row["max_new_tokens"], eos, attention)
                answer = make_answer(p, run, tok, row, eos)
                if answer["decode_layer_calls"] != answer["decode_tokens"] * model.config.num_hidden_layers:
                    raise ValueError("decode accounting omitted expected groups")
                if answer.get("over_budget_group_steps"):
                    raise ValueError(f"physical budget overrun in {p.name}")
                if p == DENSE:
                    full_tokens = answer["qk_mac"] // (model.config.head_dim *
                                      (model.config.num_attention_heads // model.config.num_key_value_heads))
                    answer["k_records"] = answer["v_records"] = full_tokens
                answer["sample_id"] = row["sample_id"]
                answer["task"] = row["task"]
                results.append(answer)
                del attention
            prompt_bundle(target, results)
            del state
        predictions.extend(read_bundle(target))
        print(f"{cohort} {tag} {index + 1}/{len(rows)}", flush=True)
    result = summarize_predictions(predictions)
    write_csv(output / "quality_cost.csv", result)
    _write_json(output / "predictions.json", predictions)
    _write_json(output / "complete.json", dict(config=_sha(output / "config.json"), predictions=len(predictions),
                                               bundles={p.name: _sha(p) for p in sorted((output / "bundles").glob("*.json.gz"))}))
    return result, predictions, config


def pair_result(path, rows, predictions, ref, cand):
    by = {row["policy"]: row for row in rows}
    a, b = by[ref.name], by[cand.name]
    pair = pairing(path.with_name(path.stem + "_prompts.json"), predictions, ref.name, cand.name)
    score, read = pair["official_score"], pair["read"]
    gates = independent_decision(
        score_delta=score["point_delta"], score_ci_low=score["ci_low"], reference_bytes=a["read_per_decode"],
        candidate_bytes=b["read_per_decode"], answer_read_relative_delta=read["relative_delta"],
        answer_read_relative_hi=read["relative_ci_high"], reference_over=a["overrun_groups"],
        candidate_over=b["overrun_groups"], read_tolerance=TOL)
    both_compute = (b["qk_per_decode"] < a["qk_per_decode"] and b["pv_per_decode"] < a["pv_per_decode"])
    answer = dict(reference=a, candidate=b, score_delta=score["point_delta"],
                  score_ci=[score["ci_low"], score["ci_high"]], answer_read_delta=read["relative_delta"],
                  answer_read_ci=[read["relative_ci_low"], read["relative_ci_high"]],
                  qk_mac_delta=b["qk_per_decode"] / a["qk_per_decode"] - 1,
                  pv_mac_delta=b["pv_per_decode"] / a["pv_per_decode"] - 1,
                  both_compute_reduced=both_compute, **gates)
    answer["passes"] = gates["zero_overruns"] and (gates["quality_route"] or (gates["saving_route"] and both_compute))
    _write_json(path, answer)
    return answer


def freeze_test(root, cohort, policies, model_info, *, count, calibration, dense=False):
    _, rows = dataset(root, cohort, "test")
    model, _, _, model_sha = model_info
    needs = any(p.method.startswith("mr96") for p in policies)
    test = dict(cohort=cohort, phase="independent", split="test", base=BASE, sources=sources(), model=model_sha,
                manifest=digest(root / "data" / cohort), ids=[r["sample_id"] for r in rows[:count]],
                seed_partition="new official RULER Multi-Value", dense=dense, prefill_chunk=2048,
                policies=list(map(asdict, policies)), torch=torch.__version__, transformers=transformers.__version__,
                device=torch.cuda.get_device_name(), channels_sha256=_sha(root / "channels.json") if needs else None)
    freeze = dict(test=test, calibration=calibration, primary=[p.name for p in policies],
                  gates=dict(max_read_delta_for_quality=TOL, min_score_gain=.01, lower_score_bound=0,
                             noninferiority=-.01, min_answer_saving=.10, both_qk_and_pv_reduction_for_saving=True,
                             no_physical_overruns=True))
    seal(root / cohort / "frozen_test.json", freeze)


def execute_test(root, cohort):
    output = root / cohort
    path = output / "frozen_test.json"
    if not path.exists():
        return
    freeze = json.loads(path.read_text())
    config = freeze["test"]
    policies = [Policy(**p) for p in config["policies"]]
    ref, cand = policies
    manifest, _ = dataset(root, cohort, "test")
    model_info = init_model(manifest)
    count, dense = len(config["ids"]), config["dense"]
    rows, predictions, _ = run_phase(root, cohort, "independent", policies, model_info, dense=dense, count=count,
                                     freeze=freeze)
    outcome = pair_result(output / "independent" / "paired.json", rows, predictions, ref, cand)
    _write_json(output / "decision.json", dict(selected=cand.name, opened_test=True, results=outcome))


def independence(root):
    history = Path("results/round6_subpage_head/data/test")
    _, _, prior = load_rows(history, "test")
    sets = {"round9_used_32k": {r["source_metadata"]["instance_sha256"] for r in prior}}
    receipts = {}
    for cohort in ("16k", "32k", "64k"):
        for split in (("calibration",) if cohort == "16k" else ("calibration", "test")):
            manifest, rows = dataset(root, cohort, split)
            sha = {r["source_metadata"]["instance_sha256"] for r in rows}
            if len(sha) != len(rows) or any(sha & others for others in sets.values()):
                raise ValueError("data independence check failed")
            sets[f"{cohort}_{split}"] = sha
            receipts[f"{cohort}_{split}"] = dict(rows=len(rows), manifest_sha256=digest(root / "data" / cohort),
                                                 dataset_sha256=manifest["splits"][split]["sha256"])
    seal(root / "data_receipt.json", receipts)


def calibration_table(root, cohort, phase, policies, model_info, *, dense=False):
    rows, predictions, _ = run_phase(root, cohort, phase, policies, model_info, dense=dense)
    return {r["policy"]: r for r in rows}, predictions


def match_reference(root, cohort, ref, cand, observed, model_info, phase):
    """Use at most one extra reservation point, selected without test outcomes."""
    read_gap = observed[cand.name]["read_per_decode"] / observed[ref.name]["read_per_decode"] - 1
    if abs(read_gap) <= TOL:
        return ref, dict(read_delta=read_gap, reference_phase=phase)
    # Only the KV part changes as reservation adjusts. Summary traffic is fixed.
    baseline = observed[ref.name]
    target = observed[cand.name]["read_per_decode"]
    fixed = baseline["summary_per_decode"]
    factor = (baseline["read_per_decode"] - fixed) / max(1, target - fixed)
    reserve = min(1., max(.45, round((1 + ref.reserve) * factor - 1, 3)))
    if abs(reserve - ref.reserve) < .005:
        return ref, dict(read_delta=read_gap, reference_phase=phase)
    alt = replace(ref, name=f"{ref.name}_match{round(reserve * 1000):03d}", reserve=reserve)
    extra, _ = calibration_table(root, cohort, "matching", [alt], model_info)
    gap = observed[cand.name]["read_per_decode"] / extra[alt.name]["read_per_decode"] - 1
    if abs(gap) < abs(read_gap):
        return alt, dict(read_delta=gap, reference_phase="matching", original_delta=read_gap)
    return ref, dict(read_delta=read_gap, reference_phase=phase, alternate_delta=gap)


def closed_loop(root, cohort):
    out = root / cohort
    out.mkdir(exist_ok=True)
    if cohort == "32k":
        decision = json.loads((root / "diagnostic32" / "decision.json").read_text())
        if decision["selected"] is None:
            _write_json(out / "decision.json", dict(selected=None, opened_test=False,
                        reason="No structure qualified on common-state evidence, tail error and physical cost."))
            return
        ref = MR
        cand = next(p for p in STRUCTURES if p.name == decision["selected"])
        count, dense_test = 128, False
    else:
        ref, cand = Q4, MR
        count, dense_test = 64, True

    manifest, _ = dataset(root, cohort, "calibration")
    model_info = init_model(manifest)
    observed, predictions = calibration_table(root, cohort, "cap5", [ref, cand], model_info, dense=True)
    initial = pair_result(out / "cap5" / "paired.json", list(observed.values()), predictions, ref, cand)
    calibration = dict(initial_cap5=initial, selected_phase="cap5", widened=False)
    if cohort == "64k":
        dense_score = observed[DENSE.name]["official_score"]
        sparse_scores = [observed[name]["official_score"] for name in (ref.name, cand.name)]
        if max(sparse_scores) < .90 or dense_score - max(sparse_scores) > .08:
            ref7 = replace(ref, name="quest4_7_r65", budget=7)
            cand7 = replace(cand, name="mr25_7_r80", budget=7)
            cap7 = [ref7, cand7]
            path = root / "diagnostic32" / "decision.json"
            if path.exists():
                selection = json.loads(path.read_text())["selected"]
                cap7.extend(replace(p, name=p.name + "_cap7", budget=7) for p in STRUCTURES if p.name == selection)
            observed7, pred7 = calibration_table(root, cohort, "cap7", cap7, model_info)
            improved = pair_result(out / "cap7" / "paired.json", list(observed7.values()), pred7, ref7, cand7)
            calibration["cap7"] = improved
            if observed7[cand7.name]["official_score"] > observed[cand.name]["official_score"]:
                ref, cand, observed = ref7, cand7, observed7
                calibration.update(selected_phase="cap7", widened=True)
    if cohort == "32k":
        permits_comparison = (initial["score_delta"] >= .01 or
                              (initial["score_delta"] >= -.01 and initial["read_delta"] <= -.05 and
                               initial["both_compute_reduced"]))
        if not permits_comparison:
            _write_json(out / "decision.json", dict(selected=cand.name, opened_test=False, calibration=calibration,
                        reason="The screened structure did not retain its improvement in closed-loop calibration."))
            return
    saving_only = cohort == "32k" and initial["score_delta"] < .01
    if not saving_only:
        ref, match = match_reference(root, cohort, ref, cand, observed, model_info, calibration["selected_phase"])
        calibration["matching"] = match
    freeze_test(root, cohort, [ref, cand], model_info, count=count, calibration=calibration, dense=dense_test)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("data", "diagnostic", "closed32", "closed64", "test32", "test64"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    with torch.inference_mode():
        if args.phase == "data":
            independence(root)
        elif args.phase == "diagnostic":
            diagnostic(root)
        elif args.phase.startswith("test"):
            execute_test(root, args.phase[-2:] + "k")
        else:
            closed_loop(root, args.phase[-2:] + "k")


if __name__ == "__main__":
    main()
