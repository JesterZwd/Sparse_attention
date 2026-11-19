"""Grouped FP32 decode attention and device-resident Oracle page selection.

The full-history QK, and the masked full-width AV used here, are quality-reference
work. This is not a sparse-kernel or speed implementation. No expanded GQA KV
history, full logits, or full V is copied to CPU. SDPA owns dense prefill.

Shadow callback schema (one JSON-serializable dict per spec/KV group):
``spec, algorithm, budget_fraction, target, delta, max_swaps, layer, step,
kv_group, valid_tokens, budget_pages, available_pages, pages, head_coverage,
head_output_absolute_errors, head_output_relative_errors,
original_target_deficit, gate_hit, swaps, candidate_pairs,
executed_candidate_pairs, gold_span_recall, gold_token_recall,
gold_page_recall, all_gold_spans_retained, all_gold_pages_retained, work``.
Errors are per-head L2 errors relative to the common FP32 dense output, before
model-dtype casting; relative errors use ||dense|| + 1e-12. Optional spec flag
``audit_float64=True`` also emits errors against a float64 QK-logit/AV audit
(softmax from the same FP32 raw logits, not a new QK).

``stats()`` materializes actual-path aggregate tensors at end of generation.
``shadow_work`` and ``shadow_audit_seconds`` are separate audit accounting;
shadow timing synchronizes only at sampled audit boundaries and includes callback
I/O. Generation timing with shadows must subtract that audit time and remains an
instrumented reference, not a systems-speed measurement.
"""

import math
from numbers import Integral
import time

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .selection import protected_pages
from .torch_selection import select_torch


PAGE_SIZE = 32


def grouped_logits(query, key, scaling):
    """[1,G*H,1,D] Q × shared [1,G,T,D] K -> [G,H,T] FP32."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("grouped decode supports batch=1 Q/K tensors only")
    if query.shape[-2] != 1 or query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
        raise ValueError("incompatible single-token grouped query/key shapes")
    groups, tokens, dimension = key.shape[1:]
    q = query.reshape(groups, query.shape[1] // groups, dimension).float()
    return torch.bmm(q, key[0].float().transpose(1, 2)) * scaling


def torch_page_mass(logits, page_size=PAGE_SIZE):
    """Float64 full-history softmax mass [G,J,H], including partial last pages."""
    if logits.ndim != 3 or logits.shape[-1] == 0 or page_size < 1:
        raise ValueError("expected nonempty [groups,heads,tokens] logits")
    groups, heads, tokens = logits.shape
    weights = torch.softmax(logits.double(), dim=-1)
    weights = F.pad(weights, (0, (-tokens) % page_size))
    return weights.reshape(groups, heads, -1, page_size).sum(-1).transpose(1, 2).contiguous()


def grouped_output(logits, value, selected_pages=None, page_size=PAGE_SIZE):
    """Stable selected-raw-logit softmax × shared V, returning [G,H,D].

    Selection masks logits to -inf. AV still executes a full-width batched GEMM;
    counters explicitly distinguish that executed work from selected-token work.
    Float64 logits are accepted for an optional audit with float64 softmax/AV.
    """
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != logits.shape[0] or value.shape[-2] != logits.shape[-1]:
        raise ValueError("shared values must be [1,groups,tokens,dimension]")
    scores = logits
    if selected_pages is not None:
        tokens = logits.shape[-1]
        expected_pages = (tokens + page_size - 1) // page_size
        if selected_pages.shape != (logits.shape[0], expected_pages):
            raise ValueError("selection must be a [groups,pages] boolean mask")
        torch._assert_async(selected_pages.any(-1).all(), "attention selection must be nonempty")
        token_mask = selected_pages.repeat_interleave(page_size, dim=-1)[:, :tokens]
        scores = scores.masked_fill(~token_mask[:, None], -torch.inf)
    return torch.bmm(torch.softmax(scores, dim=-1), value[0].to(logits.dtype))


def _validate_spec(spec):
    result = dict(spec)
    if result.get("algorithm") not in {"dense", "A1", "SWAP"}:
        raise ValueError("algorithm must be dense, A1 or SWAP")
    if not math.isfinite(result["budget_fraction"]) or not 0 < result["budget_fraction"] <= 1:
        raise ValueError("budget_fraction must be in (0,1]")
    if not math.isfinite(result["target"]) or not 0 <= result["target"] <= 1:
        raise ValueError("target must be in [0,1]")
    if not math.isfinite(result["delta"]) or result["delta"] < 0:
        raise ValueError("delta must be nonnegative")
    if not isinstance(result["max_swaps"], Integral) or result["max_swaps"] < 0:
        raise ValueError("max_swaps must be nonnegative integer")
    return result


class OracleAttention:
    """Patch every SDPA decode layer; runner sets ``step=-1`` for prefill.

    Multi-token calls and all prefill calls delegate the original SDPA. Decode
    supports a single unpadded, fully visible sequence and inference dropout=0.
    Budget is exactly floor(J*fraction), never silently raised for protection;
    small histories whose protection exceeds B must use a larger budget.
    """

    def __init__(self, algorithm="dense", budget_fraction=.2, target=.8, delta=.01,
                 max_swaps=1, shadow_specs=None, sample_layers=None,
                 sample_steps=None, callback=None, gold_spans=None):
        self.spec = _validate_spec(dict(algorithm=algorithm, budget_fraction=budget_fraction,
                                       target=target, delta=delta, max_swaps=max_swaps))
        self.algorithm = algorithm
        self.shadow_specs = []
        for index, spec in enumerate(shadow_specs or []):
            merged = {**self.spec, **spec}
            merged.setdefault("name", f"shadow_{index}")
            self.shadow_specs.append(_validate_spec(merged))
        if self.shadow_specs and algorithm != "dense":
            raise ValueError("shadow audits require actual dense attention")
        names = [spec["name"] for spec in self.shadow_specs]
        if len(set(names)) != len(names):
            raise ValueError("shadow spec names must be unique")
        self.sample_layers = None if sample_layers is None else set(sample_layers)
        self.sample_steps = None if sample_steps is None else set(sample_steps)
        self.callback = callback
        self.gold_spans = []
        for span in gold_spans or []:
            if len(span) != 2 or any(not isinstance(i, Integral) for i in span) or not 0 <= span[0] < span[1]:
                raise ValueError("gold spans must be nonempty token [start,end) intervals")
            self.gold_spans.append((int(span[0]), int(span[1])))
        gold_tokens = {token for start, end in self.gold_spans for token in range(start, end)}
        self._gold_page_tokens = {}
        for token in gold_tokens:
            page = token // PAGE_SIZE
            self._gold_page_tokens[page] = self._gold_page_tokens.get(page, 0) + 1
        self._gold_device_indices = {}
        self.step = -1
        self.original = None
        self._totals = {}
        self._minimum_coverage = None
        self.shadow_work = {"group_spec_evaluations": 0, "page_mass_evaluations": 0,
                            "executed_candidate_pairs": 0, "logical_candidate_pairs": 0,
                            "executed_av_multiply_adds": 0, "float64_audit_av_multiply_adds": 0}
        self.shadow_audit_seconds = 0.

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("attention context already entered")
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.attention
        return self

    def __exit__(self, *_):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.original
        self.original = None

    def _add(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().sum()
        self._totals[name] = self._totals.get(name, 0) + value

    def stats(self):
        """JSON-safe actual-path totals; host synchronization occurs here only."""
        names = ["decode_layer_calls", "kv_group_calls", "selected_pages", "gate_hits",
                 "swaps", "candidate_pairs", "executed_candidate_pairs",
                 "coverage_head_count", "coverage_sum", "original_target_deficit",
                 "executed_qk_multiply_adds", "executed_av_multiply_adds",
                 "logical_selected_av_multiply_adds", "executed_softmax_elements",
                 "logical_selected_softmax_elements", "page_mass_softmax_elements",
                 "page_mass_evaluations", "gold_token_hits", "gold_token_total",
                 "gold_page_hits", "gold_page_total", "gold_span_hits", "gold_span_total",
                 "all_gold_spans_group_hits", "gold_group_count"]
        result = {name: self._totals.get(name, 0) for name in names}
        for name, value in self._totals.items():
            result[name] = value
        result = {key: value.item() if isinstance(value, torch.Tensor) else value
                  for key, value in result.items()}
        result["minimum_head_coverage"] = (None if self._minimum_coverage is None
                                           else self._minimum_coverage.item())
        result["mean_head_coverage"] = (result["coverage_sum"] / result["coverage_head_count"]
                                        if result["coverage_head_count"] else None)
        result["shadow_audit_seconds"] = self.shadow_audit_seconds
        for kind in ("span", "token", "page"):
            total = result[f"gold_{kind}_total"]
            result[f"gold_{kind}_recall"] = result[f"gold_{kind}_hits"] / total if total else None
        return result

    def _selection(self, mass, spec, tokens):
        pages = mass.shape[1]
        budget = math.floor(pages * spec["budget_fraction"])
        protection = protected_pages(tokens)
        if budget < len(protection) or budget == 0:
            raise ValueError(f"physical page budget {budget} cannot cover {len(protection)} protected pages")
        return select_torch(mass, protection, budget, spec["target"], spec["algorithm"],
                            delta=spec["delta"], max_swaps=spec["max_swaps"])

    def _record_actual(self, logits, value, selection=None):
        groups, heads, tokens = logits.shape
        dimension = value.shape[-1]
        pages = (tokens + PAGE_SIZE - 1) // PAGE_SIZE
        self._add("decode_layer_calls", 1)
        self._add("kv_group_calls", groups)
        self._add("executed_qk_multiply_adds", groups * heads * tokens * dimension)
        self._add("executed_av_multiply_adds", groups * heads * tokens * dimension)
        self._add("executed_softmax_elements", groups * heads * tokens)
        self._record_gold_aggregate(None if selection is None else selection.mask,
                                    groups, tokens, logits.device)
        if selection is None:
            self._add("selected_pages", groups * pages)
            self._add("logical_selected_av_multiply_adds", groups * heads * tokens * dimension)
            self._add("logical_selected_softmax_elements", groups * heads * tokens)
            return
        self._add("selected_pages", selection.mask.sum())
        sizes = torch.full((pages,), PAGE_SIZE, device=logits.device, dtype=torch.int64)
        sizes[-1] = tokens - PAGE_SIZE * (pages - 1)
        selected_tokens = (selection.mask * sizes).sum()
        self._add("logical_selected_av_multiply_adds", selected_tokens * heads * dimension)
        self._add("logical_selected_softmax_elements", selected_tokens * heads)
        self._add("gate_hits", selection.gate_hit.sum())
        self._add("swaps", selection.swaps.sum())
        self._add("candidate_pairs", selection.candidate_pairs.sum())
        self._add("executed_candidate_pairs", selection.metrics["executed_candidate_pairs"].sum())
        self._add("coverage_sum", selection.coverage.sum())
        self._add("coverage_head_count", groups * heads)
        self._add("original_target_deficit", (self.spec["target"] - selection.coverage).clamp_min(0).sum())
        minimum = selection.coverage.min().detach()
        self._minimum_coverage = minimum if self._minimum_coverage is None else torch.minimum(self._minimum_coverage, minimum)
        for name, val in selection.metrics.items():
            if name not in {"initial_coverage", "executed_candidate_pairs", "logical_candidate_pairs"}:
                self._add("selector_" + name, val)
        self._add("page_mass_evaluations", groups)
        self._add("page_mass_softmax_elements", groups * heads * tokens)

    def _record_gold_aggregate(self, mask, groups, tokens, device):
        """Audit only: gold never participates in scores, candidates or decisions."""
        if not self.gold_spans:
            return
        if any(end > tokens for _, end in self.gold_spans):
            raise ValueError("gold span exceeds the final token history")
        page_total = len(self._gold_page_tokens)
        token_total = sum(self._gold_page_tokens.values())
        span_total = len(self.gold_spans)
        self._add("gold_group_count", groups)
        self._add("gold_page_total", groups * page_total)
        self._add("gold_token_total", groups * token_total)
        self._add("gold_span_total", groups * span_total)
        if mask is None:
            self._add("gold_page_hits", groups * page_total)
            self._add("gold_token_hits", groups * token_total)
            self._add("gold_span_hits", groups * span_total)
            self._add("all_gold_spans_group_hits", groups)
            return
        if device not in self._gold_device_indices:
            pages = sorted(self._gold_page_tokens)
            self._gold_device_indices[device] = (
                torch.tensor(pages, device=device, dtype=torch.int64),
                torch.tensor([self._gold_page_tokens[p] for p in pages], device=device, dtype=torch.int64))
        page_ids, token_counts = self._gold_device_indices[device]
        selected_gold = mask[:, page_ids]
        self._add("gold_page_hits", selected_gold.sum())
        self._add("gold_token_hits", (selected_gold * token_counts).sum())
        span_hits = torch.stack([mask[:, start // PAGE_SIZE:(end - 1) // PAGE_SIZE + 1].all(-1)
                                 for start, end in self.gold_spans], -1)
        self._add("gold_span_hits", span_hits.sum())
        self._add("all_gold_spans_group_hits", span_hits.all(-1).sum())

    def _gold_recall(self, chosen, tokens):
        if not self.gold_spans:
            return {"gold_span_recall": None, "gold_token_recall": None, "gold_page_recall": None,
                    "all_gold_spans_retained": None, "all_gold_pages_retained": None}
        if any(end > tokens for _, end in self.gold_spans):
            raise ValueError("gold span exceeds the final token history")
        gold_tokens = {token for start, end in self.gold_spans for token in range(start, end)}
        gold_pages = {token // PAGE_SIZE for token in gold_tokens}
        span_hits = [all(page in chosen for page in range(start // PAGE_SIZE, (end - 1) // PAGE_SIZE + 1))
                     for start, end in self.gold_spans]
        return dict(gold_span_recall=sum(span_hits) / len(span_hits),
                    gold_token_recall=sum(token // PAGE_SIZE in chosen for token in gold_tokens) / len(gold_tokens),
                    gold_page_recall=len(gold_pages & chosen) / len(gold_pages),
                    all_gold_spans_retained=all(span_hits),
                    all_gold_pages_retained=gold_pages <= chosen)

    def _audit(self, module, logits, value, dense):
        groups, heads, tokens = logits.shape
        dimension = value.shape[-1]
        mass = torch_page_mass(logits)
        self.shadow_work["page_mass_evaluations"] += groups
        dense64 = None
        for spec in self.shadow_specs:
            if spec["algorithm"] == "dense":
                raise ValueError("shadow spec must select A1 or SWAP")
            selected = self._selection(mass, spec, tokens)
            output = grouped_output(logits, value, selected.mask)
            absolute = torch.linalg.vector_norm(output - dense, dim=-1)
            relative = absolute / (torch.linalg.vector_norm(dense, dim=-1) + 1e-12)
            # Only compact page selections, per-head metrics and counters transfer.
            # Stack before transfer to avoid a separate synchronization per group.
            metrics = torch.stack([selected.swaps, selected.candidate_pairs,
                                   selected.metrics["executed_candidate_pairs"], selected.gate_hit.to(torch.int64)], -1).cpu().tolist()
            coverage = selected.coverage.cpu().tolist()
            errors = torch.stack([absolute, relative], -1).cpu().tolist()
            mask = selected.mask.cpu().tolist()
            audit_errors = None
            if spec.get("audit_float64", False):
                if dense64 is None:
                    dense64 = grouped_output(logits.double(), value)
                    self.shadow_work["float64_audit_av_multiply_adds"] += groups * heads * tokens * dimension
                out64 = grouped_output(logits.double(), value, selected.mask)
                absolute64 = torch.linalg.vector_norm(out64 - dense64, dim=-1)
                relative64 = absolute64 / (torch.linalg.vector_norm(dense64, dim=-1) + 1e-12)
                audit_errors = torch.stack([absolute64, relative64], -1).cpu().tolist()
                self.shadow_work["float64_audit_av_multiply_adds"] += groups * heads * tokens * dimension
            for group in range(groups):
                chosen = [page for page, present in enumerate(mask[group]) if present]
                n_selected_tokens = sum(min(PAGE_SIZE, tokens - page * PAGE_SIZE) for page in chosen)
                swaps, pairs, executed, gate = metrics[group]
                record = dict(spec=spec["name"], algorithm=spec["algorithm"],
                              budget_fraction=spec["budget_fraction"], target=spec["target"],
                              delta=spec["delta"], max_swaps=spec["max_swaps"],
                              layer=module.layer_idx, step=self.step, kv_group=group,
                              valid_tokens=tokens, budget_pages=len(chosen), available_pages=mass.shape[1],
                              pages=chosen, head_coverage=coverage[group],
                              head_output_absolute_errors=[pair[0] for pair in errors[group]],
                              head_output_relative_errors=[pair[1] for pair in errors[group]],
                              original_target_deficit=sum(max(spec["target"] - c, 0.) for c in coverage[group]),
                              gate_hit=bool(gate), swaps=swaps, candidate_pairs=pairs,
                              executed_candidate_pairs=executed,
                              **self._gold_recall(set(chosen), tokens),
                              work=dict(common_executed_qk_multiply_adds=heads * tokens * dimension,
                                        common_dense_reference_av_multiply_adds=heads * tokens * dimension,
                                        shared_page_mass_softmax_elements=heads * tokens,
                                        executed_av_multiply_adds=heads * tokens * dimension,
                                        logical_selected_av_multiply_adds=heads * n_selected_tokens * dimension,
                                        executed_softmax_elements=heads * tokens,
                                        logical_selected_softmax_elements=heads * n_selected_tokens,
                                        executed_candidate_head_evaluations=executed * heads,
                                        logical_candidate_head_evaluations=pairs * heads))
                if audit_errors is not None:
                    record["head_output_absolute_errors_float64_audit"] = [p[0] for p in audit_errors[group]]
                    record["head_output_relative_errors_float64_audit"] = [p[1] for p in audit_errors[group]]
                if self.callback is not None:
                    self.callback(record)
                self.shadow_work["group_spec_evaluations"] += 1
                self.shadow_work["executed_candidate_pairs"] += executed
                self.shadow_work["logical_candidate_pairs"] += pairs
                self.shadow_work["executed_av_multiply_adds"] += heads * tokens * dimension

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if query.shape[-2] != 1 or self.step < 0:
            if self.original is None:
                raise RuntimeError("use OracleAttention as a context manager")
            return self.original(module, query, key, value, attention_mask, **kwargs)
        if kwargs.get("dropout", 0.) != 0.:
            raise ValueError("Oracle decode supports inference dropout=0 only")
        if attention_mask is not None:
            visible = attention_mask.all() if attention_mask.dtype == torch.bool else (attention_mask == 0).all()
            torch._assert_async(visible, "Oracle decode requires all cached tokens visible")
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = query.shape[-1] ** -.5
        logits = grouped_logits(query, key, scaling)
        selection = None
        if self.algorithm != "dense":
            selection = self._selection(torch_page_mass(logits), self.spec, logits.shape[-1])
        output = grouped_output(logits, value, None if selection is None else selection.mask)
        self._record_actual(logits, value, selection)
        layer = getattr(module, "layer_idx", None)
        sample = (self.shadow_specs and (self.sample_layers is None or layer in self.sample_layers)
                  and (self.sample_steps is None or self.step in self.sample_steps))
        if sample:
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            started = time.perf_counter()
            self._audit(module, logits, value, output)
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            self.shadow_audit_seconds += time.perf_counter() - started
        result = output.reshape(1, query.shape[1], 1, value.shape[-1])
        return result.to(query.dtype).transpose(1, 2).contiguous(), None
