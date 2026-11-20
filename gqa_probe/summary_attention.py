"""Calibration attention and same-state diagnostics for the read-budget study."""

from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch

from .oracle_attention import OracleAttention, grouped_logits, grouped_output, torch_page_mass
from .page_read_budget import SUMMARY_VECTORS, page_history, read_ledger
from .page_summary import OnlineSummary, decode_vector, gathered_output, select_ranked, summary_weights


SAMPLED_LAYERS = (0, 7, 14, 21, 28, 35)
SAMPLED_STEPS = (0, 1, 3, 7, 15, 31, 63, 127)
LAYOUT_METHODS = ("quest_softmax", "quest_max", "centroid8")


@dataclass(frozen=True)
class Policy:
    method: str
    page_size: int = 32
    percent: int | None = None
    quota_percent: int | None = None
    resident_bytes: int = 0

    @property
    def id(self):
        kind = f"b{self.percent}" if self.percent is not None else f"p{self.quota_percent or 100}"
        return f"{self.method}_P{self.page_size}_{kind}" + (f"_sram{self.resident_bytes}" if self.resident_bytes else "")


class SummaryAttention(OracleAttention):
    """Round 3 Mask reference. Policy selection reads only encoded metadata."""

    def __init__(self, policy, *, gold_spans=(), prefixes=None, gather_check=False, audit_policies=()):
        super().__init__("dense", gold_spans=gold_spans, page_size=policy.page_size)
        self.policy = policy
        self.prefixes = prefixes or {}
        self.online = {}
        self.gather_check = gather_check
        self.audit_policies = tuple(p for p in audit_policies if p.percent is not None)
        self.gather_errors = []
        self.diagnostics = {}
        self.ledger = Counter()
        self.audit_spans = {size: sorted({token // size for a, b in self.gold_spans for token in range(a, b)})
                            for size in self.prefixes}

    def _pages(self, layer, size, key):
        tracker = (size, layer)
        if tracker not in self.online:
            self.online[tracker] = OnlineSummary(self.prefixes[size][layer])
        return self.online[tracker].append(key[0, :, -1], key.shape[-2])

    def _audit_same_state(self, query, key, value, logits, dense, histories, layer, scaling):
        for size in self.prefixes:
            history = histories[size]
            pages = self._pages(layer, size, key)
            if self.step == 0:
                sealed = key[0, :, :-history.tail_size].unflatten(1, (history.pages-1, size))
                tail = key[0, :, -history.tail_size:].unsqueeze(1)
                minimum = torch.cat((sealed.amin(2), tail.amin(2)), dim=1)
                maximum = torch.cat((sealed.amax(2), tail.amax(2)), dim=1)
                torch._assert_async((decode_vector(pages.min_key) <= minimum).all() &
                                    (decode_vector(pages.max_key) >= maximum).all(),
                                    "decoded Quest record fails to enclose the live keys")
            mass = torch_page_mass(logits, size)
            rank = mass.sum(-1)
            scores = {"oracle": rank}
            for method in (*LAYOUT_METHODS, "centroid_fp32"):
                scores[method] = summary_weights(query, pages, method, scaling)
            selections = []
            for fraction in (5, 10):
                count = history.pages * fraction // 100
                if count < len(history.protected):
                    continue
                selections.extend((f"{method}_P{size}_p{fraction}", method, count, False) for method in scores)
            for policy in self.audit_policies:
                if policy.page_size != size:
                    continue
                ledger = read_ledger(history, policy.method, key.shape[-1], percent=policy.percent,
                                     resident_bytes=policy.resident_bytes)
                if ledger["feasible"]:
                    selections.append((policy.id, policy.method, ledger["selected_pages"],
                                       self.step == 0 and policy.percent == 15 and not policy.resident_bytes))
            gold = self.audit_spans[size]
            dense_norm = dense.norm(dim=-1)
            masks = {}
            for name, method, count, check in selections:
                if count not in masks:
                    masks[count] = select_ranked(rank, history.protected, count)
                oracle = masks[count]
                mask = select_ranked(scores[method], history.protected, count)
                output = grouped_output(logits, value, mask, size)
                error = (output - dense).norm(dim=-1)
                relative = error / (dense_norm + 1e-12)
                retained = (mass * mask[..., None]).sum(1)
                optional_hits = (mask & oracle).sum(-1) - len(history.protected)
                overlap = optional_hits.float().mean() / max(1, count - len(history.protected))
                recall = mask[:, gold].float().mean() if gold else torch.zeros((), device=mask.device)
                data = self.diagnostics.setdefault(name, ([], []))
                data[0].append(torch.stack((error, relative, retained), dim=0).reshape(3, -1))
                data[1].append(torch.stack((overlap, recall.to(overlap.dtype), torch.full_like(overlap, count))))
                if check and self.gather_check:
                    gathered = gathered_output(query, key, value, mask, history, count, scaling)
                    torch._assert_async(torch.isclose(gathered, output, atol=2e-4, rtol=2e-4).all(),
                                        "summary Gather differs from Mask")
                    self.gather_errors.append((gathered - output).abs().amax())

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        if kwargs.get("dropout", 0.) != 0.:
            raise ValueError("supports only deterministic decode")
        if attention_mask is not None:
            valid = attention_mask.all() if attention_mask.dtype == torch.bool else (attention_mask == 0).all()
            torch._assert_async(valid, "expected fully visible history")
        policy, layer = self.policy, module.layer_idx
        size = policy.page_size
        _, groups, tokens, dim = key.shape
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = dim ** -.5
        history = page_history(tokens, size, dim)
        mask = None
        method = policy.method
        if method in SUMMARY_VECTORS:
            ledger = read_ledger(history, method, dim, percent=policy.percent, resident_bytes=policy.resident_bytes)
            if not ledger["feasible"]:
                raise ValueError(f"required reads exceed policy ceiling at T={tokens}: {policy}")
            count = ledger["selected_pages"]
            metadata = self._pages(layer, size, key)
            mask = select_ranked(summary_weights(query, metadata, method, scaling), history.protected, count)
            for field, val in ledger.items():
                if isinstance(val, int) and not isinstance(val, bool):
                    self.ledger[field] += groups * val
            vectors = SUMMARY_VECTORS[method]
            self.ledger["score_mac"] += groups * vectors * query.shape[1] // groups * history.pages * dim
            self.ledger["summary_scan_passes"] += groups
            self.ledger["summary_rereads"] += 0
            self.ledger["scanned_score_values"] += query.shape[1] * history.pages
            self.ledger["summary_update_vector_ops"] += groups * vectors * dim
            self.ledger["max_fp32_score_sram_bytes_per_group"] = max(
                self.ledger["max_fp32_score_sram_bytes_per_group"],
                (query.shape[1] // groups + 1) * history.pages * 4)
        # Complete QK serves the Mask backend. The summary mask is fixed above.
        logits = grouped_logits(query, key, scaling)
        if method == "oracle":
            count = history.pages * policy.quota_percent // 100
            mask = select_ranked(torch_page_mass(logits, size).sum(-1), history.protected, count)
        if method in ("dense_bf16", "dense_kv4"):
            count = history.pages
        if method == "dense_kv4":
            self.ledger["total_read_bytes"] += groups * history.dense.total
            self.ledger["dense_read_bytes"] += groups * history.dense.total
        output = grouped_output(logits, value, mask, size)
        self._record_gold_aggregate(mask, groups, tokens, key.device)
        self._add("decode_layer_calls", 1)
        self._add("kv_group_calls", groups)
        self._add("selected_pages", groups * count)
        self._add("available_pages", groups * history.pages)
        sample = self.prefixes and method == "dense_kv4" and layer in SAMPLED_LAYERS
        if sample:
            histories = {p: page_history(tokens, p, dim) for p in self.prefixes}
            if self.step in SAMPLED_STEPS:
                self._audit_same_state(query, key, value, logits, output, histories, layer, scaling)
            else:
                for p in self.prefixes:
                    self._pages(layer, p, key)
        return output.reshape(1, query.shape[1], 1, dim).to(query.dtype).transpose(1, 2).contiguous(), None

    def stats(self):
        result = super().stats()
        result["byte_ledger"] = dict(self.ledger)
        result["gather_max_abs_difference"] = max((x.item() for x in self.gather_errors), default=None)
        diagnostic_rows = []
        for policy, (heads, groups) in self.diagnostics.items():
            errors = torch.cat(heads, dim=1).cpu().numpy()
            mean_group = torch.stack(groups).mean(0).cpu().tolist()
            absolute, relative = np.quantile(errors[:2], [.95, .99], axis=1).T
            diagnostic_rows.append(dict(policy=policy,
                                        absolute_p95=absolute[0], absolute_p99=absolute[1],
                                        relative_p95=relative[0], relative_p99=relative[1],
                                        coverage_p05=float(np.quantile(errors[2], .05)),
                                        retained_mass=float(errors[2].mean()),
                                        optional_oracle_overlap=mean_group[0],
                                        gold_page_recall=mean_group[1] if self.gold_spans else None,
                                        selected_pages=mean_group[2], sampled_heads=errors.shape[1]))
        result["same_state_diagnostics"] = diagnostic_rows
        return result
