"""Shared-state sampling for Round 11 quality diagnostics and physical replays."""
from dataclasses import replace

import numpy as np
import torch

from .compact_summary import CompactOnline, CompactPrefix, SUMMARY_BYTES, weights
from .fixed_k_v import ranked_pages, vector_burst
from .oracle_attention import grouped_logits, grouped_output
from .page_read_budget import page_history
from .pagepack_budget import K_BASE, V_BASE
from .round6_memory import merge, value_runs
from .round7_attention import Attention
from .round11_workpoints import allocation
from .summary_decode import screen


SAMPLE_LAYERS = (0, 7, 14, 21, 28, 35)
SAMPLE_STEPS = (0, 7, 15, 31)


def trace_group(policy, plan, group, mask, ids, counts, ledger, prefix, fallback, heads, dim):
    history = plan.history
    page_size = history.page_size
    summary_stride = SUMMARY_BYTES[policy.method]
    scorer_channels = 128 if policy.method == "mr25" else 96
    stride = vector_burst(page_size)
    full = [dict(id=page, records=count, addr=K_BASE + page * stride,
                 bytes=vector_burst(count), v_addr=V_BASE + page * stride)
            for page, count in zip(ids, counts, strict=True)]
    # [page, head, token]. The union pays packed code, scale and ZP once.
    union = mask.any(1)
    ranges = []
    for i in sorted(range(len(ids)), key=ids.__getitem__):
        ranges.extend(value_runs(union[i], counts[i], full[i]["v_addr"]))
    runs = merge(ranges)
    if sum(x["bytes"] for x in full) != ledger["k_bytes"] or sum(size for _, size in runs) != ledger["v_bytes"]:
        raise ValueError("hardware trace disagrees with the physical debit")
    if sum(ledger[k] for k in ("summary_bytes", "k_bytes", "v_bytes")) != ledger["total"]:
        raise ValueError("unaccounted physical debit")
    pv_by_head = mask.sum((0, 2)).tolist()
    hex_masks = [[bits[::-1].tobytes().hex()
                  for bits in page] for page in np.packbits(mask, axis=-1, bitorder="little")]
    union_masks = [bits[::-1].tobytes().hex()
                   for bits in np.packbits(union, axis=-1, bitorder="little")]
    return dict(policy=policy.name, group=group, summary_stride=summary_stride,
                summary_address=0, summary_bytes=ledger["summary_bytes"], history_pages=history.pages,
                page_size=page_size, history_tokens=history.tokens, head_dim=dim, heads=heads,
                scorer_channels=scorer_channels, scorer_terms=2 * history.pages * heads * scorer_channels,
                fused_updates=history.pages * heads * scorer_channels, k_pages=full, v_runs=runs,
                pv_masks=hex_masks, v_union_masks=union_masks, pv_by_head=pv_by_head,
                initial_prefix=len(ids), final_prefix=prefix, gate_tokens=sum(counts),
                prefix_records=sum(counts[:prefix]), fallback=bool(fallback), cap_bytes=plan.cap,
                qk_mac=sum(counts) * heads * dim, pv_mac=sum(pv_by_head) * dim,
                k_bytes=ledger["k_bytes"], v_bytes=ledger["v_bytes"], v_records=ledger["v_records"],
                protected_records=sum(counts[:len(history.protected)]), total_bytes=ledger["total"])


class CommonAttention(Attention):
    def __init__(self, dense_policy, policies, masks, spans, starts, sample_id, cohort, index, whole):
        super().__init__(dense_policy, None, spans)
        self.policies, self.masks, self.starts = policies, masks, starts
        self.identity = dict(sample_id=sample_id, cohort=cohort)
        self.index, self.whole = index, whole
        self.metrics = []
        self.traces = []

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        layer = module.layer_idx
        whole = self.step == self.whole
        if (self.step < 0 or query.shape[-2] != 1 or
                (not whole and (layer not in SAMPLE_LAYERS or
                 (self.step not in SAMPLE_STEPS and self.step not in self.starts)))):
            return super().attention(module, query, key, value, attention_mask, **kwargs)
        groups, tokens, dim = key.shape[1:]
        q = query.reshape(groups, -1, dim).float()
        heads = q.shape[1]
        scale = kwargs.get("scaling") or dim ** -.5
        dense = grouped_output(grouped_logits(query, key, scale), value)
        history = page_history(tokens, 32)
        methods = tuple(dict.fromkeys(p.method for p in self.policies))
        channels = {method: by_layer[layer] for method, by_layer in self.masks.items()}
        prefix = CompactPrefix.build(key[:, :, :-1], 32, methods, channels)
        records = CompactOnline(prefix).append(key[0, :, -1], tokens)
        ranks = {method: ranked_pages(weights(q, records[method], method, scale,
                                             channels=channels.get(method)), history)
                 for method in methods}
        gold = self.starts.get(self.step, self.gold_spans)
        plans = {p.name: allocation(p, ranks[p.method], history) for p in self.policies}
        a, b = plans["A"], plans.get("B")
        if b is not None:
            pins = len(history.protected)
            if a.ids.shape != b.ids.shape or not torch.equal(a.ids[:, :pins], b.ids[:, :pins]):
                raise ValueError("matched A/B pages or protection diverged")
            if not torch.equal(a.k_counts, b.k_counts):
                raise ValueError("matched A/B valid K and QK differ")
        outcomes = {p.name: screen(plans[p.name], q, key, value, scale, gold_spans=gold)
                    for p in self.policies}
        pressure = torch.stack([out.fallback for out in outcomes.values()]).any(0).tolist()
        for policy in self.policies:
            alloc, out = plans[policy.name], outcomes[policy.name]
            self.record_metrics(policy.name, alloc, out, dense, heads, dim, layer, whole)
            chosen = [i for i in range(groups) if whole or pressure[i] or
                      i in {self.index % groups, (self.index + groups // 2) % groups}]
            masks = out.final_mask.transpose(1, 2).detach().cpu().numpy()
            ids, counts = alloc.ids.tolist(), alloc.k_counts.tolist()
            ledger = {name: values.tolist() for name, values in out.ledger.items()}
            fallback, safe = out.fallback.tolist(), out.prefix.tolist()
            for group in chosen:
                record = trace_group(policy, alloc, group, masks[group], ids[group], counts[group],
                                     {n: values[group] for n, values in ledger.items()}, safe[group],
                                     fallback[group], heads, dim)
                self.traces.append(dict(**record, **self.identity, layer=layer, step=self.step,
                                        scope="fallback_pressure" if pressure[group] else "regular",
                                        whole_step=whole, answer_start=self.step in self.starts))
        if b is not None and torch.any(outcomes["A"].fallback != outcomes["B"].fallback).item():
            # Match A's V headroom, while keeping the B ranking and accounting.

            check = replace(b, cap=a.cap - a.summary_bytes + b.summary_bytes, safe_count=a.safe_count)
            res = screen(check, q, key, value, scale, gold_spans=gold)
            self.record_metrics("B_matched_v", check, res, dense, heads, dim, layer, whole)
            self.record_metrics("A_matched_v", a, outcomes["A"], dense, heads, dim, layer, whole)
            self.record_metrics("B_own_v_subset", b, outcomes["B"], dense, heads, dim, layer, whole)
        return self.output(query, dense)

    def record_metrics(self, name, allocation, out, dense, heads, dim, layer, whole):
        error = (out.output - dense).norm(dim=-1)
        relative = error / (dense.norm(dim=-1) + 1e-12)
        row = dict(policy=name, layer=layer, step=self.step, whole_step=whole,
                   answer_start=self.step in self.starts,
                   qk_mac=out.ledger["k_records"] * heads * dim,
                   pv_mac=out.final_mask.sum((1,2,3)) * dim,
                   pages=allocation.count, prefix=out.prefix, fallback=out.fallback,
                   read_k_loss=out.loss_read_k, final_prefix_loss=out.loss_prefix,
                   abs_error=error, rel_error=relative, **out.evidence,
                   **{n: out.ledger[n] for n in ("total", "summary_bytes", "k_bytes", "v_bytes", "k_records",
                                                 "v_records", "over_bytes")})
        self.metrics.append({k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in row.items()})
