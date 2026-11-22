"""Common-prefix screens for summary reuse and cold-page compensation."""

from dataclasses import dataclass
from math import ceil
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

from .fixed_k_v import burst_round, byte_quota, ranked_pages
from .round6_memory import read_phases, transaction_trace
from .vskip_summary import METHOD_BYTES, weights as summary_weights


def summary_bursts(pages, record_bytes):
    """Coalesce selected record addresses, including shared boundary bursts."""
    runs = []
    for page in sorted(set(pages)):
        start = (page * record_bytes) // 32 * 32
        end = burst_round((page + 1) * record_bytes)
        if runs and start <= runs[-1][1]:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return [(start, end - start) for start, end in runs]


def compensation_outputs(logits, value, hot_pages, h):
    """Return dense, hot-only, and hot plus exact cold page mass times mean V."""
    # This is an offline oracle. Reduce in FP64 so broad, 32K score
    # distributions do not trip the mass-conservation check from FP32 error.
    scores = logits.double()
    p = scores.softmax(-1)
    values = value[0].double()
    hot = hot_pages.repeat_interleave(h.page_size, dim=-1)[:, None, :h.tokens]
    hot_p = p * hot
    hot_mass = hot_p.sum(-1, keepdim=True)
    hot_contribution = torch.bmm(hot_p, values)
    # Normalize hot-only in logit space to handle a tiny dense hot mass.
    hot_only = torch.bmm(scores.masked_fill(~hot, -torch.inf).softmax(-1), values)
    pad = h.pages * h.page_size - h.tokens
    page_mass = F.pad(p, (0, pad)).unflatten(-1, (h.pages, h.page_size)).sum(-1)
    page_sums = F.pad(values, (0, 0, 0, pad)).unflatten(1, (h.pages, h.page_size)).sum(2)
    lengths = torch.full((h.pages,), h.page_size, device=value.device)
    lengths[-1] = h.tail_size
    means = page_sums / lengths[None, :, None]
    cold_mass = page_mass * ~hot_pages[:, None]
    corrected = hot_contribution + torch.bmm(cold_mass, means)
    torch._assert_async(((cold_mass.sum(-1) + hot_mass.squeeze(-1) - 1).abs() < 1e-10).all(),
                        "cold and hot probabilities must share a full-history denominator")
    return torch.bmm(p, values).to(logits.dtype), hot_only.to(logits.dtype), corrected.to(logits.dtype)


def value_start_pages(tokenizer, tokens, evidence, page_size=32):
    """Map decode query steps to source association pages of newly output values."""
    rendered = tokenizer.decode(tokens, skip_special_tokens=True).lower()
    hits = [(rendered.find(item["value"].lower()), item["association_span"]) for item in evidence]
    hits = [(start, span) for start, span in hits if start >= 0]
    if not hits:
        return {}
    lengths = [len(tokenizer.decode(tokens[:n], skip_special_tokens=True))
               for n in range(1, len(tokens) + 1)]
    steps = {}
    for start, (lo, hi) in hits:
        answer_index = next(i for i, n in enumerate(lengths) if n > start)
        if answer_index > 0:  # Index zero comes from prefill; it has no decode query.
            steps.setdefault(answer_index - 1, set()).update(range(lo // page_size, (hi - 1) // page_size + 1))
    return steps


def _selected(priority, h, count):
    mandatory = set(h.protected)
    remote = [page for page in priority if page not in mandatory and page < h.pages]
    if len(remote) + len(mandatory) < count:
        raise ValueError("cached ranking does not contain enough pages after restoring protection")
    return [*h.protected, *remote[:count - len(mandatory)]]


def _score_cached(q, record, scale, shortlist, method):
    ids = torch.tensor(sorted(shortlist), dtype=torch.long, device=q.device)
    summary = tuple(x.index_select(1, ids) for x in record)
    scores = summary_weights(q, summary, method, scale)
    ranking = scores[0].argsort(descending=True, stable=True)
    return ids[ranking].cpu().tolist()


def _replay(h, picked, scan, heads, dim, method, compare):
    valid = {p: np.arange(h.page_size) < min(h.page_size, h.tokens - p * h.page_size) for p in picked}
    phases, _ = transaction_trace(h, picked, [], valid, 0)
    metadata = summary_bursts(scan, METHOD_BYTES[method])
    # Emit index requests after the current selector has produced its list.
    phases = [metadata, phases[0], phases[1], phases[3]]
    # Per-group fixed lanes, as in Round 6. Cosine uses the current norm and
    # dot product against a cached reference norm, plus a comparator margin.
    score_ck = ceil(dim * len(scan) / 16) + ceil(heads * len(scan) / 16) * 2
    score_ck += 12 * heads + ceil(len(scan) / 4) if scan else 0
    compare_ck = ceil(2 * dim / 16) + 12 if compare else 0
    k_tokens = sum(int(mask.sum()) for mask in valid.values())
    timing = read_phases(phases, (compare_ck, score_ck, 0, 8 * k_tokens))
    return dict(bursts=timing[4], read_bytes=32 * timing[4], summary_bursts=sum(s // 32 for _, s in metadata),
                row_hits=timing[5], row_activations=timing[6], scorer_ck=score_ck, trigger_ck=compare_ck,
                before_first_v_ck=timing[2] + 8 * k_tokens, dependency_ck=timing[3] + 8 * k_tokens,
                qk_mac=heads * k_tokens * dim, pv_mac=heads * k_tokens * dim)


@dataclass
class _LayerCache:
    step: int
    interval_step: int
    interval_pages: list
    reference_query: torch.Tensor
    reference_step: list
    cosine_pages: list


class SideScreen:
    """Record successive steps from a shared Dense-KV4 decode trajectory.

    The reuse probe keeps a 128-remote-page shortlist from a full scan. It
    restores current protection before selection, and charges every current
    protection summary on a shortlist rescore. Subset softmax is intentional.
    The cold probe fixes the full-scan hot pages and uses an offline mass oracle.
    """

    def __init__(self, *, layers=(7, 21), page_count=None, top_count=128, interval=4, cosine=.95,
                 cold_steps=(0, 1, 7, 15, 31, 63), method="q6", budget=10):
        if interval < 1 or top_count < 1 or not -1 <= cosine <= 1:
            raise ValueError("invalid reuse parameters")
        self.layers = tuple(layers)
        self.page_count = page_count
        self.top_count = top_count
        self.interval = interval
        self.cosine = cosine
        self.cold_steps = set(cold_steps)
        self.method = method
        self.budget = budget
        self.cache = {}
        self.rows = []
        self.cold = []

    @torch.inference_mode()
    def observe(self, step, layer, h, q, key, value, logits, ranks, weights, record, scale, evidence,
                *, new_value_head=False):
        if layer not in self.layers:
            return
        count = (byte_quota(h.tokens, h.page_size, self.method, self.budget)
                 if self.page_count is None else self.page_count)
        if not len(h.protected) <= count <= h.pages:
            raise ValueError("side-screen page count must include the protection set")
        if ranks is None:
            ranks = ranked_pages(weights, h)
        groups, heads, dim = q.shape
        mandatory = set(h.protected)
        full = ranks[:, :max(count, self.top_count) + len(mandatory)].cpu().tolist()
        remote = [[p for p in ranking if p not in mandatory] for ranking in full]
        evidence = set(evidence)
        cache = self.cache.get(layer)
        first = cache is None
        if first:
            cache = _LayerCache(step, step, [], q.clone(), [step] * groups, list(remote))
            self.cache[layer] = cache
            similarities = [None] * groups
            refresh = [True] * groups
        else:
            if cache.step != step - 1:
                raise ValueError("reuse screening requires consecutive decode states for each sampled layer")
            similarities = F.cosine_similarity(q, cache.reference_query, dim=-1).amin(-1).cpu().tolist()
            refresh = [similarity < self.cosine for similarity in similarities]
            cache.step = step
        fixed_refresh = first or step - cache.interval_step >= self.interval
        if fixed_refresh:
            cache.interval_step = step
            cache.interval_pages = [ranking[:count] for ranking in remote]
        full_picks = []
        for group in range(groups):
            if refresh[group]:
                cache.reference_query[group].copy_(q[group])
                cache.reference_step[group] = step
                cache.cosine_pages[group] = remote[group]
            selected = _selected(full[group], h, count)
            full_picks.append(selected)
            shortlist = [p for p in cache.cosine_pages[group] if p not in mandatory][:self.top_count]
            scan = [*h.protected, *shortlist]
            reranked = full[group] if refresh[group] else _score_cached(
                q[group:group + 1], tuple(x[group:group + 1] for x in record), scale, scan, self.method)
            routes = (
                ("full", full[group], True, step, 0),
                ("interval", cache.interval_pages[group], fixed_refresh, cache.interval_step, count * 4 + 4),
                ("cosine", cache.cosine_pages[group][:count], refresh[group], cache.reference_step[group],
                 count * 4 + heads * (dim * 2 + 4) + 4),
                ("rerank128", reranked, refresh[group], cache.reference_step[group],
                 max(count, self.top_count) * 4 + heads * (dim * 2 + 4) + 4),
            )
            for name, ranking, full_scan, reference_step, state_bytes in routes:
                picked = _selected(ranking, h, count)
                records = range(h.pages) if full_scan else scan if name == "rerank128" else ()
                physical = _replay(h, picked, records, heads, dim, self.method,
                                   not first and name in ("cosine", "rerank128"))
                self.rows.append(dict(step=step, layer=layer, group=group, strategy=name, tokens=h.tokens,
                                      full_scan=full_scan, reference_step=reference_step,
                                      cosine_min=similarities[group], summary_records=len(records),
                                      selected_pages=picked, state_bytes_fp16_query=state_bytes,
                                      gold_hits=len(evidence.intersection(picked)), gold_pages=len(evidence),
                                      full_k_overlap=len(set(selected).intersection(picked)),
                                      new_value_head=new_value_head, **physical))
        if step in self.cold_steps or new_value_head:
            self._cold_row(step, layer, h, key, value, logits, full_picks, new_value_head)

    def _cold_row(self, step, layer, h, key, value, logits, picked, new_value_head):
        hot = torch.zeros((len(picked), h.pages), dtype=torch.bool, device=key.device)
        hot.scatter_(1, torch.tensor(picked, device=key.device), True)
        dense, hot_out, corrected = compensation_outputs(logits, value, hot, h)
        err1 = (hot_out - dense).norm(dim=-1)
        err2 = (corrected - dense).norm(dim=-1)
        denom = dense.norm(dim=-1) + 1e-8
        pairs = torch.arange(step % 16, h.tokens - 1, 16, device=key.device)
        similarities = F.cosine_similarity(key[0, :, pairs].float(), key[0, :, pairs + 1].float(), dim=-1)
        adjacent_hits = (similarities >= .95).sum(-1).cpu().tolist()
        per_head = torch.stack((err1, err2, err1 / denom, err2 / denom), dim=1).cpu().tolist()
        for g, (e1, e2, r1, r2) in enumerate(per_head):
            mean_payload = ceil(value.shape[-1] / 2) + 2 + 1  # INT4, FP16 scale, uint8 valid count.
            bytes_mean_layout = ceil(mean_payload / 16) * 16
            mean_bursts = summary_bursts([p for p in range(h.pages) if p not in picked[g]], bytes_mean_layout)
            self.cold.append(dict(step=step, layer=layer, group=g, new_value_head=new_value_head,
                                  hot_abs=e1, cold_oracle_abs=e2, hot_relative=r1, cold_oracle_relative=r2,
                                  cold_vmean_payload_per_page=mean_payload,
                                  cold_vmean_aligned_burst_bytes=sum(s for _, s in mean_bursts),
                                  adjacent_cosine_ge_095=adjacent_hits[g], adjacent_pairs=len(pairs)))

    def finalize(self, *, value_starts=None):
        rows = self.rows
        if value_starts:
            rows = []
            for row in self.rows:
                pages = value_starts.get(row["step"])
                if pages:
                    row = dict(row, new_value_head=True, value_start_pages=len(pages),
                               value_start_hits=len(pages.intersection(row["selected_pages"])))
                rows.append(row)
        result = dict(rows=rows, cold=self.cold)
        return {**result, "summary": summarize_side_screens([result])}


def summarize_side_screens(payloads):
    """Aggregate public per-group observations without carrying prompt text."""
    rows = [row for prompt in payloads for row in prompt["rows"]]
    cold = [row for prompt in payloads for row in prompt["cold"]]

    def recall(batch, at_value_start=False):
        total = sum(x.get("value_start_pages", x["gold_pages"]) if at_value_start else x["gold_pages"] for x in batch)
        hits = sum(x.get("value_start_hits", x["gold_hits"]) if at_value_start else x["gold_hits"] for x in batch)
        return hits / total if total else None

    routes = {}
    for name in ("full", "interval", "cosine", "rerank128"):
        route = [row for row in rows if row["strategy"] == name]
        if route:
            value_start = [row for row in route if row["new_value_head"]]
            routes[name] = dict(observations=len(route), value_start_observations=len(value_start),
                                depth_mean=mean(len(r["selected_pages"]) for r in route),
                                refresh_rate=mean(r["full_scan"] for r in route),
                                gold_page_recall=recall(route), value_start_page_recall=recall(value_start, True),
                                avg_dependency_ck=mean(r["dependency_ck"] for r in route),
                                avg_summary_records=mean(r["summary_records"] for r in route),
                                avg_read_bytes=mean(r["read_bytes"] for r in route),
                                peak_state_bytes_fp16_query=max(r["state_bytes_fp16_query"] for r in route))
    cold_result = {}
    if cold:
        for key in ("hot_abs", "cold_oracle_abs", "hot_relative", "cold_oracle_relative"):
            values = [value for row in cold for value in row[key]]
            cold_result[f"{key}_p95"], cold_result[f"{key}_p99"] = np.quantile(values, [.95, .99]).tolist()
        pairs = sum(r["adjacent_pairs"] for r in cold)
        cold_result.update(observations=len(cold), adjacent_pairs=pairs,
                           adjacent_cosine_ge_095_fraction=(sum(r["adjacent_cosine_ge_095"] for r in cold) / pairs
                                                           if pairs else None),
                           avg_cold_vmean_aligned_burst_bytes=mean(r["cold_vmean_aligned_burst_bytes"] for r in cold))
    return {"summary_reuse": routes, "cold_compensation": cold_result}
