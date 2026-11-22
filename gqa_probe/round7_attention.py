"""Round 7 decode reference and common-state LPDDR accounting."""
from dataclasses import dataclass
from math import ceil

import torch

from .fixed_k_v import byte_quota, gather_tokens, masked_output, ranked_pages
from .oracle_attention import OracleAttention, grouped_logits, grouped_output
from .page_read_budget import page_history
from .round6_attention import SAMPLE_LAYERS, SAMPLE_STEPS, dense_bytes, physical_ledger
from .round6_audit import compute_cycles, dense_timing
from .round6_memory import simulate
from .round7_gates import gate_cost
from .round7_single import decide_v, prepare_single, SingleOutcome
from .shared_refill import RefillPolicy, run_shared
from .vskip_summary import METHOD_BYTES, StaticOnline, weights


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str = "single"
    method: str = "q6"
    budget: int = 10
    max_pages: int = 68
    gate: str = "exact"
    threshold: float = 0.
    bonus: int | None = None

    def page_limit(self, tokens):
        if self.bonus is None:
            return self.max_pages
        return min(self.max_pages, byte_quota(tokens, 32, self.method, self.budget) + self.bonus)


def timing_rows(h, policy, ranks, out, led):
    k = out.final
    ids = k.ids.cpu().tolist()
    active = k.valid.any(-1).cpu().tolist()
    heads = k.logits.shape[1]
    union = k.mask.any(1).cpu().numpy()
    actual = led["total"].cpu().tolist()
    ktokens = led["k_tokens"].cpu().tolist()
    pvtokens = k.mask.sum((1, 2, 3)).cpu().tolist()
    starters = ranks[:, :out.initial_count].cpu().tolist()
    extra = out.additional.cpu().tolist()
    if isinstance(out, SingleOutcome):
        fallback = out.fallback.cpu().tolist()
        prefix = out.prefix_pages.cpu().tolist()
        lost = out.lost_all.cpu().tolist()
        lost_prefix = out.lost_prefix.cpu().tolist()
        over = out.initial_over_bytes.cpu().tolist()
    else:
        fallback = [False] * len(ids)
        prefix = [out.initial_count+n for n in extra]
        lost = lost_prefix = [None] * len(ids)
        over = [0] * len(ids)
    for i, pages in enumerate(ids):
        initial = starters[i]
        added = pages[out.initial_count:out.initial_count + extra[i]]
        reads = {p: val for p, val, live in zip(pages, union[i], active[i]) if live}
        kt = ktokens[i]
        nt_in = sum(min(32, h.tokens-32*p) for p in initial)
        nt_add = kt - nt_in
        nprot = sum(min(32, h.tokens-32*p) for p in h.protected)
        gate = "hist64" if policy.kind == "dynamic" else policy.gate
        # Dynamic selections in the closure use exact Top-p. Histogram costs
        # are explicitly a modeled architecture proxy for this dynamic path.
        own = gate_cost(gate, nt_in, heads, optional_tokens=nt_in-nprot, pages=len(initial))
        more = gate_cost(gate, nt_add, heads, pages=len(added)) if added else None
        if more:
            # Re-normalize against the widened candidate set and update mass
            # retained by the locked first mask before gating appended tokens.
            read_old = nt_in*heads
            more["gate_ck"] += ceil(read_old/4)
            more["logits_sram_reads"] += read_old
            more["gate_exponentials"] += read_old
            more["gate_reduce_adds"] += 2*read_old
            more["mask_sram_reads"] += read_old
        redo = gate_cost("exact", nprot+(prefix[i]-len(h.protected))*32, heads,
                         optional_tokens=(prefix[i]-len(h.protected))*32,
                         pages=prefix[i]) if fallback[i] else None
        cycle = compute_cycles(h, heads, nt_in, nt_add, heads, "none")
        score_ck = sum(cycle[x] for x in ("score_dot_ck","normalize_ck","topk_insert_ck"))
        summary_dots = 2 if policy.method.startswith("mr") else 1
        score_ck += (summary_dots-1)*cycle["score_dot_ck"]
        budget_ck = len(initial) + (1 + prefix[i] if redo else 0) if policy.kind == "single" else 0
        first_gate_ck = own["gate_ck"]
        second_gate_ck = (more or redo or {}).get("gate_ck", 0)
        b1 = cycle["initial_qk_ck"] + first_gate_ck
        b2 = cycle["extra_qk_ck"] + second_gate_ck + budget_ck
        wait = (0, score_ck, b1 if added else 0, b2 if added else b1+b2)
        m = simulate(h, initial, added, reads, h.pages*METHOD_BYTES[policy.method], wait,
                     refill_index=True)
        if m["reads_bytes"] != actual[i] or actual[i] > out.ceiling:
            raise ValueError("round 7 modeled trace and exact budget ledger disagree")
        pv_fixed = 8*sum(int(x.sum()) for x in reads.values())
        # One shared logits buffer and one mask set. Refills hold both sets;
        # fallback reuses scratch, but charges an additional gate scan.
        control_peak = (max(own["control_sram_bytes"], (redo or {}).get("control_sram_bytes", 0)) +
                        (more["head_mask_bytes"]+more["union_mask_bytes"] if more else 0)
                        + (2*4 if policy.kind == "single" else 0))
        counts = {k: own[k]+sum(x[k] for x in (more, redo) if x is not None)
                  for k in own if k.startswith(("gate_", "histogram_", "mask_sram_", "logits_sram_"))
                  and k not in ("gate_ck", "logits_sram_bytes", "histogram_bytes", "logits_sram_writes")}
        yield dict(group=i, **m, ceiling_bytes=out.ceiling, over_budget=False,
                   before_fallback_over_bytes=over[i], fallback=fallback[i],
                   k_pages=len(initial)+len(added), prefix_pages=prefix[i],
                   added_k_pages=len(added), lost_read_k=lost[i], lost_prefix=lost_prefix[i],
                   qk_mac=kt*heads*128, pv_mac=pvtokens[i]*128, scorer_mac=summary_dots*heads*h.pages*128,
                   total_group_ck=m["lpddr_ck"]+pv_fixed, control_sram_bytes=control_peak,
                   logits_sram_bytes=kt*heads*4, logits_sram_writes=kt*heads,
                   histogram_bytes=own["histogram_bytes"], gate_ck=first_gate_ck+second_gate_ck,
                   score_ck=score_ck, budget_check_ck=budget_ck, gate_model=gate,
                   fallback_gate_model="exact" if redo else None, **counts)


class Attention(OracleAttention):
    def __init__(self, policy, prefix, gold, *, audit=False):
        super().__init__("dense", gold_spans=gold, page_size=32)
        self.policy = policy
        self.prefix = prefix
        self.audit = audit
        self.online = {}
        self.timing = []
        self.peak_pages = None
        self.peak_loss = {}
        self.group_stats = []

    def output(self, query, y):
        return y.reshape(1, query.shape[1], 1, -1).to(query.dtype).transpose(1, 2).contiguous(), None

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        g, t, dim = key.shape[1:]
        h = page_history(t, 32)
        q = query.reshape(g, -1, dim).float()
        scale = kwargs.get("scaling") or dim**-.5
        layer = module.layer_idx
        policy = self.policy
        heads = q.shape[1]
        self._add("decode_layer_calls", 1)
        self._add("kv_group_calls", g)
        sample = self.audit and self.step in SAMPLE_STEPS and layer in SAMPLE_LAYERS
        if policy.kind == "dense":
            logits = grouped_logits(query, key, scale)
            result = grouped_output(logits, value)
            kv = dense_bytes(t)
            for k, v in (("read", kv*g), ("k_bytes", kv//2*g), ("v_bytes", kv//2*g),
                         ("qk_mac", g*heads*t*dim), ("pv_mac", g*heads*t*dim)):
                self._add(k, v)
            self._record_gold_aggregate(None, g, t, q.device)
            if sample:
                self.timing.append(dict(step=self.step, layer=layer, group=0, **dense_timing(h)))
            return self.output(query, result)
        if layer not in self.online:
            self.online[layer] = StaticOnline(self.prefix[layer], (policy.method,))
        record = self.online[layer].append(key[0, :, -1], t)[policy.method]
        ranks = ranked_pages(weights(q, record, policy.method, scale), h)
        if policy.kind == "dynamic":
            limit = policy.max_pages or None
            out = run_shared(RefillPolicy("sparta", .02, policy.budget*100), q, key, ranks, t,
                             scale, max_pages=limit)
        else:
            batch = prepare_single(q, key, ranks, t, scale, policy.method, policy.budget*100,
                                   policy.page_limit(t))
            out = decide_v(batch, policy.method, policy.gate, policy.threshold)
        k = out.final
        led = physical_ledger(k, h, policy.method, torch.ones(g, dtype=torch.bool, device=q.device))
        over = led["total"] > out.ceiling
        torch._assert_async((~over).all(), "closed-loop read exceeds cap")
        self._add("over_budget_group_steps", over)
        self._add("added_k_pages", out.additional)
        self._add("qk_mac", led["k_tokens"]*heads*dim)
        pvtokens = k.mask.sum((1, 2, 3))
        self._add("pv_mac", pvtokens*dim)
        self._add("scorer_mac", g*heads*h.pages*dim*(2 if policy.method.startswith("mr") else 1))
        self._add("summary_build_write_bytes", g*METHOD_BYTES[policy.method])
        self._add("summary_append_channel_adds", g*dim)
        self._add("summary_append_extrema_comparisons", 2*g*dim)
        self._add("summary_prefill_build_write_bytes", g*((t-1)//32)*METHOD_BYTES[policy.method] if self.step == 0 else 0)
        for name, val in led.items():
            self._add("read" if name == "total" else name, val)
        live = k.valid.any(-1)
        k_pages = torch.zeros(g, h.pages, device=q.device, dtype=torch.bool).scatter_(1, k.ids, live)
        gold_pages = sorted(self._gold_page_tokens)
        if gold_pages:
            self._add("evidence_k_page_hits", k_pages[:, gold_pages])
            self._add("evidence_k_page_total", g*len(gold_pages))
        v_pages = torch.zeros_like(k_pages).scatter_(1, k.ids, k.mask.any((1, 3)))
        self._record_gold_aggregate(v_pages, g, t, q.device)
        peak = led["k_pages"].max()
        self.peak_pages = peak if self.peak_pages is None else torch.maximum(self.peak_pages, peak)
        if isinstance(out, SingleOutcome):
            self._add("fallback_group_steps", out.fallback)
            self._add("v_check_proposed_excess_bytes", out.initial_over_bytes)
            self._add("prefix_pages", out.prefix_pages)
            for name, loss in (("lost_all", out.lost_all), ("lost_prefix", out.lost_prefix)):
                self._add(f"{name}_sum", loss)
                high = loss.max()
                self.peak_loss[name] = high if name not in self.peak_loss else torch.maximum(self.peak_loss[name], high)
        if self.audit:
            prefix = out.prefix_pages if isinstance(out, SingleOutcome) else led["k_pages"]
            excess = out.initial_over_bytes if isinstance(out, SingleOutcome) else torch.zeros_like(prefix)
            self.group_stats.append((self.step, layer, out.ceiling, torch.stack(
                (led["total"], led["k_pages"], led["k_tokens"], led["v_tokens"],
                 out.fallback.long(), prefix, excess, pvtokens), dim=1)))
        if sample:
            self.timing.extend(dict(step=self.step, layer=layer, **r) for r in timing_rows(h, policy, ranks, out, led))
        y = masked_output(k.logits, gather_tokens(value, k.positions), k.mask)
        return self.output(query, y)

    def stats(self):
        r = super().stats()
        r.setdefault("read", 0)
        r["peak_k_pages"] = int(self.peak_pages) if self.peak_pages is not None else 0
        r.update({f"{k}_max":float(v) for k,v in self.peak_loss.items()})
        return r
