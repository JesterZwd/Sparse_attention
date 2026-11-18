"""Full-pool Oracle selectors. Scores are normalized independently per head."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    order: list[int]
    coverage: np.ndarray
    batches: list[dict]
    metrics: dict


def protected_pages(length: int, page_size: int = 32, sink_pages: int = 1,
                    local_tokens: int = 128) -> list[int]:
    if length <= 0 or page_size <= 0 or sink_pages < 0 or local_tokens < 0:
        raise ValueError("invalid length/page/protection configuration")
    pages = (length + page_size - 1) // page_size
    sink = set(range(min(sink_pages, pages)))
    local = set(range(max(0, length - local_tokens) // page_size, pages)) if local_tokens else set()
    return sorted(sink | local)


def _rank(ids, scores):
    return ids[np.lexsort((ids, -scores))]


def select_pages(mass, protected, budget, target, algorithm, batch_size=1,
                 mode="fixed", eps=1e-12) -> SelectionResult:
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 2 or 0 in mass.shape or not np.all(np.isfinite(mass)):
        raise ValueError("mass must be a finite nonempty [pages, heads] matrix")
    if np.any(mass < 0) or not np.allclose(mass.sum(axis=0), 1., atol=1e-7, rtol=1e-7):
        raise ValueError("mass must be nonnegative and normalized over all pages per head")
    pages, heads = mass.shape
    protected = list(protected)
    if any(not isinstance(g, (int, np.integer)) or not 0 <= g < pages for g in protected):
        raise ValueError("invalid protected page")
    order = sorted(set(int(g) for g in protected))
    if not isinstance(budget, (int, np.integer)) or not len(order) <= budget <= pages:
        raise ValueError("budget must satisfy protected_count <= budget <= pages")
    if algorithm not in {"A1", "A3", "A3-B"} or mode not in {"fixed", "adaptive"}:
        raise ValueError("unsupported algorithm or budget mode")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    p = np.broadcast_to(np.asarray(target, dtype=np.float64), (heads,))
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not np.isfinite(eps) or eps < 0:
        raise ValueError("targets must be finite in [0,1], eps finite and nonnegative")
    selected = np.zeros(pages, dtype=bool)
    selected[order] = True
    coverage = mass[order].sum(axis=0) if order else np.zeros(heads)
    shared = mass.sum(axis=1)
    group_order = _rank(np.flatnonzero(~selected), shared[~selected])
    batches = []
    metrics = {"selector_iterations": 0, "selector_gain_evaluations": 0,
               "coverage_selection_pages": 0, "a1_fill_pages": 0,
               "static_selection_pages": 0, "clip_contributions": 0,
               "evaluated_contributions": 0, "changed_decisions": 0,
               "batch_changed_pages": 0, "batch_redundancy": 0.,
               "predicted_gain_sum": 0., "batch_overshoot_pages": 0}

    def add(ids):
        nonlocal coverage
        ids = [int(g) for g in ids]
        selected[ids] = True
        order.extend(ids)
        coverage = coverage + mass[ids].sum(axis=0)

    if algorithm == "A1":
        for g in group_order:
            if len(order) >= budget or (mode == "adaptive" and np.all(p - coverage <= eps)):
                break
            add([g])
            metrics["static_selection_pages"] += 1
    else:
        while len(order) < budget:
            deficit = np.maximum(p - coverage, 0.)
            if np.all(deficit <= eps):
                if mode == "fixed":
                    fill = [g for g in group_order if not selected[g]][:budget - len(order)]
                    add(fill)
                    metrics["a1_fill_pages"] += len(fill)
                break
            ids = np.flatnonzero(~selected)
            gains = np.minimum(mass[ids], deficit).sum(axis=1)
            ranked = _rank(ids, gains)
            size = min(batch_size if algorithm == "A3-B" else 1, budget - len(order))
            chosen = ranked[:size]
            baseline = _rank(ids, shared[ids])[:size]
            predicted = float(gains[np.isin(ids, chosen)].sum())
            before = float(np.minimum(p, coverage).sum())
            first_satisfied = None
            running = coverage.copy()
            for position, g in enumerate(chosen, 1):
                running += mass[g]
                if first_satisfied is None and np.all(p - running <= eps):
                    first_satisfied = position
            add(chosen)
            actual = float(np.minimum(p, coverage).sum()) - before
            redundancy = predicted - actual
            overshoot = size - first_satisfied if first_satisfied is not None else 0
            changed = size - len(set(chosen) & set(baseline))
            batch = {"pages": chosen.tolist(), "actual_batch_size": size,
                     "predicted_gain": predicted, "actual_gain": actual,
                     "redundancy": redundancy,
                     "normalized_redundancy": redundancy / (predicted + 1e-30),
                     "batch_selection_change_rate": changed / size,
                     "first_satisfied_position": first_satisfied,
                     "batch_end_pages": len(order), "batch_overshoot_pages": overshoot}
            batches.append(batch)
            metrics["selector_iterations"] += 1
            metrics["selector_gain_evaluations"] += len(ids)
            metrics["coverage_selection_pages"] += size
            metrics["clip_contributions"] += int((mass[ids] > deficit).sum())
            metrics["evaluated_contributions"] += len(ids) * heads
            metrics["changed_decisions"] += int(ranked[0] != baseline[0])
            metrics["batch_changed_pages"] += changed
            metrics["batch_redundancy"] += redundancy
            metrics["predicted_gain_sum"] += predicted
            metrics["batch_overshoot_pages"] += overshoot

    def ratio(a, b):
        return a / b if b else None

    metrics.update({
        "clip_rate": ratio(metrics["clip_contributions"], metrics["evaluated_contributions"]),
        "decision_change_rate": ratio(metrics["changed_decisions"], metrics["selector_iterations"]),
        "batch_selection_change_rate": ratio(metrics["batch_changed_pages"], metrics["coverage_selection_pages"]),
        "batch_selection_change_rate_mean": float(np.mean([b["batch_selection_change_rate"] for b in batches])) if batches else None,
        "normalized_batch_redundancy": ratio(metrics["batch_redundancy"], metrics["predicted_gain_sum"]),
    })
    return SelectionResult(order, coverage, batches, metrics)
