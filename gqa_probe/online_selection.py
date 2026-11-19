"""CPU Oracle selectors with online work separated from optional swap audits.

Counters describe executed NumPy reference arithmetic, reductions, ranking
inputs, and container visits/writes. They are not cycle or ASIC estimates.
``count_comparisons=True`` instruments actual Python TimSort comparator calls;
the uninstrumented runtime uses TimSort keys with the identical score/ID order.
No clip/change/first-satisfied or other offline greedy audits run here.
"""

from collections import deque
from dataclasses import dataclass
from functools import cmp_to_key
from numbers import Integral

import numpy as np


EPS = 1e-12


@dataclass
class OnlineSelectionResult:
    order: list[int]
    coverage: np.ndarray
    metrics: dict
    audit: list[dict]


_COUNTERS = (
    "shared_score_head_additions", "shared_ranking_calls", "shared_ranking_items",
    "gain_ranking_calls", "gain_ranking_items", "upper_ranking_calls", "upper_ranking_items",
    "ranking_calls", "ranking_items", "ranking_key_negations",
    "initial_coverage_additions", "greedy_coverage_additions", "reset_coverage_additions",
    "coverage_update_arithmetic_ops", "coverage_selected_flag_writes", "selection_order_writes",
    "protected_id_visits", "reset_selected_flag_writes", "a1_fill_membership_visits",
    "a1_fill_pages", "static_selection_pages", "greedy_selection_pages", "selector_iterations",
    "gain_evaluations", "gain_head_evaluations", "gain_min_comparisons", "gain_sum_additions",
    "greedy_deficit_subtractions", "greedy_deficit_clamp_comparisons", "greedy_stop_head_comparisons",
    "greedy_unselected_mask_visits", "upper_accumulation_additions", "upper_target_multiplications",
    "upper_target_min_comparisons", "gate_head_comparisons", "gate_target_subtractions",
    "gate_boolean_reductions", "guard_floor_subtractions", "guard_floor_max_comparisons",
    "swaps", "swap_rounds", "candidate_pairs", "pair_head_evaluations",
    "candidate_build_out_membership_visits", "candidate_build_in_membership_visits",
    "candidate_build_out_buffer_writes", "candidate_build_out_buffer_evictions",
    "candidate_build_out_slice_writes",
    "candidate_build_in_buffer_writes", "candidate_build_rounds",
    "swap_initial_deficit_subtractions", "swap_initial_deficit_clamp_comparisons",
    "swap_initial_deficit_sum_additions", "pair_trial_arithmetic_ops",
    "pair_deficit_subtractions", "pair_deficit_clamp_comparisons", "pair_deficit_sum_additions",
    "pair_gain_subtractions", "pair_gain_comparisons", "pair_guard_head_comparisons",
    "pair_guard_boolean_reductions", "pair_choice_comparisons", "admissible_pairs",
    "swap_stop_comparisons", "order_removal_visits", "order_removal_moves",
)


def select_online(mass, protected, budget, target, algorithm="A1", delta=.01,
                  max_swaps=1, batch_size=8, alpha=1., count_comparisons=False,
                  record_audit=False, validate=True) -> OnlineSelectionResult:
    """Select exactly ``budget`` unique physical pages from ``mass[J,H]``.

    A1 ranks by shared W=sum_h mass. A3 and A3-B greedily rank the original
    capped-coverage gains (one page and a fixed batch respectively), filling
    any unused fixed budget in W order after the target is met. GATE-A3 first
    constructs A1, and only on a failed gate resets to protected pages and runs
    A3 using the same W ranking.

    SWAP starts with A1. Each round considers its last four selected,
    unprotected pages and first eight unselected pages in the original W
    ranking. It accepts the largest strictly positive original-deficit gain,
    breaking ties by incoming ID then outgoing ID. Initially satisfied heads
    keep floor max(target, initial_A1_coverage-delta) across every round.

    A3-U computes true per-head Top-free upper bounds, including their ranking
    and accumulation work, and uses min(target, alpha*U). At alpha=1 the
    feasible-set objective identity permits the original target directly;
    this preserves exact A3 ordering without numerical bound-roundoff drift.

    ``validate=False`` skips finite/nonnegative/normalization scans of trusted
    mass inputs. Structural, index, and parameter checks remain active.
    ``record_audit`` retains only accepted-swap facts already used online.
    """
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 2 or 0 in mass.shape:
        raise ValueError("mass must be a nonempty [pages, heads] matrix")
    if validate and (not np.all(np.isfinite(mass)) or np.any(mass < 0) or not np.allclose(
            mass.sum(axis=0), 1., atol=1e-7, rtol=1e-7)):
        raise ValueError("mass must be finite, nonnegative, and normalized per head")
    pages, heads = mass.shape
    protected_input = list(protected)
    if any(isinstance(page, bool) or not isinstance(page, Integral) or not 0 <= page < pages
           for page in protected_input):
        raise ValueError("invalid protected page index")
    protected = sorted(set(int(page) for page in protected_input))
    if isinstance(budget, bool) or not isinstance(budget, Integral) or not len(protected) <= budget <= pages:
        raise ValueError("budget must satisfy protected_count <= budget <= pages")
    budget = int(budget)
    if algorithm not in {"A1", "A3", "A3-B", "GATE-A3", "SWAP", "A3-U"}:
        raise ValueError("unsupported online selection algorithm")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(max_swaps, bool) or not isinstance(max_swaps, Integral) or max_swaps < 1:
        raise ValueError("max_swaps must be a positive integer")
    if not np.isscalar(delta) or not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite in (0,1]")
    try:
        original_target = np.broadcast_to(np.asarray(target, dtype=np.float64), (heads,))
    except (TypeError, ValueError) as exc:
        raise ValueError("target must broadcast to the query heads") from exc
    if not np.all(np.isfinite(original_target)) or np.any((original_target < 0) | (original_target > 1)):
        raise ValueError("target must be finite in [0,1]")

    metrics = {name: 0 for name in _COUNTERS}
    metrics.update(gate_hit=False, upper_per_head=None, effective_target_per_head=None,
                   effective_targets=None,
                   ranking_implementation="python_timsort_cmp_to_key" if count_comparisons else "python_timsort_key",
                   input_mass_elements=pages * heads, state_shared_score_elements=pages,
                   state_selected_flags=pages, state_coverage_elements=heads,
                   state_ranked_page_ids=pages - len(protected), state_order_page_ids=budget,
                   state_anchor_elements=0, state_guard_mask_elements=0,
                   state_guard_floor_elements=0, peak_candidate_page_ids=0,
                   peak_pair_trial_head_elements=0, peak_gain_matrix_elements=0)
    for name in ("ranking_comparisons", "ranking_score_comparisons", "ranking_id_comparisons",
                 "shared_ranking_comparisons", "gain_ranking_comparisons", "upper_ranking_comparisons"):
        metrics[name] = 0 if count_comparisons else None
    metrics["protected_id_visits"] = len(protected_input)
    audit = []

    def rank(ids, scores, role):
        items = [(int(page), float(score)) for page, score in zip(ids, scores)]
        metrics["ranking_calls"] += 1
        metrics["ranking_items"] += len(items)
        metrics[f"{role}_ranking_calls"] += 1
        metrics[f"{role}_ranking_items"] += len(items)
        if count_comparisons:
            def compare(left, right):
                metrics["ranking_comparisons"] += 1
                metrics[f"{role}_ranking_comparisons"] += 1
                metrics["ranking_score_comparisons"] += 1
                if left[1] > right[1]:
                    return -1
                metrics["ranking_score_comparisons"] += 1
                if left[1] < right[1]:
                    return 1
                metrics["ranking_id_comparisons"] += 1
                if left[0] < right[0]:
                    return -1
                metrics["ranking_id_comparisons"] += 1
                return 1 if left[0] > right[0] else 0
            ranked = sorted(items, key=cmp_to_key(compare))
        else:
            metrics["ranking_key_negations"] += len(items)
            ranked = sorted(items, key=lambda pair: (-pair[1], pair[0]))
        return [page for page, _ in ranked]

    selected = np.zeros(pages, dtype=bool)
    order = []
    coverage = np.zeros(heads, dtype=np.float64)

    def add(ids, arithmetic_counter):
        nonlocal coverage
        if not len(ids):
            return
        ids = list(ids)
        # Reduction of n rows plus the accumulator add is exactly n*H
        # logical additions, including the initial zero accumulation.
        coverage = coverage + mass[ids].sum(axis=0)
        metrics[arithmetic_counter] += len(ids) * heads
        selected[ids] = True
        metrics["coverage_selected_flag_writes"] += len(ids)
        order.extend(ids)
        metrics["selection_order_writes"] += len(ids)

    add(protected, "initial_coverage_additions")
    shared = mass.sum(axis=1)
    metrics["shared_score_head_additions"] = pages * max(heads - 1, 0)
    available = np.flatnonzero(~selected)
    w_order = rank(available, shared[available], "shared")

    def fill_from_w(counter):
        wanted = budget - len(order)
        fill = []
        if wanted:
            for page in w_order:
                metrics["a1_fill_membership_visits"] += 1
                if not selected[page]:
                    fill.append(page)
                    if len(fill) == wanted:
                        break
        add(fill, counter)
        return len(fill)

    def greedy(targets, size):
        while len(order) < budget:
            deficit = np.maximum(targets - coverage, 0.)
            metrics["greedy_deficit_subtractions"] += heads
            metrics["greedy_deficit_clamp_comparisons"] += heads
            metrics["greedy_stop_head_comparisons"] += heads
            if np.all(deficit <= EPS):
                metrics["a1_fill_pages"] += fill_from_w("greedy_coverage_additions")
                break
            ids = np.flatnonzero(~selected)
            metrics["greedy_unselected_mask_visits"] += pages
            gains = np.minimum(mass[ids], deficit).sum(axis=1)
            metrics["gain_evaluations"] += len(ids)
            metrics["gain_head_evaluations"] += len(ids) * heads
            metrics["gain_min_comparisons"] += len(ids) * heads
            metrics["gain_sum_additions"] += len(ids) * max(heads - 1, 0)
            metrics["peak_gain_matrix_elements"] = max(metrics["peak_gain_matrix_elements"], len(ids) * heads)
            chosen = rank(ids, gains, "gain")[:min(size, budget - len(order))]
            add(chosen, "greedy_coverage_additions")
            metrics["selector_iterations"] += 1
            metrics["greedy_selection_pages"] += len(chosen)

    if algorithm in {"A1", "GATE-A3", "SWAP"}:
        metrics["static_selection_pages"] = fill_from_w("initial_coverage_additions")
        if algorithm == "A1":
            return OnlineSelectionResult(order, coverage, metrics, audit)
        anchor = coverage.copy()
        good = anchor >= original_target - EPS
        metrics["state_anchor_elements"] = heads
        metrics["state_guard_mask_elements"] = heads
        metrics["gate_target_subtractions"] += heads
        metrics["gate_head_comparisons"] += heads
        metrics["gate_boolean_reductions"] += max(heads - 1, 0)
        metrics["gate_hit"] = bool(np.all(good))
        if metrics["gate_hit"]:
            return OnlineSelectionResult(order, coverage, metrics, audit)
        if algorithm == "GATE-A3":
            selected[:] = False
            metrics["reset_selected_flag_writes"] += pages
            order = []
            coverage = np.zeros(heads, dtype=np.float64)
            add(protected, "reset_coverage_additions")
            greedy(original_target, 1)
            return OnlineSelectionResult(order, coverage, metrics, audit)

        # Fixed initial-A1 guard; never re-anchor after the first swap.
        floors = np.maximum(original_target, anchor - delta) - EPS
        metrics["guard_floor_subtractions"] += 2 * heads
        metrics["guard_floor_max_comparisons"] += heads
        metrics["state_guard_floor_elements"] = heads
        good_count = int(good.sum())
        current_deficit = None
        for round_index in range(max_swaps):
            out = deque(maxlen=4)
            for page in w_order:
                metrics["candidate_build_out_membership_visits"] += 1
                if selected[page]:
                    if len(out) == 4:
                        metrics["candidate_build_out_buffer_evictions"] += 1
                    out.append(page)
                    metrics["candidate_build_out_buffer_writes"] += 1
            incoming = []
            for page in w_order:
                metrics["candidate_build_in_membership_visits"] += 1
                if not selected[page]:
                    incoming.append(page)
                    metrics["candidate_build_in_buffer_writes"] += 1
                    if len(incoming) == 8:
                        break
            metrics["candidate_build_rounds"] += 1
            metrics["peak_candidate_page_ids"] = max(metrics["peak_candidate_page_ids"], len(out) + len(incoming))
            if not out or not incoming:
                break
            if current_deficit is None:
                current_deficit = float(np.maximum(original_target - coverage, 0.).sum())
                metrics["swap_initial_deficit_subtractions"] += heads
                metrics["swap_initial_deficit_clamp_comparisons"] += heads
                metrics["swap_initial_deficit_sum_additions"] += max(heads - 1, 0)
            metrics["swap_rounds"] += 1
            best = None
            for page_in in incoming:
                for page_out in out:
                    trial = coverage - mass[page_out] + mass[page_in]
                    deficit = float(np.maximum(original_target - trial, 0.).sum())
                    gain = current_deficit - deficit
                    guard = bool(np.all(trial[good] >= floors[good]))
                    metrics["candidate_pairs"] += 1
                    metrics["pair_head_evaluations"] += heads
                    metrics["pair_trial_arithmetic_ops"] += 2 * heads
                    metrics["pair_deficit_subtractions"] += heads
                    metrics["pair_deficit_clamp_comparisons"] += heads
                    metrics["pair_deficit_sum_additions"] += max(heads - 1, 0)
                    metrics["pair_gain_subtractions"] += 1
                    metrics["pair_gain_comparisons"] += 1
                    metrics["pair_guard_head_comparisons"] += good_count
                    metrics["pair_guard_boolean_reductions"] += max(good_count - 1, 0)
                    metrics["peak_pair_trial_head_elements"] = heads
                    if gain <= EPS or not guard:
                        continue
                    metrics["admissible_pairs"] += 1
                    take = best is None
                    if best is not None:
                        metrics["pair_choice_comparisons"] += 1
                        take = gain > best[0]
                        if not take:
                            metrics["pair_choice_comparisons"] += 1
                            if gain == best[0]:
                                metrics["pair_choice_comparisons"] += 1
                                take = page_in < best[1]
                                if not take:
                                    metrics["pair_choice_comparisons"] += 1
                                    if page_in == best[1]:
                                        metrics["pair_choice_comparisons"] += 1
                                        take = page_out < best[2]
                    if take:
                        best = (gain, page_in, page_out, deficit)
            if best is None:
                break
            gain, page_in, page_out, new_deficit = best
            coverage = coverage - mass[page_out] + mass[page_in]
            metrics["coverage_update_arithmetic_ops"] += 2 * heads
            selected[page_out], selected[page_in] = False, True
            metrics["coverage_selected_flag_writes"] += 2
            position = order.index(page_out)
            metrics["order_removal_visits"] += position + 1
            metrics["order_removal_moves"] += len(order) - position - 1
            order.pop(position)
            order.append(page_in)
            metrics["selection_order_writes"] += 1
            metrics["swaps"] += 1
            if record_audit:
                audit.append(dict(round=round_index + 1, incoming=page_in, outgoing=page_out,
                                  gain=gain, deficit_before=current_deficit, deficit_after=new_deficit))
            current_deficit = new_deficit
            metrics["swap_stop_comparisons"] += 1
            if current_deficit <= EPS:
                break
        return OnlineSelectionResult(order, coverage, metrics, audit)

    effective_target = original_target
    if algorithm == "A3-U":
        upper = coverage.copy()
        free = budget - len(protected)
        for head in range(heads):
            ranked = rank(available, mass[available, head], "upper")
            if free:
                upper[head] += mass[ranked[:free], head].sum()
                metrics["upper_accumulation_additions"] += free
        effective = np.minimum(original_target, alpha * upper)
        metrics["upper_target_multiplications"] += heads
        metrics["upper_target_min_comparisons"] += heads
        metrics["upper_per_head"] = upper.tolist()
        metrics["effective_target_per_head"] = effective.tolist()
        metrics["effective_targets"] = effective.tolist()
        if alpha != 1:
            effective_target = effective
    greedy(effective_target, batch_size if algorithm == "A3-B" else 1)
    return OnlineSelectionResult(order, coverage, metrics, audit)
