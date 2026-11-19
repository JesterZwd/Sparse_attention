"""Batched device-resident A1 and bounded swap selection.

All counters are per KV group device tensors. ``candidate_pairs`` counts valid
logical candidates; ``executed_candidate_pairs`` also counts rectangular padding.
Initial-gate groups are compacted before any candidate arithmetic. Ranking is
fixed at initial A1 W order; membership and boundary candidates are refreshed.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import torch


EPS = 1e-12


@dataclass
class TorchSelectionResult:
    mask: torch.Tensor
    coverage: torch.Tensor
    gate_hit: torch.Tensor
    swaps: torch.Tensor
    candidate_pairs: torch.Tensor
    metrics: dict


def _require(condition, message):
    # CUDA async assertions validate inputs without a scalar host synchronization.
    torch._assert_async(condition, message)


def select_torch(mass, protected, budget, target, algorithm="A1", delta=.01,
                 max_swaps=1):
    """Select a unique physical budget from normalized ``mass[G,J,H]``.

    ``protected`` is a shared iterable of page IDs or a boolean [J]/[G,J]
    tensor. Targets broadcast to [G,H]. Float64 mass is recommended, including
    when the full-history logits used to form it were computed in float32.
    No full history or page masses move to CPU. As with ``nonzero`` generally,
    CUDA active-group compaction can synchronize to determine its output size.
    """
    if not isinstance(mass, torch.Tensor) or mass.ndim != 3 or 0 in mass.shape:
        raise ValueError("mass must be a nonempty [groups,pages,heads] tensor")
    if not mass.is_floating_point():
        raise ValueError("mass must have floating dtype")
    groups, pages, heads = mass.shape
    if not isinstance(budget, Integral) or not 0 <= budget <= pages:
        raise ValueError("budget must be between zero and page count")
    if algorithm not in {"A1", "SWAP"}:
        raise ValueError("algorithm must be A1 or SWAP")
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if not isinstance(max_swaps, Integral) or max_swaps < 0:
        raise ValueError("max_swaps must be a nonnegative integer")
    _require(torch.isfinite(mass).all() & (mass >= 0).all(), "mass must be finite and nonnegative")
    _require(torch.isclose(mass.sum(1), torch.ones_like(mass[:, 0]), atol=1e-7,
                           rtol=1e-7).all(), "mass must be normalized over pages")
    targets = torch.as_tensor(target, device=mass.device, dtype=mass.dtype)
    try:
        targets = torch.broadcast_to(targets, (groups, heads))
    except RuntimeError as exc:
        raise ValueError("target must broadcast to [groups,heads]") from exc
    _require((torch.isfinite(targets) & (targets >= 0) & (targets <= 1)).all(),
             "target must be finite in [0,1]")
    if isinstance(protected, torch.Tensor) and protected.dtype == torch.bool:
        try:
            protection = torch.broadcast_to(protected.to(mass.device), (groups, pages)).clone()
        except RuntimeError as exc:
            raise ValueError("boolean protection must broadcast to [groups,pages]") from exc
    else:
        ids = list(protected)
        if any(not isinstance(page, Integral) or not 0 <= page < pages for page in ids):
            raise ValueError("invalid protected page")
        protection = torch.zeros((groups, pages), device=mass.device, dtype=torch.bool)
        protection[:, sorted(set(ids))] = True
    protected_counts = protection.sum(1)
    _require((protected_counts <= budget).all(), "protected count exceeds budget")
    w = mass.sum(-1).masked_fill(protection, -torch.inf)
    ranking = torch.argsort(w, dim=1, descending=True, stable=True)
    position = torch.arange(pages, device=mass.device).expand(groups, -1)
    free_fill = position < (budget - protected_counts)[:, None]
    selected = protection.clone()
    selected.scatter_(1, ranking, free_fill | protection.gather(1, ranking))
    coverage = (mass * selected[:, :, None]).sum(1)
    anchor = coverage.clone()
    initially_good = anchor >= targets - EPS
    floor = torch.maximum(targets, anchor - delta)
    gate = initially_good.all(-1)
    swaps = torch.zeros(groups, device=mass.device, dtype=torch.int64)
    pairs = torch.zeros_like(swaps)
    executed = torch.zeros_like(swaps)
    rounds = torch.zeros_like(swaps)
    active = ~gate & (protected_counts < budget)
    out_width, in_width = min(4, budget), min(8, pages - budget)
    if algorithm == "SWAP" and out_width and in_width:
        for _ in range(max_swaps):
            ids = active.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                break
            rank = ranking[ids]
            selected_rank = selected[ids].gather(1, rank)
            free_rank = ~protection[ids].gather(1, rank)
            positions = position[ids]
            # Last four current selected/nonprotected and first eight current
            # unselected, in the original W order (not a new gain ranking).
            out_pos = positions.masked_fill(~(selected_rank & free_rank), -1).topk(
                out_width, dim=1).values
            in_pos = positions.masked_fill(selected_rank, pages).topk(
                in_width, dim=1, largest=False).values
            out_valid, in_valid = out_pos >= 0, in_pos < pages
            out_ids = rank.gather(1, out_pos.clamp_min(0))
            in_ids = rank.gather(1, in_pos.clamp_max(pages - 1))
            # Flatten candidates in (incoming ID, outgoing ID) ascending order
            # so argmax implements the required secondary/tertiary tie rule.
            out_ids, order = out_ids.sort(dim=1, stable=True)
            out_valid = out_valid.gather(1, order)
            in_ids, order = in_ids.sort(dim=1, stable=True)
            in_valid = in_valid.gather(1, order)
            current = coverage[ids]
            incoming = mass[ids[:, None], in_ids]
            outgoing = mass[ids[:, None], out_ids]
            trial = current[:, None, None, :] - outgoing[:, None, :, :] + incoming[:, :, None, :]
            deficit = (targets[ids] - current).clamp_min(0).sum(-1)
            gain = deficit[:, None, None] - (targets[ids, None, None, :] - trial).clamp_min(0).sum(-1)
            valid_pairs = in_valid[:, :, None] & out_valid[:, None, :]
            guard = (~initially_good[ids, None, None, :] |
                     (trial >= floor[ids, None, None, :] - EPS)).all(-1)
            admissible = valid_pairs & guard & (gain > EPS)
            scores = gain.masked_fill(~admissible, -torch.inf).flatten(1)
            best_gain, best = scores.max(1)
            accepted = torch.isfinite(best_gain)
            chosen_in = in_ids.gather(1, (best // out_width)[:, None]).squeeze(1)
            chosen_out = out_ids.gather(1, (best % out_width)[:, None]).squeeze(1)
            accepted_ids = ids[accepted]
            selected[accepted_ids, chosen_out[accepted]] = False
            selected[accepted_ids, chosen_in[accepted]] = True
            coverage[accepted_ids] = (current - mass[ids, chosen_out] + mass[ids, chosen_in])[accepted]
            swaps[ids] += accepted.to(torch.int64)
            pairs[ids] += valid_pairs.sum((1, 2))
            executed[ids] += out_width * in_width
            rounds[ids] += 1
            active[ids] = accepted & ((targets[ids] - coverage[ids]).clamp_min(0).sum(-1) > EPS)
    metrics = {
        "executed_candidate_pairs": executed,
        "logical_candidate_pairs": pairs,
        "swap_rounds": rounds,
        "candidate_head_evaluations": executed * heads,
        "logical_candidate_head_evaluations": pairs * heads,
        "shared_score_head_additions": torch.full_like(swaps, pages * max(heads - 1, 0)),
        "ranked_page_items": torch.full_like(swaps, pages),
        "initial_coverage_mass_elements": torch.full_like(swaps, budget * heads),
        "executed_initial_coverage_mass_elements": torch.full_like(swaps, pages * heads),
        "accepted_coverage_update_elements": swaps * heads,
        "initial_coverage": anchor,
    }
    return TorchSelectionResult(selected, coverage, gate, swaps, pairs, metrics)
