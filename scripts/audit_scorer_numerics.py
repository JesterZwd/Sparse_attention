"""Audit scalar scores and group A1 ranking, and emit local RTL vectors."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import numpy as np
import torch

from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.scorer_int import (
    COEFFICIENT_BITS,
    COEFFICIENT_MAX,
    Q24,
    factored_scores,
    fixed_scores,
    float_scores,
    integer_scores,
    prepare_inputs,
    q8_scores,
)
from gqa_probe.vskip_summary import encode


METHODS = ("q4", "mr25")


def _load_case(path):
    with np.load(path, allow_pickle=False) as archive:
        query = torch.from_numpy(archive["query"])
        records = tuple(torch.from_numpy(archive[f"r{i}"]) for i in range(4))
        method = archive["method"].item()
        if isinstance(method, bytes):
            method = method.decode("ascii")
        scale = float(archive["scale"].item())
        protected = tuple(int(i) for i in archive["protected"].reshape(-1)) if "protected" in archive else ()
        k = archive["k"].reshape(-1).tolist() if "k" in archive else [min(32, records[0].shape[1])]
    return str(path.name), query, records, method, scale, protected, k


def _synthetic_cases():
    rng = torch.Generator().manual_seed(20261123)
    query = torch.randn(2, 4, 128, generator=rng)
    pages = torch.randn(2, 97, 32, 128, generator=rng)
    lengths = torch.full((2, 97, 1), 32, dtype=torch.uint8)
    lengths[:, -1] = 7
    valid = torch.arange(32)[None, None, :] < lengths
    packet = (
        pages.masked_fill(~valid[..., None], 0).sum(2),
        pages.masked_fill(~valid[..., None], float("inf")).amin(2),
        pages.masked_fill(~valid[..., None], -float("inf")).amax(2),
        lengths,
    )
    for method in METHODS:
        yield f"synthetic_{method}", query, encode(*packet, method), method, 128**-.5, (0, 96), [12]


def _dist(values):
    vector = values.detach().abs().double().flatten().numpy()
    return {
        "mean": float(vector.mean()),
        "p95": float(np.percentile(vector, 95)),
        "p99": float(np.percentile(vector, 99)),
        "max": float(vector.max()),
    }


def _compare_stage(reference, candidate, valid, protected, sizes):
    mask = valid.expand_as(reference)
    a = reference.softmax(-1).sum(1)
    b = candidate.softmax(-1).sum(1)
    history = SimpleNamespace(protected=protected)
    ar, br = ranked_pages(a, history), ranked_pages(b, history)
    matches, optional, exact, ordered, gaps = [], [], [], [], []
    for group, k in enumerate(sizes):
        left, right = ar[group, :k].tolist(), br[group, :k].tolist()
        selected, proposed = set(left), set(right)
        eligible_left, eligible_right = selected.difference(protected), proposed.difference(protected)
        matches.append(len(selected & proposed) / k)
        optional.append(len(eligible_left & eligible_right) / len(eligible_left) if eligible_left else 1.)
        exact.append(selected == proposed)
        ordered.append(left == right)
        if len(protected) < k < int(valid[group, 0].sum()):
            gaps.append(float(a[group, ar[group, k - 1]] - a[group, ar[group, k]]))
    return {
        "score_absolute": _dist(reference[mask] - candidate[mask]),
        "a1_weight_absolute": _dist(a - b),
        "mean_topk_overlap": fmean(matches),
        "minimum_topk_overlap": min(matches),
        "mean_optional_topk_overlap": fmean(optional),
        "minimum_optional_topk_overlap": min(optional),
        "exact_topk_groups": sum(exact),
        "ordered_topk_groups": sum(ordered),
        "minimum_reference_boundary_gap": min(gaps) if gaps else None,
    }


def _case_audit(case):
    name, query, records, method, scale, protected, sizes = case
    reference = float_scores(query, records, method, scale)
    prepared = prepare_inputs(query, records, method, scale)
    groups = query.shape[0]
    if len(sizes) == 1:
        sizes = sizes * groups
    if len(sizes) != groups or any(k < max(1, len(protected)) for k in sizes):
        raise ValueError(f"invalid top-k for capture {name}")
    if any(k > int(prepared.valid[g, 0].sum()) for g, k in enumerate(sizes)):
        raise ValueError(f"capture {name} requests more pages than are valid")
    if any(i < 0 or i >= reference.shape[-1] or not prepared.valid[:, 0, i].all() for i in protected):
        raise ValueError(f"capture {name} has an invalid protected page")
    factor = factored_scores(query, records, method, scale)
    q8 = q8_scores(prepared)
    fixed = fixed_scores(prepared)
    variants = {
        "factored_fp32": (reference, factor),
        "q8_unrounded_coefficients": (factor, q8),
        "q24_coefficient_ablation": (q8, fixed),
        "q24": (reference, fixed),
    }
    stages = {
        name: _compare_stage(a, b, prepared.valid, protected, sizes)
        for name, (a, b) in variants.items()
    }
    row = {
        "case": name,
        "method": method,
        "groups": groups,
        "heads": query.shape[1],
        "pages": reference.shape[-1],
        "selected_pages": sizes,
        "protected_pages": list(protected),
        "valid_scores": int(prepared.valid.expand_as(reference).sum()),
        "saturated_coefficients": [int(prepared.saturated_a.sum()), int(prepared.saturated_b.sum())],
        "zero_rounded_coefficients": [int(prepared.underflow_a.sum()), int(prepared.underflow_b.sum())],
        "maximum_rounded_coefficient_unclamped": [
            round(float(prepared.float_coeff_a.max()) * Q24),
            round(float(prepared.float_coeff_b.max()) * Q24),
        ],
        "stages": stages,
    }
    return row, prepared


def _fixture(prepared, group, head, limit):
    if prepared.query.shape[-1] != 128:
        raise ValueError("RTL fixtures require 128 query channels")
    valid_ids = prepared.valid[group, 0].nonzero(as_tuple=True)[0].tolist()
    if len(valid_ids) > limit:
        locations = np.linspace(0, len(valid_ids) - 1, limit).round().astype(int)
        valid_ids = [valid_ids[i] for i in locations]
    raw = integer_scores(prepared)
    pages = []
    for pid in valid_ids:
        a = prepared.code_a[group, pid].to(torch.int16)
        b = prepared.code_b[group, pid].to(torch.int16)
        code = ((a & 15) | ((b & 15) << 4)) if prepared.method == "q4" else ((a & 63) | (b << 6))
        pages.append({
            "page": pid,
            "codes": code.tolist(),
            "coeff_a": int(prepared.coeff_a[group, head, pid]),
            "coeff_b": int(prepared.coeff_b[group, head, pid]),
            "bias": int(prepared.bias[group, 0, pid]),
            "score": int(raw[group, head, pid]),
        })
    return {"query": prepared.query[group, head].tolist(), "pages": pages}


def _aggregate(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    out = {}
    for method, entries in by_method.items():
        groups = sum(r["groups"] for r in entries)
        stats = {}
        for stage in entries[0]["stages"]:
            pairs = [(r["stages"][stage], r["groups"]) for r in entries]
            stats[stage] = {
                "maximum_score_error": max(s["score_absolute"]["max"] for s, _ in pairs),
                "mean_capture_score_error_p99": fmean(s["score_absolute"]["p99"] for s, _ in pairs),
                "maximum_capture_score_error_p99": max(s["score_absolute"]["p99"] for s, _ in pairs),
                "mean_optional_topk_overlap": sum(s["mean_optional_topk_overlap"] * n for s, n in pairs) / groups,
                "minimum_optional_topk_overlap": min(s["minimum_optional_topk_overlap"] for s, _ in pairs),
                "exact_topk_groups": sum(s["exact_topk_groups"] for s, _ in pairs),
                "ordered_topk_groups": sum(s["ordered_topk_groups"] for s, _ in pairs),
            }
        out[method] = {
            "captures": len(entries),
            "groups": groups,
            "valid_scores": sum(r["valid_scores"] for r in entries),
            "saturated_coefficients": [sum(r["saturated_coefficients"][i] for r in entries) for i in range(2)],
            "zero_rounded_coefficients": [sum(r["zero_rounded_coefficients"][i] for r in entries) for i in range(2)],
            "maximum_rounded_coefficient_unclamped": [
                max(r["maximum_rounded_coefficient_unclamped"][i] for r in entries) for i in range(2)
            ],
            "stages": stats,
        }
    return out


@torch.no_grad()
def audit(inputs=None, *, rtl_blocks=8, rtl_pages=16):
    if inputs is None:
        stream = _synthetic_cases()
    else:
        paths = sorted(Path(inputs).rglob("*.npz"))
        if not paths:
            raise ValueError(f"no scorer captures found in {inputs}")
        stream = (_load_case(path) for path in paths)
    rows, fixture_inputs = [], defaultdict(list)
    for case in stream:
        row, prepared = _case_audit(case)
        rows.append(row)
        if rtl_pages > 0 and rtl_blocks > 0:
            fixture_inputs[row["method"]].append(prepared)
    fixtures = {}
    for method, candidates in fixture_inputs.items():
        count = min(rtl_blocks, len(candidates))
        selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count).round().astype(int)]
        blocks = []
        # Cover different captures before sampling more groups or heads of any one.
        for slot in range(rtl_blocks):
            for prepared in selected:
                groups, heads = prepared.query.shape[:2]
                if slot < groups * heads and len(blocks) < rtl_blocks:
                    g, h = slot % groups, slot // groups
                    blocks.append(_fixture(prepared, g, h, rtl_pages))
        fixtures[method] = blocks
    return rows, _aggregate(rows), fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, help="Use synthetic cases if no capture directory is given.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtl-fixtures", type=Path)
    parser.add_argument("--rtl-blocks", type=int, default=8)
    parser.add_argument("--rtl-pages", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows, summary, fixtures = audit(args.inputs, rtl_blocks=args.rtl_blocks if args.rtl_fixtures else 0,
                                    rtl_pages=args.rtl_pages)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "input_kind": "captured" if args.inputs else "synthetic",
        "coefficient_bits": COEFFICIENT_BITS,
        "coefficient_max_integer": COEFFICIENT_MAX,
        "coefficient_denominator": Q24,
        "summary": summary,
        "captures": rows,
    }
    (args.output / "scorer_numerics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.rtl_fixtures:
        args.rtl_fixtures.mkdir(parents=True, exist_ok=True)
        for method, blocks in fixtures.items():
            name = "quest_vectors.json" if method == "q4" else "mr_vectors.json"
            (args.rtl_fixtures / name).write_text(json.dumps(blocks) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
