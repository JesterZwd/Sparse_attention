"""CoMRAD Round 13 workpoints and prespecified paired acceptance rules."""
from .round11_workpoints import A, M, Workpoint

QUEST = Workpoint("Quest", "q4")
SPARSE = (QUEST, A, M)
GATES = dict(quality_pp=1.0, quality_ci_low_gt=0.0, quality_read_ratio_max=1.01,
             noninferiority_ci_low_min=-.01, saving_read_ratio_max=.95,
             requires_zero_overruns=True, saving_requires_arithmetic_decrease=True)


def compare_gate(reference, candidate, delta, ci_low):
    ratio = candidate["read_per_decode"] / reference["read_per_decode"]
    arithmetic = candidate["arithmetic_per_decode"] / reference["arithmetic_per_decode"]
    ok = not (reference["overrun_groups"] or candidate["overrun_groups"])
    quality = ok and delta >= .01 - 1e-12 and ci_low > 0 and ratio <= 1.01 + 1e-12
    saving = ok and ci_low >= -.01 - 1e-12 and ratio <= .95 + 1e-12 and arithmetic < 1
    return dict(passes=quality or saving, quality_pass=quality, saving_pass=saving,
                read_ratio=ratio, arithmetic_ratio=arithmetic, zero_overruns=ok)


def fwe_insufficient_disagreement(predictions, max_different=1):
    """Choose the prespecified CWE fallback from sparse calibration results."""
    grouped = {}
    for item in predictions:
        if item["policy"] in {p.name for p in SPARSE}:
            grouped.setdefault(item["sample_id"], {})[item["policy"]] = item["official_score"]
    if not grouped or any(len(scores) != len(SPARSE) for scores in grouped.values()):
        raise ValueError("incomplete paired FWE calibration")
    count = sum(len(set(scores.values())) > 1 for scores in grouped.values())
    return count <= max_different, count
