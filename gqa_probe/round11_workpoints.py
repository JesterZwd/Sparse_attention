"""Physically budgeted Round 11 work points and frozen algorithm gates."""
from dataclasses import dataclass

from .compact_summary import SUMMARY_BYTES
from .summary_decode import plan


@dataclass(frozen=True)
class Workpoint:
    name: str
    method: str
    extra_pages: int | None = None
    page: int = 32
    budget: float = 5
    v_reserve: float = .80


A = Workpoint("A", "mr25")
B = Workpoint("B", "mr96_score", 0)
C = Workpoint("C", "mr96_score")
M = Workpoint("M", "mr96_score", 4)
BASELINE = {item.name: item for item in (A, B, C, M)}
GATES = dict(quality_margin=-.01, min_read_saving=.05, max_qk_ratio=1.01,
             min_quality_gain=.01, max_quality_read_ratio=1.01)


def allocation(policy, ranks, history):
    args = dict(budget_percent=policy.budget, v_reserve=policy.v_reserve)
    limit = None
    if policy.extra_pages is not None:
        # Only the width of the 144 B plan is inherited. The rank tensor can
        # belong to the candidate because width and mandatory pages depend on layout.
        reference = plan(ranks, history, summary_bytes=SUMMARY_BYTES[A.method], **args)
        limit = reference.ids.shape[1] + policy.extra_pages
    return plan(ranks, history, summary_bytes=SUMMARY_BYTES[policy.method], max_pages=limit, **args)


def verdict(policy, ref, new, score_delta, ci_low):
    """Evaluate the name-specific algorithm gate on completed paired results."""
    read_ratio = new["read_per_decode"] / ref["read_per_decode"]
    qk_ratio = new["qk_per_decode"] / ref["qk_per_decode"]
    arithmetic_ratio = new["arithmetic_per_decode"] / ref["arithmetic_per_decode"]
    zero = not (ref["overrun_groups"] or new["overrun_groups"])
    ni = ci_low >= GATES["quality_margin"] - 1e-12
    gain = score_delta >= GATES["min_quality_gain"] - 1e-12 and ci_low > 0
    save = (read_ratio <= 1 - GATES["min_read_saving"] + 1e-12 and
            qk_ratio <= GATES["max_qk_ratio"] + 1e-12 and arithmetic_ratio < 1.)
    if policy.name == B.name:
        passes = zero and ni and save
    else:
        passes = zero and gain and read_ratio <= GATES["max_quality_read_ratio"] + 1e-12
    return dict(passes=passes, zero_overruns=zero, quality_noninferior=ni,
                quality_gain=gain, read_ratio=read_ratio, qk_ratio=qk_ratio,
                arithmetic_ratio=arithmetic_ratio)
