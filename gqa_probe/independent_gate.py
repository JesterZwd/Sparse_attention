"""Apply preregistered quality and physical-cost checks to independent results."""
from math import isfinite


def independent_decision(*, score_delta, score_ci_low, reference_bytes, candidate_bytes,
                         answer_read_relative_delta, answer_read_relative_hi,
                         reference_over=0, candidate_over=0, read_tolerance=.01,
                         min_score_gain=.01, ni_margin=.01, min_answer_saving=.10,
                         cycle_reduction=None, min_cycle_reduction=None):
    """The score interval uses candidate minus control. Savings use fractions."""
    inputs=(score_delta,score_ci_low,reference_bytes,candidate_bytes,answer_read_relative_delta)
    if not all(isfinite(x) for x in inputs) or reference_bytes<=0 or candidate_bytes<0:
        raise ValueError("independent comparison requires finite quality and byte totals")
    read_change=(candidate_bytes-reference_bytes)/reference_bytes
    bytes_matched=abs(read_change)<=read_tolerance+1e-12
    budget_ok=reference_over==0 and candidate_over==0
    ni=score_ci_low>=-ni_margin-1e-12
    score_route=bytes_matched and score_delta>=min_score_gain-1e-12 and score_ci_low>0
    save_route=(ni and read_change<=read_tolerance+1e-12 and
                answer_read_relative_hi is not None and isfinite(answer_read_relative_hi) and
                answer_read_relative_delta<=-min_answer_saving+1e-12 and
                answer_read_relative_hi<=-min_answer_saving+1e-12)
    cycle_route=(ni and read_change<=read_tolerance+1e-12 and
                 min_cycle_reduction is not None and cycle_reduction is not None and
                 isfinite(cycle_reduction) and cycle_reduction>=min_cycle_reduction-1e-12)
    return dict(passes=budget_ok and (score_route or save_route or cycle_route),
                zero_overruns=budget_ok, matched_bytes=bytes_matched, read_delta=read_change,
                meets_quality_noninferiority=ni, quality_route=bool(score_route),
                saving_route=bool(save_route), cycle_route=bool(cycle_route))
