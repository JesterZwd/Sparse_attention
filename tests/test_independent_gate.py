from gqa_probe.independent_gate import independent_decision


def compare(score=.02,low=.003,ref=100.,new=100.8,total=-.01,total_hi=0.,**extra):
    return independent_decision(score_delta=score,score_ci_low=low,
                                reference_bytes=ref,candidate_bytes=new,
                                answer_read_relative_delta=total,answer_read_relative_hi=total_hi,**extra)


def test_score_improvement_needs_same_actual_reads_and_zero_violations():
    assert compare()["passes"]
    assert not compare(new=101.01)["passes"]
    assert not compare(candidate_over=1)["passes"]
    assert not compare(low=0.)["passes"]


def test_noninferiority_needs_measured_cost_gain_on_independent_set():
    assert not compare(score=0.,low=-.001)["passes"]
    assert compare(score=0.,low=-.009,new=95.,total=-.15,total_hi=-.101)["passes"]
    assert not compare(score=0.,low=-.011,new=95.,total=-.15,total_hi=-.101)["passes"]
    assert not compare(score=0.,low=-.009,new=95.,total=-.15,total_hi=-.09)["passes"]


def test_optional_cycles_route_requires_independent_measured_cycle_gain():
    assert compare(score=0.,low=-.005,cycle_reduction=.06,min_cycle_reduction=.05)["passes"]
    assert not compare(score=0.,low=-.005,cycle_reduction=.04,min_cycle_reduction=.05)["passes"]
