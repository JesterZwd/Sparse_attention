from scripts.report_ruler_round2 import decide


def test_accuracy_route_requires_tail_noninferiority_and_cost():
    primary = dict(complete_correct={"ci_low": .003},
                   absolute_error_p99={"relative_delta": .002, "relative_ci_high": .009,
                                       "ci_high": .015})
    assert decide(primary, 1.25)["passed"]
    assert not decide(primary, 1.2501)["passed"]
    primary["absolute_error_p99"]["relative_ci_high"] = .011
    assert not decide(primary, 1.)["passed"]


def test_tail_route_requires_effect_size_and_paired_accuracy_noninferiority():
    primary = dict(complete_correct={"ci_low": -.007},
                   absolute_error_p99={"relative_delta": -.018, "relative_ci_high": -.003,
                                       "ci_high": -.005})
    assert decide(primary, 1.1)["passed"]
    primary["complete_correct"]["ci_low"] = -.011
    assert not decide(primary, 1.1)["passed"]
    primary["complete_correct"]["ci_low"] = 0
    primary["absolute_error_p99"]["relative_delta"] = -.009
    assert not decide(primary, 1.1)["passed"]
