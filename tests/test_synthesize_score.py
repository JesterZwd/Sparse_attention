"""Regression checks for score synthesis receipts."""
from io import StringIO

import pytest

from scripts.synthesize_score import parse_area, timing_metrics, vcd_activity


def test_area_uses_top_module_only_and_counts_mapped_state():
    metrics = parse_area(
        {
            "modules": {
                "\\score": {
                    "area": 105.5, "num_cells": 7,
                    "num_cells_by_type": {"DFF_X1": 3, "DFFR_X1": 2, "NAND2_X1": 2},
                },
                "\\unused": {"area": 900, "num_cells": 18, "num_cells_by_type": {}},
            }
        },
        "score",
    )
    assert metrics["area_um2"] == 105.5
    assert metrics["cell_count"] == 7
    assert metrics["flop_bits"] == 5


def test_timing_throughput_counts_cycles_per_page():
    slow = timing_metrics(1.25, -0.25, 8, 9)
    assert slow["passes_target"] is False
    assert slow["critical_period_ns"] == pytest.approx(1.5)
    assert slow["target_rate_mpage_s"] == 0
    assert slow["critical_rate_mpage_s"] == pytest.approx(1e3 / 12)
    fast = timing_metrics(1.25, 0.1, 16, 17)
    assert fast["passes_target"] is True
    assert fast["target_rate_mpage_s"] == pytest.approx(50)


def test_activity_deduplicates_port_aliases_and_omits_unknown_transitions():
    vcd = StringIO(
        "$scope module tb $end\n"
        "$var wire 1 ! stim $end\n"
        "$scope module dut $end\n"
        "$var wire 1 ! clk $end\n"
        "$var wire 4 % sum [3:0] $end\n"
        "$var wire 4 % alias_sum [3:0] $end\n"
        "$var wire 1 & score_valid $end\n"
        "$upscope $end\n$upscope $end\n"
        "$enddefinitions $end\n"
        "#0\nx!\nb00x1 %\nx&\n"
        "#5\n1!\nb01x1 %\n0&\n"
        "#10\n0!\nb1111 %\n1&\n"
        "#15\n0&\n"
    )
    result = vcd_activity(vcd)
    assert result["observed_nets"] == 2
    assert result["observed_bits"] == 5
    assert result["known_bit_toggles"] == 4
    assert result["scored_pages"] == 1
    assert result["toggles_per_page"] == 4


def test_mapped_activity_uses_top_nets_without_cell_model_internal_signals():
    vcd = StringIO(
        "$scope module tb $end\n$scope module dut $end\n"
        "$var wire 1 ! score_valid $end\n"
        "$scope module cell_name $end\n$var wire 1 % cell_detail $end\n"
        "$upscope $end\n$upscope $end\n$upscope $end\n$enddefinitions $end\n"
        "#0\n0!\n0%\n#10\n1!\n1%\n#15\n0!\n0%\n"
    )
    result = vcd_activity(vcd, include_children=False)
    assert result["known_bit_toggles"] == 2
    assert result["observed_nets"] == 1
