"""Regression cases for packed transfers and Round 11 pipeline dependencies."""
import csv
import json

from gqa_probe.lpddr5_banked import read_banked
from gqa_probe.round11_timing import BankSession, replay_group
from scripts.replay_round11 import analyze


def trace():
    return dict(
        policy="C", summary_stride=112, summary_address=0, summary_bytes=448,
        history_pages=4, history_tokens=128, page_size=32, head_dim=128, heads=4,
        k_pages=[
            dict(id=0, records=32, addr=4194304, bytes=2432, v_addr=33554432),
            dict(id=1, records=32, addr=4196736, bytes=2432, v_addr=33556864),
        ],
        k_bytes=4864, v_bytes=608, total_bytes=5920, final_prefix=1, fallback=True,
        v_runs=[[33554432, 512], [33556480, 64], [33556736, 32]],
        pv_masks=[["ff"] * 4, ["0"] * 4], v_records=8, pv_by_head=[8] * 4,
        qk_mac=32768, pv_mac=4096, scorer_terms=3072, fused_updates=1536,
        scorer_channels=96, gate_tokens=64, prefix_records=32,
    )


def test_cross_record_bursts_and_fallback_are_paired():
    row = trace()
    # Four 112-byte records occupy 14 bursts, including two shared boundaries.
    results = {mode: replay_group(row, mode=mode) for mode in (
        "serial", "serial_one_slot", "summary_stream", "double_buffer", "dependency_bound"
    )}
    for result in results.values():
        assert result["rd"] == 185
        assert result["summary_rd"] == 14
        assert result["qk_mac"] == 32768
        assert result["pv_mac"] == 4096
        assert result["histograms"] == 2
        assert result["first_v_command_ns"] >= result["gate_done_ns"] > result["qk_done_ns"]
        assert result["dependency_lower_bound_ns"] <= result["completion_ns"]
        assert result["k_scale_bytes"] == 512
        assert result["v_zero_bytes"] == 32
    serial = results["serial"]
    assert serial["first_score_ns"] >= serial["summary_read_done_ns"]
    assert serial["first_qk_ns"] >= serial["k_read_done_ns"]
    stream = results["summary_stream"]
    assert stream["first_score_ns"] < stream["summary_read_done_ns"]
    assert stream["peak_summary_slots"] <= 2
    pipelined = results["double_buffer"]
    assert pipelined["peak_k_slots"] == 2
    assert pipelined["first_k_command_ns"] >= pipelined["topk_done_ns"]
    assert pipelined["first_qk_ns"] < pipelined["k_read_done_ns"] or (
        pipelined["k_read_done_ns"] < pipelined["qk_done_ns"]
    )


def test_shared_session_preserves_open_rows_and_work():
    row = trace()
    with BankSession() as session:
        cold = replay_group(row, mode="double_buffer", session=session)
        warm = replay_group(row, mode="double_buffer", session=session, address_offset=64 << 20)
    assert cold["act"] > 0 and warm["pre"] > 0
    assert cold["rd"] == warm["rd"] == 185
    assert cold["session_start_ns"] == 0
    assert warm["session_start_ns"] == cold["completion_ns"]


def test_c_sensitivity_does_not_change_work():
    row = trace()
    baseline = replay_group(row, mode="double_buffer")
    faster = replay_group(row, mode="double_buffer", scorer_variant="scorer_ii6")
    assert faster["scorer_busy_ns"] < baseline["scorer_busy_ns"]
    assert faster["completion_ns"] < baseline["completion_ns"]
    assert faster["scorer_terms"] == baseline["scorer_terms"]
    assert faster["rd"] == baseline["rd"]


def test_integrated_frontend_uses_whole_record_latency_once_and_keeps_physical_bursts():
    result = replay_group(trace(), frontend=rtl_timing())
    assert result["summary_rd"] == 14
    assert result["rd"] == 185
    assert result["peak_summary_slots"] == 2
    assert result["first_score_ns"] - result["frontend_first_record_ns"] == 42
    assert result["frontend_last_result_ns"] - result["frontend_last_record_ns"] >= 81
    assert result["scorer_busy_ns"] == 4 * 6 * 3
    assert result["coefficient_busy_ns"] == 4 * 4 * 3
    assert 2.5 <= result["score_done_ns"] - result["frontend_last_result_ns"] < 3.75
    assert result["first_v_command_ns"] >= result["gate_done_ns"] > result["qk_done_ns"]


def rtl_timing(**kwargs):
    from gqa_probe.round11_timing import FrontendTiming

    return FrontendTiming(
        profile="fixture", clock_period_ns=3, record_interval_cycles=13,
        record_to_dot_cycles=14, record_to_result_cycles=27, record_release_cycles=20,
        dot_ii_cycles=6, dot_tail_cycles=8, coefficient_service_cycles=4,
        coefficient_latency_cycles=12, record_slots=2, ingress_bytes_per_cycle=32,
        ingress_cdc_cycles=2, result_cdc_pe_cycles=2, auxiliary_bytes=32, **kwargs,
    )


def test_frontend_fifo_bounds_prefetch_without_duplicating_any_shared_burst():
    tight = replay_group(trace(), frontend=rtl_timing())
    buffered = replay_group(trace(), frontend=rtl_timing(burst_fifo_slots=8))
    assert tight["summary_rd"] == buffered["summary_rd"] == 14
    assert tight["peak_summary_slots"] == buffered["peak_summary_slots"] == 2
    assert buffered["peak_frontend_reserved_bytes"] - tight["peak_frontend_reserved_bytes"] == 256
    assert tight["summary_slot_wait_ns"] > buffered["summary_slot_wait_ns"]
    assert tight["score_done_ns"] > buffered["score_done_ns"]


def test_saturated_frontend_matches_full96_dual_rtl_timing():
    from gqa_probe.round11_timing import saturated_frontend_schedule

    complete, dot, reuse, output = saturated_frontend_schedule(rtl_timing(), 6, 112).T
    assert complete.tolist() == [3, 6, 29, 32, 55, 58]
    assert (dot - complete).tolist() == [14, 17] * 3
    assert (reuse - dot).tolist() == [6] * 6
    assert (output - dot).tolist() == [13] * 6


def test_projected_legacy_dot_clock_does_not_slow_coefficient_or_backend():
    baseline = replay_group(trace())
    slow = replay_group(trace(), legacy_scorer_period_ns=2.5)
    assert slow["scorer_busy_ns"] == 2 * baseline["scorer_busy_ns"]
    assert slow["coefficient_busy_ns"] == baseline["coefficient_busy_ns"]
    assert slow["completion_ns"] > baseline["completion_ns"]
    assert slow["qk_busy_ns"] == baseline["qk_busy_ns"]
    assert slow["rd"] == baseline["rd"]


def test_projected_legacy_accepts_equal_fifo_sensitivity():
    tight = replay_group(trace(), legacy_scorer_period_ns=2.5)
    buffered = replay_group(trace(), legacy_scorer_period_ns=2.5, legacy_burst_fifo_slots=16)
    assert buffered["frontend_burst_fifo_slots"] == 16
    assert buffered["peak_frontend_reserved_bytes"] >= 16 * 32
    assert buffered["rd"] == tight["rd"]
    assert buffered["score_done_ns"] < tight["score_done_ns"]


def test_barrier_matches_existing_banked_scan_and_exposes_buffer_stalls():
    row = trace()
    serial = replay_group(row, mode="serial")
    expected = read_banked([[(0, 448)]])["phase_end_ck"][0] * 1.25
    assert serial["summary_read_done_ns"] == expected
    assert serial["data_compute_overlap_ns"] == 0
    stream = replay_group(row, mode="summary_stream")
    assert stream["score_data_overlap_ns"] > 0
    assert stream["k_slot_wait_ns"] > 0
    assert stream["qk_data_overlap_ns"] == 0
    pipelined = replay_group(row, mode="double_buffer")
    assert pipelined["qk_data_overlap_ns"] > 0
    assert pipelined["peak_live_sram_bytes"] <= pipelined["peak_sram_bytes"]


def test_join_preserves_natural_fallback_in_whole_step(tmp_path):
    source = tmp_path / "states.jsonl"
    rows = []
    for group in range(2):
        row = trace()
        row.update(cohort="fixture", sample_id="p0",
                   step=1, layer=0, group=group, whole_step=True, fallback=True,
                   scope="fallback_pressure" if group else "regular")
        rows.append(row)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "replay"
    receipt = analyze([source], output, lanes=(64,), periods=(1.25,), modes=("double_buffer",),
                      ii6=False, layers=1, groups=2)
    assert receipt["whole_steps"] == 1
    with (output / "step_timing.csv").open() as file:
        step, = csv.DictReader(file)
    assert step["scope"] == "regular"
    assert step["rd"] == "370"
    assert step["groups"] == "2"
    assert step["fallback_groups"] == "2"
    assert float(step["completion_ns"]) == float(step["last_group_end_ns"])
