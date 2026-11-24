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
