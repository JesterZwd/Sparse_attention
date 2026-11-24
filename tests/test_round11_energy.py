import csv
import json

import pytest

from gqa_probe.round11_energy import break_even_row, estimate_energy
from scripts.round11_energy import run


def prediction(policy, sample_id, steps, v_bytes):
    return dict(policy=policy, sample_id=sample_id, decode_tokens=steps, kv_group_calls=2 * steps,
                read=64 * steps + v_bytes, summary_bytes=32 * steps, k_bytes=32 * steps,
                v_bytes=v_bytes, k_records=2 * steps, v_records=steps, scorer_terms=8 * steps,
                fused_updates=4 * steps, qk_mac=1024 * steps, pv_mac=512 * steps,
                fallback_group_steps=0, fallback_prefix_records=0, scanned_pages=16 * steps,
                k_pages=2 * steps, prefix_pages=2 * steps)


def step(policy, sample_id, number, elapsed, act, scope="regular"):
    return dict(cohort="32k", policy=policy, sample_id=sample_id, step=number, groups=2,
                layers=1, kv_groups_per_layer=2, lanes=64, pe_period_ns=2.5, scorer_variant="base",
                mode="double_buffer", scope=scope, completion_ns=elapsed,
                act=act, pre=act - 1, rd=4, total_bytes=128, summary_bytes=32, k_bytes=32,
                v_bytes=64)


def test_energy_uses_total_burst_commands_and_real_answer_lengths():
    predictions = [prediction("A", "short", 1, 32), prediction("A", "long", 3, 96)]
    rows, assumptions = estimate_energy(
        predictions,
        [step("A", "short", 0, 100, 1), step("A", "long", 1, 300, 2),
         step("A", "long", 1, 2000, 4, "fallback_pressure")],
        cohort="32k",
    )
    nominal = next(row for row in rows if row["scenario"] == "nominal")
    assert nominal["decode_steps"] == 4
    assert nominal["mean_generated_tokens"] == 3
    assert nominal["trace_steps"] == 2
    assert nominal["act_per_decode_step"] == pytest.approx(1.125)
    assert nominal["pre_per_decode_step"] == pytest.approx(.375)
    assert nominal["rd_per_decode_step"] == 3
    assert nominal["runtime_ns_per_decode_step"] == 200
    assert nominal["dram_uj_per_decode_step"] == pytest.approx(.00539225)
    assert nominal["sram_unpacked_kv_bytes_per_decode_step"] == 2304
    assert nominal["energy_mj_per_answer"] == pytest.approx(nominal["energy_uj_per_decode_step"] * 2 / 1000)
    assert nominal["energy_uj_per_decode_step"] == pytest.approx(sum(
        nominal[f"{name}_uj_per_decode_step"]
        for name in ("dram", "scorer", "qk", "pv", "sram", "control")
    ))
    assert assumptions["coefficients"]["nominal"]["rd_core_pj_per_burst"] == 400


def test_missing_whole_step_group_is_rejected():
    incomplete = dict(step("A", "x", 0, 100, 1), groups=1)
    with pytest.raises(ValueError, match="whole-step group coverage"):
        estimate_energy([prediction("A", "x", 1, 32)], [incomplete], cohort="32k")


def test_coefficient_supply_sweep_keeps_same_step_in_distinct_designs():
    primary = dict(step("A", "x", 0, 100, 1), pe_period_ns=1.25, scorer_variant="common_ii8",
                   coefficient_ii=1, coefficient_latency_pe_cycles=2, sram_capacity_bytes=65536)
    slow = dict(primary, coefficient_ii=2, completion_ns=200)
    rows, _ = estimate_energy([prediction("A", "x", 1, 32)], [primary, slow], cohort="32k")
    nominal = {row["coefficient_ii"]: row for row in rows if row["scenario"] == "nominal"}
    assert nominal[1]["primary"]
    assert nominal[1]["runtime_ns_per_decode_step"] == 100
    assert not nominal[2]["primary"]
    assert nominal[2]["runtime_ns_per_decode_step"] == 200


def test_break_even_reports_direction_and_preserves_zero_slope():
    base = dict(energy_uj_per_decode_step=.0002, v_bytes_per_decode_step=10,
                sram_bytes_per_decode_step=5, v_read_pj_per_byte=8, sram_pj_per_byte=2)
    candidate = dict(base, energy_uj_per_decode_step=.00021, v_bytes_per_decode_step=5)
    result = break_even_row(base, candidate, "decode_step")
    assert result["v_break_even_pj_per_byte"] == pytest.approx(10)
    assert result["v_savings_condition"] == "above"
    assert result["sram_break_even_pj_per_byte"] is None
    assert result["sram_savings_condition"] == "never"


def test_cli_keeps_per_answer_break_even_distinct_and_writes_source_metadata(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([prediction("A", "x", 2, 64), prediction("B", "x", 1, 96)]))
    timing = tmp_path / "step_timing.csv"
    sample = [step("A", "x", 0, 100, 1), step("B", "x", 0, 100, 1)]
    with timing.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=sample[0]).writeheader()
        csv.DictWriter(handle, fieldnames=sample[0]).writerows(sample)
    output = tmp_path / "analysis"
    run(predictions, timing, output, "32k")
    with (output / "energy_break_even.csv").open() as handle:
        break_even = {row["normalization"]: row for row in csv.DictReader(handle)
                      if row["scenario"] == "nominal"}
    assert float(break_even["decode_step"]["delta_v_bytes"]) == 64
    assert float(break_even["answer"]["delta_v_bytes"]) == 32
    assert float(break_even["decode_step"]["candidate_minus_reference_pj"]) > 0
    assert float(break_even["answer"]["candidate_minus_reference_pj"]) < 0
    assert (float(break_even["decode_step"]["v_max_candidate_bytes_for_savings"]) <
            float(break_even["answer"]["v_max_candidate_bytes_for_savings"]))
    notes = json.loads((output / "energy_model.json").read_text())
    assert len(notes["inputs"]["timing"]["sha256"]) == 64
