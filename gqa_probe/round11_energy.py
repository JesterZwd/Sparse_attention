"""Scenario-based energy accounting for the Round 11 attention path."""

from collections import defaultdict
from itertools import combinations
from math import isfinite


BURST_BYTES = 32
HEAD_DIM = 128
HEADS_PER_GROUP = 4
AFFINE_GROUP_SIZE = 32
HISTOGRAM_BINS = 64
RESOURCE_DEFAULTS = dict(coefficient_ii=1, coefficient_latency_pe_cycles=2, sram_capacity_bytes=65536)
DESIGN_AXES = ("lanes", "pe_period_ns", "scorer_variant", "mode", *RESOURCE_DEFAULTS)
PRIMARY = (64, 1.25, "common_ii8", "double_buffer", *RESOURCE_DEFAULTS.values())

# These coefficients are engineering scenarios with explicit units. Their
# provenance is the declared model, independent of the LPDDR timing preset.
COEFFICIENTS = {
    "act_pj_per_command": (200, 500, 1200),
    "pre_pj_per_command": (100, 250, 600),
    "rd_core_pj_per_burst": (150, 400, 1000),
    "read_io_pj_per_byte": (3, 16, 40),
    "dram_background_mw": (2, 10, 40),
    "scorer_pj_per_term": (.12, .4, 1.2),
    "scorer_pj_per_fused_update": (.08, .3, 1.0),
    "qk_pj_per_product": (.15, .6, 2.0),
    "pv_pj_per_product": (.2, .8, 2.5),
    "qk_pj_per_accumulation": (.1, .35, 1.3),
    "pv_pj_per_accumulation": (.1, .35, 1.3),
    "qk_pj_per_logit_scale": (.15, .5, 2),
    "pv_pj_per_weight": (.25, 1, 4),
    "kv4_pj_per_affine_element": (.06, .2, .8),
    "kv4_pj_per_side_value": (.06, .25, 1.0),
    "sram_pj_per_byte": (.3, 1.5, 8),
    "control_pj_per_hist_score": (.1, .5, 2.0),
    "control_pj_per_ranked_page": (.3, 1.5, 6.0),
    "control_pj_per_descriptor_run": (.5, 3, 15),
    "control_background_mw": (.5, 2, 8),
    "lane_clock_mw_per_lane": (.001, .005, .020),
}
SCENARIOS = {
    name: {key: values[index] for key, values in COEFFICIENTS.items()}
    for index, name in enumerate(("lower", "nominal", "upper"))
}
COMPONENTS = ("dram", "scorer", "qk", "pv", "sram", "control")
TRAFFIC = ("summary_bytes", "k_bytes", "v_bytes")
PREDICTION_COUNTS = (
    "decode_tokens", "kv_group_calls", "read", *TRAFFIC, "k_records", "v_records",
    "scorer_terms", "fused_updates", "qk_mac", "pv_mac",
    "fallback_group_steps", "fallback_prefix_records", "scanned_pages", "k_pages", "prefix_pages",
)
SIDE_FIELDS = (
    "summary_payload_bytes", "summary_alignment_bytes",
    "k_code_bytes", "k_scale_bytes", "k_zero_point_bytes", "k_alignment_bytes",
    "v_code_bytes", "v_scale_bytes", "v_zero_point_bytes", "v_alignment_bytes",
    "summary_append_bytes", "summary_prefill_bytes",
)


def _number(row, key):
    value = float(row[key])
    if not isfinite(value) or value < 0:
        raise ValueError(f"{key} requires a finite nonnegative value")
    return value


def _integer(row, key):
    value = _number(row, key)
    if not value.is_integer():
        raise ValueError(f"{key} requires an integer")
    return int(value)


def _prediction_work(rows):
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate prediction for the same policy and prompt")
    counts = {key: sum(_integer(row, key) for row in rows) for key in PREDICTION_COUNTS}
    for key in SIDE_FIELDS:
        present = [row.get(key) is not None for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"partial coverage for optional closed-loop field {key}")
        counts[key] = sum(_integer(row, key) for row in rows) if all(present) else None
    for row in rows:
        if "generated_tokens" in row and _integer(row, "generated_tokens") != row["decode_tokens"] + 1:
            raise ValueError("generated tokens must include the first token produced by prefill")
        if row["read"] != sum(row[key] for key in TRAFFIC) or any(row[key] % BURST_BYTES for key in TRAFFIC):
            raise ValueError("closed-loop reads must equal the sum of the physical burst phases")
        if not row["decode_tokens"] and any(row[key] for key in PREDICTION_COUNTS):
            raise ValueError("a zero-step answer cannot incur decode work")
        if row["qk_mac"] != row["k_records"] * HEADS_PER_GROUP * HEAD_DIM or row["pv_mac"] % HEAD_DIM:
            raise ValueError("QK/PV counts disagree with the Round 11 head geometry")
        if row["fallback_group_steps"] > row["kv_group_calls"]:
            raise ValueError("fallback groups cannot exceed decode groups")
    if not counts["decode_tokens"]:
        raise ValueError("energy per decode step requires nonempty decode work")
    if any(row["kv_group_calls"] * counts["decode_tokens"] != row["decode_tokens"] * counts["kv_group_calls"]
           for row in rows):
        raise ValueError("closed-loop prompts disagree on groups per decode step")
    return counts


def _trace_stats(rows, predictions, prediction_work):
    indices = set()
    trace_ids = set()
    totals = defaultdict(float)
    matched_steps = 0
    matched_requests = 0
    closed_loop_bytes = 0.0
    observed_bytes = 0
    runtimes = []
    sram_peaks = []
    fallback_counts = []
    expected_groups = prediction_work["kv_group_calls"] / prediction_work["decode_tokens"]
    descriptor_coverage = [row.get("descriptor_runs") not in ("", None) for row in rows]
    has_descriptors = all(descriptor_coverage)
    if any(descriptor_coverage) and not has_descriptors:
        raise ValueError("partial descriptor coverage within one timing scenario")
    for row in rows:
        groups = _integer(row, "groups")
        layout_groups = _integer(row, "layers") * _integer(row, "kv_groups_per_layer")
        if groups != layout_groups or groups != expected_groups or not groups:
            raise ValueError("whole-step group coverage disagrees with the closed-loop geometry")
        sample = row["sample_id"]
        step = _integer(row, "step")
        if (sample, step) in indices:
            raise ValueError(f"duplicate whole-step timing for {sample}/{step}")
        indices.add((sample, step))
        trace_ids.add(sample)
        rd = _integer(row, "rd")
        observed = _integer(row, "total_bytes")
        act, pre = _integer(row, "act"), _integer(row, "pre")
        if observed != BURST_BYTES * rd or observed != sum(_integer(row, key) for key in TRAFFIC):
            raise ValueError("timing RD commands disagree with physical traffic")
        if not rd or not pre <= act <= rd:
            raise ValueError("timing ACT/PRE/RD command counts are inconsistent")
        totals["act"] += act
        totals["pre"] += pre
        totals["rd"] += rd
        totals["groups"] += groups
        runtimes.append(_number(row, "completion_ns"))
        if row.get("peak_sram_bytes") not in (None, ""):
            sram_peaks.append(_integer(row, "peak_sram_bytes"))
        if row.get("fallback_groups") not in (None, ""):
            count = _integer(row, "fallback_groups")
            if count > groups:
                raise ValueError("timing fallback count exceeds whole-step groups")
            fallback_counts.append(count)
        if has_descriptors:
            totals["descriptor_runs"] += _integer(row, "descriptor_runs")
        if sample in predictions:
            matched_requests += 1
            reference = predictions[sample]
            if step < reference["decode_tokens"]:
                matched_steps += 1
                closed_loop_bytes += reference["read"] / reference["decode_tokens"]
                observed_bytes += observed
    sampled_mean_bytes = BURST_BYTES * totals["rd"] / len(rows)
    prediction_mean_bytes = prediction_work["read"] / prediction_work["decode_tokens"]
    return dict(
        trace_steps=len(rows), trace_prompts=len(trace_ids), trace_groups=int(totals["groups"]),
        trace_matching_prompts=len(trace_ids.intersection(predictions)),
        trace_matching_sampled_steps=matched_steps, trace_steps_on_matching_prompts=matched_requests,
        trace_rd=int(totals["rd"]), trace_act=int(totals["act"]), trace_pre=int(totals["pre"]),
        trace_act_per_rd=totals["act"] / totals["rd"], trace_pre_per_rd=totals["pre"] / totals["rd"],
        trace_descriptor_runs_per_rd=totals["descriptor_runs"] / totals["rd"] if has_descriptors else None,
        trace_bytes_per_decode_step=sampled_mean_bytes,
        trace_vs_population_mean_bytes_relative_delta=(
            sampled_mean_bytes / prediction_mean_bytes - 1 if prediction_mean_bytes else None),
        trace_vs_matched_prompt_mean_bytes_relative_delta=(
            observed_bytes / closed_loop_bytes - 1 if closed_loop_bytes else None),
        trace_min_runtime_ns=min(runtimes), trace_max_runtime_ns=max(runtimes),
        trace_peak_sram_bytes=max(sram_peaks) if len(sram_peaks) == len(rows) else None,
        trace_fallback_groups=sum(fallback_counts) if len(fallback_counts) == len(rows) else None,
        runtime_ns_per_decode_step=sum(runtimes) / len(rows),
    )


def _activity(counts, trace, lanes):
    steps = counts["decode_tokens"]
    work = {key: value / steps if value is not None else None for key, value in counts.items()}
    rd = work["read"] / BURST_BYTES
    work.update(act=rd * trace["trace_act_per_rd"], pre=rd * trace["trace_pre_per_rd"], rd=rd,
                runtime_ns=trace["runtime_ns_per_decode_step"])
    rate = trace["trace_descriptor_runs_per_rd"]
    work["descriptor_runs"] = rd * rate if rate is not None else None
    work["lane_clock_ns"] = work["runtime_ns"] * lanes
    groups = work["kv_group_calls"]
    k, v = work["k_records"], work["v_records"]
    affine_values = 2 * (HEAD_DIM // AFFINE_GROUP_SIZE)
    hist_records = k + work["fallback_prefix_records"]
    hist_scores = hist_records * HEADS_PER_GROUP
    hist_bins = HISTOGRAM_BINS * HEADS_PER_GROUP * (groups + work["fallback_group_steps"])
    work.update(
        k_affine_elements=k * HEAD_DIM, v_affine_elements=v * HEAD_DIM,
        k_affine_side_values=k * affine_values, v_affine_side_values=v * affine_values,
        hist_scores=hist_scores, hist_bins=hist_bins,
        sram_staging_bytes=2 * work["read"],
        sram_unpacked_kv_bytes=2 * ((k + v) * HEAD_DIM + work["qk_mac"] / HEADS_PER_GROUP + work["pv_mac"]),
        sram_logit_weight_bytes=4 * (work["qk_mac"] / HEAD_DIM + 2 * hist_scores + work["pv_mac"] / HEAD_DIM),
        sram_histogram_bytes=4 * (3 * hist_bins + 2 * hist_scores),
        sram_mask_bytes=2 * (hist_scores + hist_records) / 8,
        sram_scorer_bytes=8 * (work["scanned_pages"] * HEADS_PER_GROUP + work["k_pages"]),
        sram_summary_append_bytes=work["summary_append_bytes"],
    )
    work["sram_bytes"] = sum(value for key, value in work.items()
                             if key.startswith("sram_") and value is not None)
    return work


def _component_energy(work, p):
    dram = {
        "dram_act": work["act"] * p["act_pj_per_command"],
        "dram_pre": work["pre"] * p["pre_pj_per_command"],
        "dram_rd_core": work["rd"] * p["rd_core_pj_per_burst"],
        "dram_read_io": work["read"] * p["read_io_pj_per_byte"],
        "dram_background": work["runtime_ns"] * p["dram_background_mw"],
    }
    scorer = {
        "scorer_terms": work["scorer_terms"] * p["scorer_pj_per_term"],
        "scorer_fused_updates": work["fused_updates"] * p["scorer_pj_per_fused_update"],
    }
    compute = {
        "qk_logit_scale": work["qk_mac"] / HEAD_DIM * p["qk_pj_per_logit_scale"],
        "pv_weights": work["pv_mac"] / HEAD_DIM * p["pv_pj_per_weight"],
    }
    for path, cache in (("qk", "k"), ("pv", "v")):
        compute.update({
            f"{path}_products": work[f"{path}_mac"] * p[f"{path}_pj_per_product"],
            f"{path}_accumulation": work[f"{path}_mac"] * p[f"{path}_pj_per_accumulation"],
            f"{path}_kv4_expand": work[f"{cache}_affine_elements"] * p["kv4_pj_per_affine_element"],
            f"{path}_kv4_side": work[f"{cache}_affine_side_values"] * p["kv4_pj_per_side_value"],
        })
    control = {
        "control_hist": work["hist_scores"] * p["control_pj_per_hist_score"],
        "control_ranking": work["scanned_pages"] * p["control_pj_per_ranked_page"],
        "control_descriptors": (
            work["descriptor_runs"] * p["control_pj_per_descriptor_run"]
            if work["descriptor_runs"] is not None else None),
        "control_background": work["runtime_ns"] * p["control_background_mw"],
        "control_lane_clock": work["lane_clock_ns"] * p["lane_clock_mw_per_lane"],
    }
    parts = {**dram, **scorer, **compute, "sram": work["sram_bytes"] * p["sram_pj_per_byte"], **control}
    for component in COMPONENTS:
        if component != "sram":
            parts[component] = sum(value for key, value in parts.items()
                                   if key.startswith(component + "_") and value is not None)
    parts["energy"] = sum(parts[key] for key in COMPONENTS)
    return parts


def _v_read_coefficients(trace, p):
    v_read = p["read_io_pj_per_byte"] + p["rd_core_pj_per_burst"] / BURST_BYTES
    activation = trace["trace_act_per_rd"] * p["act_pj_per_command"]
    precharge = trace["trace_pre_per_rd"] * p["pre_pj_per_command"]
    descriptors = (trace["trace_descriptor_runs_per_rd"] or 0) * p["control_pj_per_descriptor_run"]
    marginal = v_read + (activation + precharge + descriptors) / BURST_BYTES + 2 * p["sram_pj_per_byte"]
    return v_read, marginal


def _assumptions():
    return dict(
        scope="Round 11 decode attention and selector subsystem. Energy excludes prefill, projections, MLP, "
              "sampling and other model work. The first generated token is produced by prefill.",
        provenance="Illustrative, jointly varied engineering scenarios with uncalibrated coefficients. "
                   "They describe neither measurement uncertainty nor a specific device or process. "
                   "The DRAMPower reference supports the structure of the accounting.",
        reference="https://github.com/tukl-msd/DRAMPower",
        coefficients=SCENARIOS,
        primary=dict(zip(DESIGN_AXES, PRIMARY)),
        replay_resource_defaults=RESOURCE_DEFAULTS,
        dram="One 32 B per RD. RD core counts the core increment, read_io the traffic-dependent interface "
             "increment, and background_mw a constant during modeled completion. ACT and PRE reflect "
             "the replay command counts. Refresh, writes, WCK behavior, and end-of-group command activity "
             "absent from the replay are outside this estimate.",
        replay_estimator="Ordinary complete replay steps are weighted equally within each exact "
                         "policy/cohort/lane/PE-period/scorer/mode. Summed ACT, PRE, and descriptor counts "
                         "divided by sampled RD counts are applied to closed-loop RD counts. Runtime uses "
                         "mean sampled full-step completion. Selected common-state traces may differ from "
                         "independently generated states. The byte deltas compare the replay sample to the "
                         "closed-loop population and to whole-answer means at matched IDs. They describe "
                         "traffic representativeness. Runtime min/max describe the replay sample.",
        pressure="Only regular full-step rows enter estimates. True fallbacks found in ordinary closed-loop "
                 "predictions still enter the workload. Enriched fallback-pressure rows stay in the timing artifact.",
        compute="Exact scorer terms, fused score updates, and effective QK/PV MAC from closed-loop counters. "
                "Product and accumulation costs are separate. Model QK scale and PV weight energy once per "
                "effective head-score. KV4 uses one shared affine expansion per fetched "
                "physical element and one preparation per scale and zero-point for each 32-channel group. "
                "The DRAM byte ledger already includes side-field and alignment bursts.",
        sram="Modeled logical access bytes. Stage physical read bytes with one write and one read. Charge "
             "two-byte unpacked K/V writes, one shared K read per input element, and per-head V reads. Charge "
             "four bytes per score write, two histogram score reads, and one PV weight read. Charge "
             "histogram initialization and two bin scans plus a read-modify-write per visited candidate "
             "score. Protected scores use the same visit accounting. Packed masks use one read and "
             "one write, averaged without per-group alignment. Scorer pages and selected IDs use one "
             "four-byte write and read per head-score or ID. Count summary append bytes as SRAM writes when "
             "logged. This activity does not identify bank utilization or measure SRAM capacity.",
        control="Histogram/selection work uses candidate head-score visits, all scored pages, and descriptor "
                "runs when replayed. Time terms charge baseline and lane clocks throughout completion.",
        field_availability="Empty cells indicate fields absent from the supplied ledger or replay. Missing "
                           "descriptor and summary append counters omit those individual contributions. "
                           "Totals still include control time and SRAM staging.",
        normalization="Total closed-loop work divided by all generated decode steps for µJ/step. Multiply "
                      "by actual total decode steps divided by answers for mJ/answer. Every generated answer "
                      "has one additional token from prefill. ns × mW = pJ.",
        break_even="Positive delta means the candidate consumes more energy. V coefficient varies combined "
                   "RD-core and I/O energy per physical V byte. ACT/PRE, SRAM activity, and runtime are "
                   "held fixed for this coefficient sweep. SRAM coefficient varies energy per modeled "
                   "access byte. The reported interval spans the lower, nominal and upper settings of "
                   "remaining coefficients. Crossings can be negative and only nonnegative coefficients "
                   "are physical. The separate candidate V-byte ceiling varies its physical V bytes, "
                   "assuming unchanged total ACT/PRE/descriptor intensities, two-byte staging per read "
                   "byte, and unchanged compute, logical records, and runtime. It tests feasibility "
                   "only over nonnegative bytes. Actual layouts must preserve protection and whole bursts.",
    )


def estimate_energy(predictions, timing, *, cohort):
    """Combine closed-loop predictions with ordinary complete group replay."""
    predictions_by_policy = defaultdict(list)
    for row in predictions:
        predictions_by_policy[row["policy"]].append(row)
    timing_by_design = defaultdict(list)
    for row in timing:
        if str(row["cohort"]).lower() == cohort.lower() and row["scope"] == "regular":
            extras = tuple(_integer(row, key) if row.get(key) not in (None, "") else default
                           for key, default in RESOURCE_DEFAULTS.items())
            design = (_integer(row, "lanes"), _number(row, "pe_period_ns"),
                      row["scorer_variant"], row["mode"], *extras)
            timing_by_design[row["policy"], design].append(row)
    missing = set(predictions_by_policy).difference(policy for policy, _ in timing_by_design)
    if missing or not predictions_by_policy:
        raise ValueError(f"missing regular whole-step replay for {cohort} policies {sorted(missing)}")
    results = []
    for policy, rows in sorted(predictions_by_policy.items()):
        counts = _prediction_work(rows)
        prompt_index = {row["sample_id"]: row for row in rows}
        mean_steps = counts["decode_tokens"] / len(rows)
        for (label, design), trace_rows in sorted(timing_by_design.items()):
            if label != policy:
                continue
            resources = dict(zip(DESIGN_AXES, design))
            trace = _trace_stats(trace_rows, prompt_index, counts)
            peak = trace["trace_peak_sram_bytes"]
            trace["trace_sram_fits"] = peak <= resources["sram_capacity_bytes"] if peak is not None else None
            work = _activity(counts, trace, resources["lanes"])
            for scenario, coefficients in SCENARIOS.items():
                parts = _component_energy(work, coefficients)
                v_read, v_marginal = _v_read_coefficients(trace, coefficients)
                results.append(dict(
                    cohort=cohort, policy=policy, **resources,
                    primary=design == PRIMARY, scenario=scenario,
                    answers=len(rows), decode_steps=counts["decode_tokens"], mean_decode_steps=mean_steps,
                    mean_generated_tokens=mean_steps + 1, **trace,
                    **{f"{key}_per_decode_step": value for key, value in work.items()
                       if key not in {"runtime_ns", "decode_tokens"}},
                    runtime_ms_per_answer=work["runtime_ns"] * mean_steps / 1e6,
                    v_read_pj_per_byte=v_read, v_marginal_pj_per_byte=v_marginal,
                    sram_pj_per_byte=coefficients["sram_pj_per_byte"],
                    **{f"{key}_uj_per_decode_step": value / 1e6 if value is not None else None
                       for key, value in parts.items()},
                    **{f"{key}_mj_per_answer": value * mean_steps / 1e9 if value is not None else None
                       for key, value in parts.items()},
                ))
    return results, _assumptions()


def _crossing(difference, slope, coefficient):
    if slope == 0:
        return None, "all" if difference < 0 else "never"
    threshold = coefficient - difference / slope
    if slope < 0:
        return threshold, "above" if threshold >= 0 else "all"
    return threshold, "below" if threshold > 0 else "never"


def break_even_row(reference, candidate, normalization):
    """Solve each one-parameter crossing for one matched scenario."""
    if normalization == "decode_step":
        energy_key, scale = "energy_uj_per_decode_step", 1e6
        steps_reference = steps_candidate = 1
    elif normalization == "answer":
        energy_key, scale = "energy_mj_per_answer", 1e9
        steps_reference, steps_candidate = reference["mean_decode_steps"], candidate["mean_decode_steps"]
    else:
        raise ValueError("normalization must be decode_step or answer")
    difference = (candidate[energy_key] - reference[energy_key]) * scale
    result = {"normalization": normalization, "candidate_minus_reference_pj": difference}
    slopes = {}
    for label, byte_field, coefficient_field in (
        ("v", "v_bytes_per_decode_step", "v_read_pj_per_byte"),
        ("sram", "sram_bytes_per_decode_step", "sram_pj_per_byte"),
    ):
        coefficient = reference[coefficient_field]
        if coefficient != candidate[coefficient_field]:
            raise ValueError("break-even requires matched energy coefficients")
        slope = candidate[byte_field] * steps_candidate - reference[byte_field] * steps_reference
        slopes[label] = (slope, coefficient)
        threshold, condition = _crossing(difference, slope, coefficient)
        result.update({
            f"delta_{label}_bytes": slope,
            f"{label}_coefficient_pj_per_byte": coefficient,
            f"{label}_break_even_pj_per_byte": threshold,
            f"{label}_savings_condition": condition,
        })
    result["joint_intercept_pj"] = difference - sum(delta * coeff for delta, coeff in slopes.values())
    marginal = candidate.get("v_marginal_pj_per_byte")
    if marginal is not None:
        actual_v = candidate["v_bytes_per_decode_step"] * steps_candidate
        maximum_v = actual_v - difference / marginal
        result.update(v_actual_candidate_bytes=actual_v, v_max_candidate_bytes_for_savings=maximum_v,
                      v_candidate_bytes_change_to_tie=maximum_v - actual_v,
                      v_candidate_bytes_savings_feasible=maximum_v > 0)
    return result


def break_even_rows(energy_rows):
    axes = ("cohort", *DESIGN_AXES)
    groups = defaultdict(dict)
    for row in energy_rows:
        groups[tuple(row[key] for key in axes)][row["policy"], row["scenario"]] = row
    output = []
    for design, entries in sorted(groups.items()):
        policies = sorted({policy for policy, _ in entries})
        for reference, candidate in combinations(policies, 2):
            for unit in ("decode_step", "answer"):
                rows = [
                    dict(scenario=scenario, **break_even_row(
                        entries[reference, scenario], entries[candidate, scenario], unit))
                    for scenario in SCENARIOS
                ]
                for key in ("v", "sram"):
                    roots = [row[f"{key}_break_even_pj_per_byte"] for row in rows
                             if row[f"{key}_break_even_pj_per_byte"] is not None]
                    for row in rows:
                        row[f"{key}_min_break_even_pj_per_byte"] = min(roots) if roots else None
                        row[f"{key}_max_break_even_pj_per_byte"] = max(roots) if roots else None
                output.extend(dict(zip(axes, design), reference=reference, candidate=candidate,
                                   primary=design[1:] == PRIMARY, **row) for row in rows)
    return output
