"""Deterministic CPU replay of one Round 11 shared attention resource."""

import ctypes
from dataclasses import dataclass
from functools import lru_cache, reduce
from hashlib import sha256
from math import isfinite
from operator import or_
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .round7_gates import gate_cost

CK_NS = 1.25
BURST_BYTES = 32
QUEUE_DEPTH = 32
SRAM_BYTES = 64 * 1024
PRIMARY_PE_NS = 1.25
ROUND9_SCORER_REFERENCE_NS = 2.385448
LANES = (32, 64, 128)
MODES = ("serial", "serial_one_slot", "summary_stream", "double_buffer", "dependency_bound")
BASE_SCORER = "common_ii8"
FAST_SCORER = "scorer_ii6"
GROUP_REGION_BYTES = 64 << 20
BANK_ROW_BYTES = 2048
V_BASE = 1 << 25

# Order matches Metric in the C interface. Timestamps are relative to group start
# except session_start. Negative timestamps mean the operation did not occur.
METRICS = (
    "session_start_ns", "completion_ns", "first_score_ns", "score_done_ns",
    "summary_read_done_ns", "topk_done_ns", "first_k_command_ns", "first_qk_ns",
    "k_read_done_ns", "qk_done_ns", "gate_done_ns", "first_v_command_ns",
    "first_v_read_ns", "first_v_return_ns", "first_pv_ns", "v_read_done_ns", "pv_done_ns",
    "rd", "act", "pre", "row_hits", "summary_rd", "k_rd", "v_rd", "command_ck",
    "peak_inflight_bursts", "peak_pending_descriptors", "descriptor_runs", "scorer_wait_ns",
    "qk_wait_ns", "pv_wait_ns", "scorer_busy_ns", "qk_busy_ns", "pv_busy_ns",
    "k_unpack_ns", "v_unpack_ns", "pv_weights_ns", "dram_outstanding_ns",
    "peak_sram_bytes", "peak_summary_sram_bytes", "peak_k_sram_bytes",
    "peak_gate_sram_bytes", "peak_v_sram_bytes", "peak_summary_slots", "peak_k_slots",
    "peak_v_slots", "peak_packed_tile_bytes", "peak_unpacked_tile_bytes",
    "dependency_lower_bound_ns", "coefficient_busy_ns", "score_data_overlap_ns",
    "qk_data_overlap_ns", "pv_data_overlap_ns", "request_window_wait_ns",
    "summary_slot_wait_ns", "k_slot_wait_ns", "v_slot_wait_ns", "peak_live_sram_bytes",
    "peak_pv_weight_sram_bytes",
)

FRONTEND_METRICS = (
    "frontend_first_record_ns", "frontend_last_record_ns", "frontend_last_result_ns",
    "frontend_ingress_queue_wait_ns", "frontend_record_service_wait_ns", "frontend_result_cdc_ns",
    "frontend_ingress_port_busy_ns", "peak_frontend_reserved_bytes",
)


def ceil_div(n, d):
    return (n + d - 1) // d


def burst_round(n):
    return BURST_BYTES * ceil_div(n, BURST_BYTES)


def planes(n, dim):
    return n * dim // 2, burst_round(n * dim // 16), burst_round(n * dim // 32)


def coalesced(ranges):
    """Canonical contiguous ranges. Overlap would read the same physical burst twice."""
    merged = []
    for start, size in sorted(ranges):
        if start < 0 or size <= 0 or start % BURST_BYTES or size % BURST_BYTES:
            raise ValueError("DRAM ranges must be positive and 32-byte aligned")
        if merged:
            prior, length = merged[-1]
            if start < prior + length:
                raise ValueError("physical DRAM ranges contain repeated bytes")
            if start == prior + length:
                merged[-1] = (prior, length + size)
                continue
        merged.append((start, size))
    return merged


def _engine_sources():
    sim = Path(__file__).resolve().parents[1] / "sim"
    return sim / "round11_timing.cpp", sim / "lpddr5_banked.cpp"


@lru_cache(maxsize=1)
def _engine():
    source, model = _engine_sources()
    digest = sha256(source.read_bytes() + model.read_bytes()).hexdigest()[:16]
    target = Path(".cache/round11") / f"replay-{digest}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        with tempfile.TemporaryDirectory(dir=target.parent) as directory:
            product = Path(directory) / target.name
            subprocess.run(["g++", "-std=c++17", "-O3", "-shared", "-fPIC",
                            str(source), "-o", str(product)], check=True)
            product.replace(target)
    lib = ctypes.CDLL(str(target.resolve()))
    vector = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    lib.round11_open.restype = ctypes.c_void_p
    lib.round11_close.argtypes = [ctypes.c_void_p]
    lib.round11_close.restype = None
    lib.replay_round11.argtypes = [ctypes.c_void_p, vector, vector, vector, vector]
    lib.replay_round11.restype = ctypes.c_int
    lib.round12_saturated_frontend.argtypes = [vector, ctypes.c_int64, ctypes.c_int64, vector]
    lib.round12_saturated_frontend.restype = None
    return lib


class BankSession:
    """One LPDDR channel, one command history, and one monotonically advancing timebase."""

    def __init__(self):
        self.lib = _engine()
        self.handle = self.lib.round11_open()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lib.round11_close(self.handle)
        self.handle = None

    def run(self, config, ranges, tiles):
        if not self.handle:
            raise ValueError("bank session has closed")
        output = np.zeros(len(METRICS) + len(FRONTEND_METRICS), dtype=np.int64)
        if self.lib.replay_round11(self.handle, config, ranges, tiles, output):
            raise RuntimeError("the pipeline reached a state without a future event")
        return {
            name: (None if n < 0 else n / 1_000_000) if name.endswith("_ns") else n
            for name, n in zip((*METRICS, *FRONTEND_METRICS), output.tolist(), strict=True)
        }


@dataclass(frozen=True)
class PreparedGroup:
    raw: dict
    ranges: np.ndarray
    tiles: np.ndarray
    stats: dict


@dataclass(frozen=True)
class FrontendTiming:
    """Equivalent per-record timing obtained from an integrated external prefix.

    The measured minimum record and sustainable intervals are calibration metadata.
    The recurrence derives each page's milestones from the coefficient/dot stages.
    """

    profile: str
    clock_period_ns: float
    record_interval_cycles: float
    record_to_dot_cycles: int
    record_to_result_cycles: int
    record_release_cycles: int
    dot_ii_cycles: int
    dot_tail_cycles: int
    coefficient_service_cycles: int
    coefficient_latency_cycles: int
    record_slots: int
    ingress_bytes_per_cycle: int
    ingress_cdc_cycles: int
    result_cdc_pe_cycles: int
    auxiliary_bytes: int
    result_fifo_slots: int = 2
    query_load_cycles: int = 0
    burst_fifo_slots: int = 0

    def __post_init__(self):
        if not self.profile or not isfinite(self.clock_period_ns) or self.clock_period_ns < .000001:
            raise ValueError("frontend requires a name and a positive scorer clock")
        positive = (self.record_to_result_cycles, self.dot_ii_cycles, self.dot_tail_cycles,
                    self.coefficient_service_cycles, self.coefficient_latency_cycles,
                    self.record_slots, self.ingress_bytes_per_cycle, self.result_fifo_slots)
        nonnegative = (self.record_to_dot_cycles, self.record_release_cycles, self.ingress_cdc_cycles,
                       self.result_cdc_pe_cycles, self.auxiliary_bytes, self.query_load_cycles, self.burst_fifo_slots)
        if (any(type(value) is not int or value < 1 for value in positive) or
                any(type(value) is not int or value < 0 for value in nonnegative) or
                not isfinite(self.record_interval_cycles) or
                self.record_interval_cycles < max(self.dot_ii_cycles, self.coefficient_service_cycles) or
                self.record_to_dot_cycles != self.coefficient_latency_cycles + 2 or
                self.record_release_cycles != self.record_to_dot_cycles + self.dot_ii_cycles or
                self.record_to_result_cycles != self.record_release_cycles - 1 + self.dot_tail_cycles or
                BURST_BYTES % self.ingress_bytes_per_cycle):
            raise ValueError("inconsistent measured frontend service or ingress resources")

    def config(self):
        return (round(self.clock_period_ns * 1_000_000), self.dot_ii_cycles, self.dot_tail_cycles, self.coefficient_service_cycles,
                self.coefficient_latency_cycles, self.record_slots,
                ceil_div(BURST_BYTES, self.ingress_bytes_per_cycle), self.ingress_cdc_cycles,
                self.result_cdc_pe_cycles, self.auxiliary_bytes, self.result_fifo_slots,
                self.query_load_cycles, self.burst_fifo_slots)


def saturated_frontend_schedule(frontend, count, stride):
    """Return complete, first dot, reusable and output cycles under continuous ingress."""
    if count < 1 or stride < BURST_BYTES or (stride % BURST_BYTES and frontend.record_slots < 2):
        raise ValueError("a straddling stream needs two slots and at least one whole record")
    output = np.zeros(4 * count, dtype=np.int64)
    _engine().round12_saturated_frontend(np.array(frontend.config(), dtype=np.int64), count, stride, output)
    return output.reshape((count, 4))


@lru_cache(maxsize=4096)
def _gate(tokens, heads, optional, pages):
    return gate_cost("hist64", tokens, heads, optional_tokens=optional, pages=pages)


def _history_protected(tokens, size, pages):
    sink = range(ceil_div(min(32, tokens), size))
    local = range(max(0, tokens - 128) // size, pages)
    return set(sink) | set(local)


def _value_tile(page, masks, dim, default_address):
    records = page["records"]
    if any(mask < 0 or mask >= 1 << records for mask in masks):
        raise ValueError("PV mask includes absent records")
    union = reduce(or_, masks, 0)
    if not union:
        return None
    nonzero = [i for i in range(records) if (union >> i) & 1]
    address = page.get("v_addr", default_address)
    codes, scales, _ = planes(records, dim)
    code_bytes = dim // 2
    scale_ids = set(i * (dim // 16) // BURST_BYTES for i in nonzero)
    zero_ids = set(i * (dim // 32) // BURST_BYTES for i in nonzero)
    ranges = coalesced([
        *((address + code_bytes * i, code_bytes) for i in nonzero),
        *((address + codes + BURST_BYTES * i, BURST_BYTES) for i in scale_ids),
        *((address + codes + scales + BURST_BYTES * i, BURST_BYTES) for i in zero_ids),
    ])
    head_records = [mask.bit_count() for mask in masks]
    physical = len(nonzero)
    return (ranges, physical, head_records, physical * code_bytes, len(scale_ids) * BURST_BYTES,
            len(zero_ids) * BURST_BYTES)


def prepare_group(row):
    """Validate actual physical ranges, packed planes, operation counts, and fallback masks."""
    page_size, dim, heads = (row[key] for key in ("page_size", "head_dim", "heads"))
    if page_size not in (32, 64) or dim != 128 or heads < 1:
        raise ValueError("the replay models P32/P64 affine32 pages and 128-channel heads")
    history_pages, stride = row["history_pages"], row["summary_stride"]
    summary_bytes = row["summary_bytes"]
    if history_pages <= 0 or stride <= 0 or summary_bytes != burst_round(history_pages * stride):
        raise ValueError("the summary scan must contain one rounded packed stride per history page")
    summary_address = row["summary_address"]
    pages = row["k_pages"]
    ids = [page["id"] for page in pages]
    prefix = row["final_prefix"]
    if (not 0 < prefix <= len(pages) or len(ids) != len(set(ids)) or
            min(ids) < 0 or max(ids) >= history_pages):
        raise ValueError("invalid ranking or final K prefix")
    if "initial_prefix" in row and row["initial_prefix"] != len(pages):
        raise ValueError("the initial prefix does not match the K transaction count")
    if not row["fallback"] and prefix != len(pages):
        raise ValueError("a shorter final K prefix requires a fallback")
    masks = [[int(v, 16) if isinstance(v, str) else int(v) for v in group]
             for group in row["pv_masks"]]
    if (len(masks) != len(pages) or any(len(mask) != heads for mask in masks) or
            any(any(mask) for mask in masks[prefix:])):
        raise ValueError("PV masks must be head-complete and limited to the final K prefix")

    ranges, tiles, keys, values = [], [], [], []
    stat = {f"{kind}_{plane}_bytes": 0 for kind in ("k", "v") for plane in ("code", "scale", "zero")}
    stat.update(k_records=0, v_records=0)

    def remember(kind, tile_ranges, records, macs, code, scale, zero):
        first = len(ranges)
        total = code + scale + zero
        ranges.extend(tile_ranges)
        tiles.append((records, macs, total, first, len(ranges)))
        stat[f"{kind}_records"] += records
        for name, n in zip(("code", "scale", "zero"), (code, scale, zero), strict=True):
            stat[f"{kind}_{name}_bytes"] += n

    for page in pages:
        records = page["records"]
        fields = planes(records, dim)
        if not 0 < records <= page_size or sum(fields) != page["bytes"]:
            raise ValueError("K page byte count disagrees with the affine32 planes")
        key_ranges = [(page["addr"], page["bytes"])]
        remember("k", key_ranges, records, records * heads * dim, *fields)
        keys.extend(key_ranges)

    head_records = [0] * heads
    v_stride = sum(planes(page_size, dim))
    for slot in sorted(range(len(ids)), key=ids.__getitem__):
        data = _value_tile(pages[slot], masks[slot], dim, V_BASE + ids[slot] * v_stride)
        if data is not None:
            tile_ranges, records, counts, *fields = data
            remember("v", tile_ranges, records, sum(counts) * dim, *fields)
            head_records = [a + b for a, b in zip(head_records, counts, strict=True)]
            values.extend(tile_ranges)
    if coalesced(values) != coalesced([tuple(run) for run in row["v_runs"]]):
        raise ValueError("head PV masks and V address trace disagree")
    physical = coalesced([(summary_address, summary_bytes), *keys, *values])
    if physical[-1][0] + physical[-1][1] > GROUP_REGION_BYTES - 16 * BANK_ROW_BYTES:
        raise ValueError("local address exceeds the reserved group region")
    for kind, kind_ranges in zip(("k", "v"), (keys, values), strict=True):
        if sum(n for _, n in kind_ranges) != row[f"{kind}_bytes"]:
            raise ValueError(f"{kind.upper()} physical-byte ledger differs from address trace")
    if row.get("cap_bytes") is not None and row["total_bytes"] > row["cap_bytes"]:
        raise ValueError("a captured path exceeded its physical byte cap")
    if summary_bytes + row["k_bytes"] + row["v_bytes"] != row["total_bytes"]:
        raise ValueError("read ledger differs from the total physical byte count")
    if (head_records != row["pv_by_head"] or row.get("v_records", stat["v_records"]) != stat["v_records"]
            or row["qk_mac"] != stat["k_records"] * heads * dim or row["pv_mac"] != sum(head_records) * dim):
        raise ValueError("packed-record and per-head operation counts disagree")

    channels = row.get("scorer_channels", 128)
    updates = history_pages * heads * channels
    if row["fused_updates"] != updates or row["scorer_terms"] != 2 * updates:
        raise ValueError("the fused mean/radius scorer operation ledger disagrees with its geometry")
    raw_gate_count = row.get("gate_tokens", stat["k_records"])
    raw_prefix_count = row.get("prefix_records", sum(page["records"] for page in pages[:prefix]))
    if raw_gate_count < stat["k_records"] or raw_prefix_count < sum(p["records"] for p in pages[:prefix]):
        raise ValueError("hist64 replay cannot process fewer scores than were computed")
    protected_ids = _history_protected(row["history_tokens"], page_size, history_pages)
    pinned = sum(page["records"] for page in pages if page["id"] in protected_ids)
    pinned_prefix = sum(page["records"] for page in pages[:prefix] if page["id"] in protected_ids)
    initial = _gate(raw_gate_count, heads, raw_gate_count - pinned, len(pages))
    retry = _gate(raw_prefix_count, heads, raw_prefix_count - pinned_prefix, prefix) if row["fallback"] else {}
    controls = ("histogram_updates", "histogram_bin_visits", "gate_exponentials",
                "gate_comparisons", "gate_reduce_adds", "mask_sram_reads",
                "mask_sram_writes", "logits_sram_reads")
    stat.update({name: initial[name] + retry.get(name, 0) for name in controls})
    stat.update(
        histograms=1 + bool(retry),
        histogram_sram_bytes=initial["histogram_bytes"],
        gate_control_sram_bytes=max(initial["control_sram_bytes"], retry.get("control_sram_bytes", 0)),
        logits_sram_writes=stat["k_records"] * heads,
        gate_budget_pe_cycles=len(pages) + (prefix if retry else 0),
        gate_pe_cycles=initial["gate_ck"] + retry.get("gate_ck", 0),
        prefix_records=raw_prefix_count,
        gate_tokens=raw_gate_count,
        k_unpack_elements=stat["k_records"] * dim,
        v_unpack_elements=stat["v_records"] * dim,
        pv_weight_exponentials=sum(head_records),
        pv_weight_sram_write_bytes=4 * sum(head_records),
        pv_weight_sram_read_bytes=4 * sum(head_records),
        peak_scorer_score_sram_bytes=4 * heads * history_pages,
        logical_v_tiles=len(tiles) - len(pages),
        packed_sram_write_bytes=row["total_bytes"],
        packed_sram_read_bytes=row["total_bytes"],
        unpack_sram_write_bytes=2 * (stat["k_records"] + stat["v_records"]) * dim,
        unpack_sram_read_bytes=row["qk_mac"] // heads * 2 + row["pv_mac"] * 2,
        peak_pv_accumulator_sram_bytes=heads * (dim + 2) * 4,
        coefficient_records=history_pages,
        coefficient_scale_values=2 * history_pages,
        summary_payload_bytes=history_pages * stride,
        summary_alignment_bytes=summary_bytes - history_pages * stride,
        k_zero_point_bytes=stat["k_zero_bytes"],
        v_zero_point_bytes=stat["v_zero_bytes"],
        peak_logits_bytes=raw_gate_count * heads * 4,
        peak_logits_sram_bytes=raw_gate_count * heads * 4,
    )
    return PreparedGroup(row, np.array(ranges, dtype=np.int64).reshape(-1, 2),
                         np.array(tiles, dtype=np.int64).flatten(), stat)


def group_address_offset(ordinal):
    """Disjoint regions with a fixed round-robin bank rotation across actual groups."""
    return GROUP_REGION_BYTES * ordinal + (ordinal % 16) * BANK_ROW_BYTES


def replay_group(row, *, mode="double_buffer", lanes=64, pe_period_ns=PRIMARY_PE_NS,
                 scorer_variant=BASE_SCORER, coefficient_ii=1, coefficient_latency=2,
                 sram_capacity_bytes=SRAM_BYTES, session=None, address_offset=0, frontend=None,
                 legacy_scorer_period_ns=None, legacy_burst_fifo_slots=0):
    """Replay one matched state and return stage times, work counts, and simultaneous peaks."""
    prepared = row if isinstance(row, PreparedGroup) else prepare_group(row)
    raw, stat = prepared.raw, prepared.stats
    if (mode not in MODES or lanes not in LANES or pe_period_ns <= 0 or
            coefficient_ii < 1 or coefficient_latency < 1 or sram_capacity_bytes <= 0):
        raise ValueError("unsupported replay resource")
    scorer_channels = raw.get("scorer_channels", 128)
    if scorer_variant not in (BASE_SCORER, FAST_SCORER) or (scorer_variant == FAST_SCORER and scorer_channels != 96):
        raise ValueError("the II=6 scorer sensitivity requires 96 summary channels")
    if frontend is not None and (mode in ("serial", "dependency_bound") or scorer_variant != BASE_SCORER or
                                 coefficient_ii != 1 or coefficient_latency != 2 or
                                 legacy_scorer_period_ns is not None or legacy_burst_fifo_slots):
        raise ValueError("measured frontend needs finite slots and replaces legacy scorer timing")
    if frontend is not None and raw["summary_stride"] % BURST_BYTES and (
        frontend.record_slots < 2 or mode == "serial_one_slot"
    ):
        raise ValueError("the integrated straddling input requires two record slots")
    if legacy_scorer_period_ns is not None and (not isfinite(legacy_scorer_period_ns) or legacy_scorer_period_ns < .000001):
        raise ValueError("legacy scorer clock must be positive")
    if type(legacy_burst_fifo_slots) is not int or legacy_burst_fifo_slots < 0:
        raise ValueError("legacy burst FIFO size must be a nonnegative integer")
    if address_offset < 0 or address_offset % BURST_BYTES:
        raise ValueError("group address remap must be burst aligned")
    if session is None:
        with BankSession() as cold:
            return replay_group(prepared, mode=mode, lanes=lanes, pe_period_ns=pe_period_ns,
                                scorer_variant=scorer_variant, coefficient_ii=coefficient_ii,
                                coefficient_latency=coefficient_latency,
                                sram_capacity_bytes=sram_capacity_bytes, session=cold,
                                address_offset=address_offset, frontend=frontend,
                                legacy_scorer_period_ns=legacy_scorer_period_ns,
                                legacy_burst_fifo_slots=legacy_burst_fifo_slots)

    pe_fs = round(pe_period_ns * 1_000_000)
    if not pe_fs:
        raise ValueError("PE period rounds to zero in the replay timebase")
    n, h = raw["history_pages"], raw["heads"]
    ii = 6 if scorer_variant == FAST_SCORER else 8
    norm = 2 * ceil_div(h * n, 16) + 12 * h
    topk = ceil_div(n, 4)
    planner = ceil_div(len(raw["k_pages"]) + 1, 4)
    unpack_lanes = 32
    config = np.array([
        lanes, pe_fs, MODES.index(mode), ii, ii + 1, coefficient_ii, coefficient_latency, 9,
        (raw["summary_address"] + address_offset) // BURST_BYTES,
        raw["summary_bytes"] // BURST_BYTES, raw["summary_stride"], n, h, raw["head_dim"],
        len(raw["k_pages"]), stat["logical_v_tiles"], norm, topk, planner,
        stat["gate_pe_cycles"] + stat["gate_budget_pe_cycles"], stat["gate_control_sram_bytes"],
        stat["gate_tokens"], 12 * h, unpack_lanes, raw["page_size"],
        *(frontend.config() if frontend is not None else (0,) * 13),
        round(legacy_scorer_period_ns * 1_000_000) if legacy_scorer_period_ns is not None else 0,
        legacy_burst_fifo_slots,
    ], dtype=np.int64)
    address_runs = (prepared.ranges + (address_offset, 0)) // BURST_BYTES
    out = session.run(config, address_runs.flatten(), prepared.tiles)
    if (out["rd"] * BURST_BYTES != raw["total_bytes"] or
            out["summary_rd"] * BURST_BYTES != raw["summary_bytes"] or
            out["k_rd"] * BURST_BYTES != raw["k_bytes"] or out["v_rd"] * BURST_BYTES != raw["v_bytes"] or
            out["peak_inflight_bursts"] > QUEUE_DEPTH or
            out["row_hits"] + out["act"] != out["rd"] or
            out["first_k_command_ns"] < out["topk_done_ns"] or
            (out["first_v_command_ns"] is not None and out["first_v_command_ns"] < out["gate_done_ns"]) or
            out["completion_ns"] + 1e-9 < out["dependency_lower_bound_ns"]):
        raise RuntimeError("replay violated address/work accounting, dependency, or queue limits")
    elapsed = out["completion_ns"]
    records = {key: raw[key] for key in ("cohort", "sample_id", "step", "layer", "group",
               "scope", "policy", "whole_step", "fallback", "history_pages", "history_tokens",
               "page_size", "head_dim", "heads", "summary_stride", "summary_bytes", "k_bytes",
               "v_bytes", "total_bytes", "qk_mac", "pv_mac", "scorer_terms", "fused_updates",
               "final_prefix", "answer_start", "cap_bytes") if key in raw}
    records.update(
        initial_prefix=len(raw["k_pages"]), mode=mode, lanes=lanes, pe_period_ns=pe_fs / 1_000_000,
        scorer_variant=scorer_variant, scorer_channels=scorer_channels, scorer_ii=ii,
        scorer_latency_pe_cycles=ii + 1, coefficient_ii=coefficient_ii,
        coefficient_latency_pe_cycles=coefficient_latency, sram_capacity_bytes=sram_capacity_bytes,
        feasible_sram=out["peak_sram_bytes"] <= sram_capacity_bytes,
        normalization_pe_cycles=norm, topk_pe_cycles=topk, planner_pe_cycles=planner,
        dram_command_util=out["command_ck"] * CK_NS / elapsed,
        dram_data_util=out["rd"] * 2 * CK_NS / elapsed,
        dram_outstanding_util=out["dram_outstanding_ns"] / elapsed,
        backend_active_util=(out["qk_busy_ns"] + out["pv_busy_ns"]) / elapsed,
        array_util=(raw["qk_mac"] + raw["pv_mac"]) * (pe_fs / 1_000_000) / (lanes * elapsed),
        scorer_active_util=out["scorer_busy_ns"] / elapsed,
        backend_qk_sram_stream_bytes_per_pe_cycle=2 * ceil_div(lanes, h),
        backend_pv_sram_stream_bytes_per_pe_cycle=2 * lanes,
        unpack_lanes=unpack_lanes, address_offset=address_offset,
        post_scorer_ns=out["completion_ns"] - out["score_done_ns"],
        post_scorer_compute_ns=(
            (norm + topk + planner + stat["gate_pe_cycles"] + stat["gate_budget_pe_cycles"]
             + 12 * h * bool(stat["logical_v_tiles"])) * (pe_fs / 1_000_000)
            + sum(out[name] for name in ("qk_busy_ns", "pv_busy_ns", "k_unpack_ns",
                                        "v_unpack_ns", "pv_weights_ns"))
        ),
        data_compute_overlap_ns=sum(out[name] for name in
                                    ("score_data_overlap_ns", "qk_data_overlap_ns", "pv_data_overlap_ns")),
        sram_port_contention_wait_ns=0,
        **stat, **out,
    )
    if frontend is None:
        for key in FRONTEND_METRICS:
            del records[key]
        if legacy_scorer_period_ns is not None:
            records.update(scorer_clock_period_ns=config[-2] / 1_000_000,
                           scorer_latency_pe_cycles=None, scorer_latency_cycles=ii + 1)
        if legacy_burst_fifo_slots:
            records.update(frontend_burst_fifo_slots=legacy_burst_fifo_slots,
                           peak_frontend_reserved_bytes=out["peak_frontend_reserved_bytes"])
    else:
        records.update(
            scorer_variant=frontend.profile,
            scorer_clock_period_ns=frontend.config()[0] / 1_000_000,
            scorer_ii=frontend.dot_ii_cycles,
            scorer_latency_pe_cycles=None,
            scorer_latency_cycles=frontend.dot_tail_cycles,
            coefficient_ii=frontend.coefficient_service_cycles,
            coefficient_latency_pe_cycles=None,
            coefficient_latency_cycles=frontend.coefficient_latency_cycles,
            frontend_record_interval_cycles=frontend.record_interval_cycles,
            frontend_record_to_dot_cycles=frontend.record_to_dot_cycles,
            frontend_record_to_result_cycles=frontend.record_to_result_cycles,
            frontend_record_release_cycles=frontend.record_release_cycles,
            frontend_record_slots=frontend.record_slots,
            frontend_ingress_bytes_per_cycle=frontend.ingress_bytes_per_cycle,
            frontend_ingress_cdc_cycles=frontend.ingress_cdc_cycles,
            frontend_result_cdc_pe_cycles=frontend.result_cdc_pe_cycles,
            frontend_burst_fifo_slots=frontend.burst_fifo_slots,
            sram_port_contention_wait_ns=out["frontend_ingress_queue_wait_ns"],
        )
    return records
