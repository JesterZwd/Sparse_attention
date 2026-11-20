"""External reads for the packed affine32 layout, per KV group and layer."""

from dataclasses import asdict, dataclass
from functools import lru_cache


SUMMARY_VECTORS = {"quest_softmax": 2, "quest_max": 2, "centroid8": 1}
ALIGNMENT = 16
INDEX_BYTES = 4
SINK_TOKENS = 32
LOCAL_TOKENS = 128


def aligned(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class KVRead:
    k_codes: int
    v_codes: int
    k_scale: int
    v_scale: int
    k_zero_point: int
    v_zero_point: int
    kv_alignment: int

    @property
    def total(self):
        return sum(asdict(self).values())


def kv_page(tokens, dim=128):
    if tokens < 1 or dim < 32 or dim % 32:
        raise ValueError("affine32 page requires positive tokens and 32-channel groups")
    codes, scales, zero_points = tokens * dim // 2, tokens * dim // 16, tokens * dim // 32
    padding = 2 * (sum(map(aligned, (codes, scales, zero_points))) - codes - scales - zero_points)
    return KVRead(codes, codes, scales, scales, zero_points, zero_points, padding)


def add_kv(left, right, multiplier=1):
    return KVRead(*(a + multiplier * b for a, b in zip(asdict(left).values(), asdict(right).values())))


@dataclass(frozen=True)
class PageHistory:
    tokens: int
    page_size: int
    pages: int
    tail_size: int
    protected: tuple[int, ...]
    expanded_tokens: int
    full: KVRead
    tail: KVRead
    dense: KVRead
    protected_kv: KVRead


@lru_cache(maxsize=4096)
def page_history(tokens, page_size, dim=128):
    if not 1 <= page_size <= 255 or tokens < 1:
        raise ValueError("valid lengths are stored as uint8, and history must be nonempty")
    pages = (tokens + page_size - 1) // page_size
    local_start = max(0, tokens - LOCAL_TOKENS)
    sink_end = min(SINK_TOKENS, tokens)
    protected = tuple(sorted(set(range((sink_end + page_size - 1) // page_size)) |
                             set(range(local_start // page_size, pages))))
    tail_size = tokens - (pages - 1) * page_size
    full, tail = kv_page(page_size, dim), kv_page(tail_size, dim)
    protected_kv = add_kv(tail, full, len(protected) - 1)
    requested = sink_end + tokens - max(local_start, sink_end)
    expanded = (len(protected) - 1) * page_size + tail_size - requested
    return PageHistory(tokens, page_size, pages, tail_size, protected, expanded,
                       full, tail, add_kv(tail, full, pages - 1), protected_kv)


@dataclass(frozen=True)
class SummaryRead:
    code_bytes: int
    scale_bytes: int
    length_bytes: int
    summary_alignment: int

    @property
    def total(self):
        return sum(asdict(self).values())


def summary_record(method, dim=128):
    vectors = SUMMARY_VECTORS[method]
    payload = vectors * (dim + 2) + 1
    return SummaryRead(vectors * dim, vectors * 2, 1, aligned(payload) - payload)


def read_ledger(history, method, dim=128, *, percent=None, page_count=None, resident_bytes=0):
    """Charge the complete summary scan, selected page indices, and K/V.

    The final page is protected. Every optional page therefore has the same
    payload. Fixed residency pins sealed records by page order in an equal
    SRAM slice per layer/KV group. The writable tail stays in external memory.
    """
    if (percent is None) == (page_count is None):
        raise ValueError("choose an integer byte percentage or an exact page count")
    if percent is not None and not 0 < percent <= 100:
        raise ValueError("byte percentage must be in (0,100]")
    if resident_bytes < 0:
        raise ValueError("resident capacity must be nonnegative")
    record = summary_record(method, dim)
    resident = min(history.pages - 1, resident_bytes // record.total)
    scans = history.pages - resident
    scan = scans * record.total
    min_pages = len(history.protected)
    minimum = scan + min_pages * INDEX_BYTES + history.protected_kv.total
    ceiling = history.dense.total * percent // 100 if percent is not None else None
    if percent is not None:
        available = (ceiling - minimum) // (history.full.total + INDEX_BYTES)
        page_count = min(history.pages, min_pages + available)
    feasible = min_pages <= page_count <= history.pages
    if feasible:
        selected = add_kv(history.protected_kv, history.full, page_count - min_pages)
        selected_reads = {**asdict(selected), "index_bytes": page_count * INDEX_BYTES}
        total = scan + sum(selected_reads.values())
        if ceiling is not None and total > ceiling:
            raise AssertionError("selected reads exceed external limit")
    else:
        page_count, selected_reads, total = None, {}, None
    return {
        "feasible": feasible, "selected_pages": page_count, "protected_pages": min_pages,
        "expanded_protection_tokens": history.expanded_tokens, "valid_tail_tokens": history.tail_size,
        "resident_records": resident, "summary_records_scanned": scans,
        "summary_code_bytes": scans * record.code_bytes, "summary_scale_bytes": scans * record.scale_bytes,
        "summary_length_bytes": scans * record.length_bytes,
        "summary_alignment": scans * record.summary_alignment,
        "summary_read_bytes": scan, "summary_update_write_bytes": record.total,
        "summary_prefill_materialization_bytes": record.total * history.pages,
        "kv_read_bytes": sum(v for k,v in selected_reads.items() if k != "index_bytes") if feasible else None,
        **selected_reads, "dense_read_bytes": history.dense.total, "read_ceiling_bytes": ceiling,
        "minimum_read_bytes": minimum, "total_read_bytes": total,
        "unused_budget_bytes": ceiling - total if feasible and ceiling is not None else None,
    }
