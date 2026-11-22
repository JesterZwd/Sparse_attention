"""Replay staged, coalesced reads through a bank-concurrent LPDDR5 model."""

import ctypes
from functools import lru_cache
from hashlib import sha256
from operator import index
from pathlib import Path
import subprocess
import tempfile

import numpy as np


@lru_cache(maxsize=1)
def _engine():
    source = Path(__file__).resolve().parents[1] / "sim" / "lpddr5_banked.cpp"
    version = sha256(source.read_bytes()).hexdigest()[:16]
    target = Path(".cache") / "lpddr5_banked" / f"replay-{version}.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        with tempfile.TemporaryDirectory(dir=target.parent) as directory:
            product = Path(directory) / target.name
            subprocess.run(["g++", "-std=c++17", "-O3", "-shared", "-fPIC",
                            str(source), "-o", str(product)], check=True)
            # Publish the complete library atomically, allowing parallel runs.
            product.replace(target)
    library = ctypes.CDLL(str(target.resolve()))
    replay = library.replay_banked
    vector = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    replay.argtypes = [vector, vector, vector, vector, ctypes.c_int64,
                       ctypes.c_int64, ctypes.c_int64, vector]
    replay.restype = None
    return replay


def read_banked(phases, wait=None, descriptor=2, queue_depth=32):
    """Replay aligned (byte address, byte count) runs with barriers between phases.

    Each nonempty run consumes one descriptor. Descriptor launches are spaced
    by `descriptor` CK and can queue while DRAM services earlier runs. Requests
    fill a front-end window in address-trace order. The window tracks both
    waiting bursts and reads whose data have yet to return. The controller
    preserves FIFO within each bank, issues a ready RD before an equally ready
    PRE or ACT pair, and otherwise picks the oldest ready bank head. Phases
    drain before `wait` CK elapse and the next phase can issue; open rows survive.

    The Round 6 mapping uses one x16 channel, 32 B bursts, 16 banks, four bank
    groups, and 2 KiB rows. The model includes a shared command bus, column
    spacing, and activate/precharge constraints. Refresh, writes, WCK changes,
    and cross-group co-issue are outside this replay. `row_hits` counts reads
    that did not themselves require an ACT. `command_cycles` counts occupied
    command CK, including both ACT slots per activation.
    """
    descriptor, queue_depth = index(descriptor), index(queue_depth)
    if descriptor < 0 or queue_depth < 1:
        raise ValueError("descriptor must be nonnegative and queue depth must be positive")

    starts, counts, bounds = [], [], [0]
    for phase in phases:
        for address, count in phase:
            address, count = index(address), index(count)
            if address < 0 or count < 0 or address % 32 or count % 32:
                raise ValueError("runs must use nonnegative, 32-byte aligned addresses and sizes")
            if count:
                starts.append(address // 32)
                counts.append(count // 32)
        bounds.append(len(starts))
    phase_count = len(bounds) - 1
    waits = [0] * phase_count if wait is None else [index(ck) for ck in wait]
    if len(waits) != phase_count or any(ck < 0 for ck in waits):
        raise ValueError("each phase requires one nonnegative delay")

    output = np.zeros(4 * phase_count + 7, dtype=np.int64)
    vectors = [np.asarray(values, dtype=np.int64) for values in (starts, counts, bounds, waits)]
    _engine()(*vectors, phase_count, descriptor, queue_depth, output)
    start, command, read, end = output[:4 * phase_count].reshape(4, phase_count).tolist()
    bursts, hits, activations, precharges, peak, command_cycles, min_pre_gap = output[4 * phase_count:].tolist()
    if bursts != sum(counts) or hits + activations != bursts or peak > queue_depth:
        raise RuntimeError("DRAM replay failed request accounting")

    return dict(phase_start_ck=start, phase_end_ck=end,
                phase_first_command_ck=[None if ck < 0 else ck for ck in command],
                phase_first_read_ck=[None if ck < 0 else ck for ck in read],
                bursts=bursts, row_hits=hits, activations=activations, precharges=precharges,
                max_outstanding=peak, command_cycles=command_cycles,
                minimum_precharge_gap_ck=None if min_pre_gap < 0 else min_pre_gap)
