"""Round 12 narrow summary workpoints with the 112 B M candidate width."""
from .compact_summary import SUMMARY_BYTES
from .round11_workpoints import A, B, C, M, Workpoint, allocation
from .summary_decode import plan

N80 = Workpoint("N80", "mr80_score", 4)
N64 = Workpoint("N64", "mr64_score", 4)
CANDIDATES = (N80, N64)
OPTS = {p.name: p for p in (A, B, C, M, *CANDIDATES)}


def same_k(policy, ranks, history):
    if policy not in CANDIDATES:
        return allocation(policy, ranks, history)
    reference = allocation(M, ranks, history)
    return plan(ranks, history, summary_bytes=SUMMARY_BYTES[policy.method],
                budget_percent=policy.budget, v_reserve=policy.v_reserve,
                max_pages=reference.ids.shape[1])
