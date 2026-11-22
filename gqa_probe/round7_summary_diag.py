"""Same-count full-V comparison of compact summaries and Oracle page mass."""

import torch

from .fixed_k_v import gather_tokens, masked_output, ranked_pages
from .oracle_attention import grouped_output, torch_page_mass
from .vskip_summary import weights


def summary_diagnostics(q, key, value, history, record, scale, *, count,
                        evidence=None, answer_start=False, logits=None, dense_output=None):
    """Return one JSON-ready row per method and group for the common Dense query.

    Evidence is a collection of physical page IDs. Cached dense logits and output
    can be passed by the trajectory runner to reuse identical QK and PV results.
    Coverage uses the softmax over the entire history, including protected pages.
    """
    if not len(history.protected) <= count <= history.pages:
        raise ValueError("page count must cover protection and fit the history")
    evidence = sorted(set(evidence)) if evidence is not None else []
    if any(page < 0 or page >= history.pages for page in evidence):
        raise ValueError("evidence page ID is outside the history")

    q = q.float()
    if logits is None:
        logits = torch.bmm(q, key[0].float().transpose(1, 2)) * scale
    if dense_output is None:
        dense_output = grouped_output(logits, value)
    mass = torch_page_mass(logits, history.page_size)
    dense_norm = torch.linalg.vector_norm(dense_output, dim=-1)
    scores = {method: weights(q, record[method], method, scale) for method in ("q4", "mr25")}
    scores["oracle_a1"] = mass.sum(-1)

    rows = []
    offsets = torch.arange(history.page_size, device=q.device)
    evidence_ids = torch.tensor(evidence, dtype=torch.long, device=q.device)
    for method, score in scores.items():
        pages = ranked_pages(score, history)[:, :count].sort(dim=-1).values
        positions = pages[..., None] * history.page_size + offsets
        valid = positions < history.tokens
        safe = positions.clamp_max(history.tokens - 1)
        indices = safe.flatten(1)[:, None].expand(-1, q.shape[1], -1)
        selected_logits = logits.gather(-1, indices).unflatten(-1, (count, history.page_size))
        value_vectors = gather_tokens(value, safe).to(logits.dtype)
        selected_output = masked_output(selected_logits, value_vectors, valid[:, None])
        absolute = torch.linalg.vector_norm(selected_output - dense_output, dim=-1)
        relative = absolute / (dense_norm + 1e-12)
        coverage = mass.gather(1, pages[..., None].expand(-1, -1, q.shape[1])).sum(1)
        metrics = torch.stack((coverage, absolute, relative), dim=1).cpu().tolist()
        hits = torch.isin(pages, evidence_ids).sum(-1).cpu().tolist()
        for group, (chosen, hit, (head_coverage, abs_error, rel_error)) in enumerate(
                zip(pages.cpu().tolist(), hits, metrics)):
            rows.append(dict(
                method=method, group=group, budget_pages=count, pages=chosen,
                evidence_selected=hit, evidence_total=len(evidence),
                gold_page_recall=hit / len(evidence) if evidence else None,
                head_coverage=head_coverage, head_output_absolute_errors=abs_error,
                head_output_relative_errors=rel_error, answer_start=bool(answer_start),
            ))
    return rows
