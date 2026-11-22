"""V selection references for fixed candidate K and shared GQA reads."""
from math import ceil, isfinite

import torch

from .fixed_k_v import keep_mask, probabilities


HISTOGRAM_BINS = 64
LOGIT_BIN_WIDTH = .25


def gate_mask(state, kind, threshold=0., order=None, epsilon=.02):
    """Return [group, head, page, token] PV masks before the byte-budget check.

    `threshold` is tau for checked and the relative local peak for BLASST.
    `order` maps rank order to physical slots for the BLASST traversal.
    """
    if not 0 <= epsilon < 1:
        raise ValueError("discard allowance must be in [0,1)")
    valid = state.valid[:, None].expand_as(state.logits)
    if kind == "full":
        return valid
    logits = state.logits.masked_fill(~valid, -torch.inf)
    if kind == "blasst":
        if not 0 <= threshold < 1:
            raise ValueError("BLASST peak threshold must be in [0,1)")
        return keep_mask(logits, None, state.valid, state.protected, "blasst", threshold, order)
    if kind == "exact":
        if epsilon == 0:
            return valid
        return keep_mask(logits, probabilities(logits), state.valid, state.protected, "sparta", epsilon)
    if kind not in ("checked", "hist64"):
        raise ValueError(f"unknown V gate: {kind}")
    if epsilon == 0:
        return valid

    valid_tokens = valid.flatten(2)
    forced = state.protected[:, None, :, None] & valid
    optional = valid_tokens & ~forced.flatten(2)
    flat = logits.float().flatten(2)
    maximum = flat.amax(-1, keepdim=True)
    gap = torch.where(torch.isfinite(maximum), maximum, 0.) - flat
    mass = (-gap).exp()
    allowed = mass.sum(-1, keepdim=True) * epsilon

    if kind == "checked":
        if not isfinite(threshold) or threshold < 0:
            raise ValueError("checked logit gap must be finite and nonnegative")
        proposed = optional & (gap >= threshold)
        accepted = (mass * proposed).sum(-1, keepdim=True) <= allowed
        return (valid_tokens & ~(proposed & accepted)).reshape_as(logits)

    # SPARTA-style boundary-bin selection adapted to GQA, with FP32 masses.
    # Equal-width gap bins are saturated at the last bin. Protected tokens are
    # retained upfront, so only optional mass participates in the histogram.
    index = (gap / LOGIT_BIN_WIDTH).clamp(0, HISTOGRAM_BINS - 1).long()
    bins = mass.new_zeros((*mass.shape[:-1], HISTOGRAM_BINS))
    bins.scatter_add_(-1, index, mass * optional)
    # Retain the whole bin if dropping it and all colder bins costs too much.
    cold_tail = bins.flip(-1).cumsum(-1).flip(-1)
    keep_bin = cold_tail > allowed
    chosen = keep_bin.gather(-1, index) & optional
    return (chosen | forced.flatten(2)).reshape_as(logits)


def gate_cost(kind, tokens, heads, *, optional_tokens=None, pages=None, score_bytes=4):
    """Analytic group schedule for one gate invocation, in the LPDDR model CK.

    Four scores per CK, one histogram bank per head on the 4:1 working point,
    one bin per CK per four heads. Counts for score SRAM include only QK
    writes and gate reads. Weight generation for PV belongs to the PV path.
    Control excludes the common logits SRAM and includes the head and union
    masks. This model has no measured SRAM timing or implemented datapath.
    """
    if kind not in ("full", "exact", "hist64", "blasst", "checked"):
        raise ValueError(f"unknown V gate: {kind}")
    if optional_tokens is None:
        optional_tokens = tokens
    if pages is None:
        pages = ceil(tokens / 32)
    scores = tokens * heads
    optional = optional_tokens * heads
    issue_ck = ceil(scores / 4)
    union_ck = ceil(tokens / 4)
    head_mask_bytes = ceil(scores / 8)
    union_mask_bytes = ceil(tokens / 8)
    control = head_mask_bytes + union_mask_bytes
    result = dict(gate_ck=0, logits_sram_bytes=scores * score_bytes,
                  logits_sram_writes=scores, logits_sram_reads=0, control_sram_bytes=control,
                  head_mask_bytes=head_mask_bytes, union_mask_bytes=union_mask_bytes,
                  histogram_bytes=0, histogram_updates=0, histogram_bin_visits=0,
                  gate_exponentials=0, gate_comparisons=0, gate_reduce_adds=0,
                  mask_sram_reads=0, mask_sram_writes=scores + tokens)
    if kind == "full" or not scores:
        return result
    if kind == "blasst":
        result.update(gate_ck=ceil(pages * heads / 4) + union_ck,
                      gate_comparisons=2 * pages * heads, control_sram_bytes=control + 2 * score_bytes * heads)
        return result
    if kind == "checked":
        # One score pass computes the proposed mask and two per-head sums.
        result.update(gate_ck=issue_ck + union_ck + 3, logits_sram_reads=scores,
                      gate_exponentials=scores, gate_comparisons=optional + heads,
                      gate_reduce_adds=scores + optional, mask_sram_reads=scores,
                      mask_sram_writes=2 * scores + tokens,
                      control_sram_bytes=control + 3 * score_bytes * heads + ceil(heads / 8))
        return result
    if kind == "hist64":
        # Zero bank, accumulate bins, scan twice, then reread logits for IDs.
        bin_count = heads * HISTOGRAM_BINS
        histogram = bin_count * score_bytes
        result.update(gate_ck=2 * issue_ck + ceil(3 * bin_count / 4) + 3,
                      logits_sram_reads=2 * scores, gate_exponentials=scores,
                      gate_comparisons=optional + bin_count, gate_reduce_adds=scores + optional + 2 * bin_count,
                      histogram_bytes=histogram, histogram_updates=optional, histogram_bin_visits=2 * bin_count,
                      control_sram_bytes=control + histogram + 3 * score_bytes * heads)
        return result
    # Exact Top-p is a quality reference. The rough sort comparison count
    # assumes a comparison sort; it is independent of the histogram estimate.
    levels = max(0, (optional_tokens - 1).bit_length()) if optional_tokens else 0
    comparisons = optional * levels
    result.update(gate_ck=2 * issue_ck + ceil(comparisons / 4), logits_sram_reads=scores,
                  gate_exponentials=scores, gate_comparisons=comparisons,
                  gate_reduce_adds=scores + optional,
                  control_sram_bytes=control + optional * (score_bytes + 2))
    return result
