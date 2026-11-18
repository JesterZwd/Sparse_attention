"""Stable full-history probabilities and selected-token attention in float64."""

import numpy as np


def _logits(logits):
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or 0 in scores.shape or not np.all(np.isfinite(scores)):
        raise ValueError("logits must be finite nonempty [heads, valid_tokens]")
    return scores


def _weights(scores):
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def page_mass(logits, page_size=32):
    scores = _logits(logits)
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be positive")
    weights = _weights(scores)
    return np.add.reduceat(weights, np.arange(0, scores.shape[1], page_size), axis=1).T


def attention_output(logits, values, pages=None, page_size=32):
    scores = _logits(logits)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != scores.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError("values must be finite [valid_tokens, dimensions] with matching length")
    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be positive")
    if pages is not None:
        pages = list(pages)
        count = (scores.shape[1] + page_size - 1) // page_size
        if not pages or any(not isinstance(p, (int, np.integer)) or not 0 <= p < count for p in pages):
            raise ValueError("selected pages must be nonempty valid page IDs")
        mask = np.isin(np.arange(scores.shape[1]) // page_size, pages)
        scores, values = scores[:, mask], values[mask]
    return _weights(scores) @ values
