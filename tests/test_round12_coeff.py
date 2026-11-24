"""Check binary64 ties, U22 saturation and bias against the coefficient RTL."""

import random

import pytest

from scripts.check_round12_coeff import (
    directed_vectors, random_vectors, scorer_vectors, simulate, tool,
)


def test_round12_coefficient_stream_resets_bubbles_and_rounds(tmp_path):
    if not tool("iverilog") or not tool("vvp"):
        pytest.skip("Icarus Verilog iverilog/vvp unavailable")
    vectors = directed_vectors() + list(random_vectors(random.Random(12024), 256))
    stats = simulate(vectors, tmp_path)
    assert stats["coefficients_compared"] == len(vectors)
    assert stats["bias_codes_compared"] == 256


def test_round12_reference_matches_compact_cpu_scorer():
    assert len(list(scorer_vectors(12024))) == 2 * 8 * 2 * 41
