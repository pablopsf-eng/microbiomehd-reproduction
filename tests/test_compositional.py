"""Tests for mbhd.compositional (M4 CLR extension).

Fully synthetic: a
hand-computed known-answer example and two algebraic invariants (sum-to-zero,
scale invariance) for ``clr``, plus a known-replacement-value check and a
fail-loudly check for ``multiplicative_replacement``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbhd.compositional import clr, multiplicative_replacement


def test_clr_hand_computed_three_genus_example():
    """Counts [2, 4, 8] in one sample: geometric mean 4, expected CLR
    [log(0.5), log(1)=0, log(2)] -- written out on paper."""
    counts = pd.DataFrame({"S1": [2.0, 4.0, 8.0]}, index=["g__A", "g__B", "g__C"])
    result = clr(counts, pseudocount_fn=lambda x: x)  # no zeros present
    expected = np.array([np.log(0.5), 0.0, np.log(2.0)])
    np.testing.assert_allclose(result["S1"].to_numpy(), expected, atol=1e-12)


def test_clr_values_sum_to_zero_within_every_sample():
    data = pd.DataFrame(
        {"S1": [1.0, 5.0, 20.0], "S2": [3.0, 3.0, 3.0], "S3": [0.1, 100.0, 4.0]},
        index=["g__A", "g__B", "g__C"],
    )
    result = clr(data, pseudocount_fn=lambda x: x)
    np.testing.assert_allclose(
        result.sum(axis=0).to_numpy(), [0.0, 0.0, 0.0], atol=1e-10
    )


def test_clr_is_scale_invariant_per_sample():
    """Multiplying one sample's raw counts by a constant (a stand-in for a
    sequencing-depth difference) must leave its CLR values unchanged."""
    data = pd.DataFrame(
        {"S1": [2.0, 4.0, 8.0], "S2": [1.0, 1.0, 1.0]}, index=["g__A", "g__B", "g__C"]
    )
    scaled = data.copy()
    scaled["S1"] = scaled["S1"] * 1000.0

    result = clr(data, pseudocount_fn=lambda x: x)
    result_scaled = clr(scaled, pseudocount_fn=lambda x: x)
    np.testing.assert_allclose(
        result["S1"].to_numpy(), result_scaled["S1"].to_numpy(), atol=1e-9
    )


def test_multiplicative_replacement_known_value():
    """One zero cell in a two-genus, one-sample column: smallest nonzero is
    0.4, so delta = 0.2; the nonzero value 0.4 must shrink to keep the
    column's total (0.4) unchanged: shrunk = 0.4 * (0.4 - 0.2*1) / 0.4 = 0.2."""
    data = pd.DataFrame({"S1": [0.4, 0.0]}, index=["g__A", "g__B"])
    result = multiplicative_replacement(data)
    assert result.loc["g__B", "S1"] == pytest.approx(0.2)
    assert result.loc["g__A", "S1"] == pytest.approx(0.2)
    assert result["S1"].sum() == pytest.approx(0.4)


def test_multiplicative_replacement_preserves_nonzero_columns():
    data = pd.DataFrame({"S1": [0.3, 0.7]}, index=["g__A", "g__B"])
    result = multiplicative_replacement(data)
    pd.testing.assert_frame_equal(result, data.astype(float))


def test_multiplicative_replacement_raises_with_no_nonzero_values_anywhere():
    data = pd.DataFrame({"S1": [0.0, 0.0]}, index=["g__A", "g__B"])
    with pytest.raises(ValueError, match="no nonzero values"):
        multiplicative_replacement(data)


def test_multiplicative_replacement_raises_when_zeros_would_go_negative():
    """A column with almost all mass in one zero-heavy column and a tiny
    nonzero value elsewhere in the table can require shrinking below zero --
    must fail loudly, not silently clip or produce a negative abundance."""
    data = pd.DataFrame(
        {
            "S1": [1e-9, 0.0, 0.0, 0.0],  # total = 1e-9, 3 zero cells
            "S2": [0.25, 0.25, 0.25, 0.25],
        },
        index=["g__A", "g__B", "g__C", "g__D"],
    )
    with pytest.raises(ValueError, match="negative"):
        multiplicative_replacement(data)


def test_clr_end_to_end_with_multiplicative_replacement():
    """clr composed with the real (not identity) pseudocount_fn: still sums
    to zero per sample, and the exact-zero cell is not -inf."""
    data = pd.DataFrame(
        {"S1": [0.5, 0.5, 0.0], "S2": [0.2, 0.3, 0.5]}, index=["g__A", "g__B", "g__C"]
    )
    result = clr(data, pseudocount_fn=multiplicative_replacement)
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_allclose(result.sum(axis=0).to_numpy(), [0.0, 0.0], atol=1e-10)
