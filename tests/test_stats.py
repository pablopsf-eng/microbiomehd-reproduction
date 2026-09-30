"""Tests for mbhd.stats. Fully synthetic -- no data or network dependency.

Hand-computed values first (independent derivation in the docstrings, not a
restatement of the code under test), then agreement with scipy/statsmodels on
seeded random inputs -- two separate verification layers, so a failure
localises to one or the other.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from mbhd.stats import (
    UnsupportedGroupCountError,
    benjamini_hochberg,
    kruskal_wallis,
    signed_qvalues,
)

# --------------------------------------------------------------------------
# kruskal_wallis: hand-computed.
# --------------------------------------------------------------------------


def test_kruskal_wallis_no_ties_hand_computed() -> None:
    """A=[1,2,3], B=[4,5,6]: ranks 1..6, no ties, C=1.

    H = 12/(6*7) * (6**2/3 + 15**2/3) - 3*7 = 3.857142857142854 (hand
    algebra); p = erfc(sqrt(H/2)). Independently confirmed against
    scipy.stats.kruskal, which agrees to full float precision.
    """
    h, p = kruskal_wallis([1, 2, 3], [4, 5, 6])
    assert h == pytest.approx(3.857142857142854)
    assert p == pytest.approx(0.04953461343562682)


def test_kruskal_wallis_with_ties_hand_computed() -> None:
    """A=[1,1,2], B=[2,3,3]: three tied pairs -- the two 1s (rank 1.5 each),
    the two 2s, one from each group (rank 3.5 each), and the two 3s (rank
    5.5 each) -- so the tie correction is not 1.

    Ranks: A=[1.5, 1.5, 3.5], B=[3.5, 5.5, 5.5]. Ra=6.5, Rb=14.5.
    H_uncorrected = 12/42 * (6.5**2/3 + 14.5**2/3) - 21 = 3.047619047619044.
    Tie term = (2**3-2)*3 = 18 (three tie groups of size 2, not two --
    the 2/2 pair ties across groups too); C = 1 - 18/210 = 0.9142857142857143.
    H = H_uncorrected / C = 3.3333333333333295 (hand algebra, confirmed
    against scipy.stats.kruskal, which agrees to full float precision).
    """
    h, p = kruskal_wallis([1, 1, 2], [2, 3, 3])
    assert h == pytest.approx(3.3333333333333295)
    assert p == pytest.approx(0.06788915486182917)


def test_kruskal_wallis_fully_degenerate_returns_zero_not_nan() -> None:
    """All values in both groups identical (here, all zero): no rank
    variation exists to detect a difference, so H must be exactly 0 and p
    exactly 1 -- a defined result, per the project's "never fall back
    silently" rule for degenerate data.

    This is not a hypothetical: scipy.stats.kruskal computes this exact case
    as a literal 0/0 in its tie-correction denominator and returns
    ``nan`` for both statistic and p-value (verified directly below, and
    also observed on real, non-synthetic zero-inflated genus abundances
    in this project -- see kruskal_wallis's own docstring). A single such
    ``nan`` p-value would corrupt every other genus's q-value once passed
    through Benjamini-Hochberg (see test_benjamini_hochberg_rejects_nan).
    """
    h, p = kruskal_wallis([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert h == 0.0
    assert p == 1.0


def test_scipy_kruskal_is_not_robust_to_this_but_ours_is() -> None:
    """Documents *why* kruskal_wallis clamps H, rather than asserting it blind.

    Same input as the previous test: scipy.stats.kruskal returns nan for
    both statistic and p-value here (its own tie-correction divides by a
    zero denominator without guarding against it) -- this is not a made-up
    justification for the clamp in mbhd.stats.kruskal_wallis, it is the
    actual observed behaviour of the reference library on this input.
    """
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        result = scipy_stats.kruskal([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert math.isnan(result.statistic)
    assert math.isnan(result.pvalue)


def test_kruskal_wallis_rejects_wrong_group_count() -> None:
    with pytest.raises(UnsupportedGroupCountError):
        kruskal_wallis([1, 2, 3])
    with pytest.raises(UnsupportedGroupCountError):
        kruskal_wallis([1, 2], [3, 4], [5, 6])


def test_kruskal_wallis_rejects_empty_group() -> None:
    with pytest.raises(UnsupportedGroupCountError):
        kruskal_wallis([], [1, 2, 3])


# --------------------------------------------------------------------------
# kruskal_wallis: reference-library agreement on seeded random inputs.
# --------------------------------------------------------------------------


def test_kruskal_wallis_matches_scipy_on_random_inputs() -> None:
    """Seeded explicitly -- no global np.random.seed()."""
    rng = np.random.default_rng(seed=42)
    for _ in range(200):
        n_a, n_b = rng.integers(2, 30, size=2)
        # Zero-inflated-ish integers, similar in shape to real abundance data,
        # but never the fully-degenerate all-tied case (covered separately).
        a = rng.poisson(lam=2.0, size=n_a).astype(float)
        b = rng.poisson(lam=2.0, size=n_b).astype(float)
        if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
            continue  # degenerate case has its own dedicated test

        h, p = kruskal_wallis(a, b)
        ref = scipy_stats.kruskal(a, b)
        assert h == pytest.approx(ref.statistic, abs=1e-9)
        assert p == pytest.approx(ref.pvalue, abs=1e-9)


# --------------------------------------------------------------------------
# benjamini_hochberg: hand-computed.
# --------------------------------------------------------------------------


def test_benjamini_hochberg_hand_computed() -> None:
    """p = [0.01, 0.04, 0.03, 0.20], m=4.

    Sorted: 0.01(rank1), 0.03(rank2), 0.04(rank3), 0.20(rank4).
    Raw BH: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.05333..., 0.20*4/4=0.20.
    Enforce monotonicity from the largest rank down (running minimum):
    0.20, min(0.05333,0.20)=0.05333, min(0.06,0.05333)=0.05333,
    min(0.04,0.05333)=0.04.
    So q = [0.04, 0.05333..., 0.05333..., 0.20] in original order
    ([p=0.01, p=0.04, p=0.03, p=0.20]).
    """
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert q == pytest.approx([0.04, 0.05333333333333333, 0.05333333333333333, 0.20])


def test_benjamini_hochberg_empty_input() -> None:
    assert benjamini_hochberg([]).tolist() == []


def test_benjamini_hochberg_rejects_nan() -> None:
    """A single nan p-value must fail loudly, not silently corrupt every
    other q-value in the same call -- see kruskal_wallis's docstring for the
    real case (a heavily-tied genus abundance) this was found from: a nan
    sorts unpredictably, so statsmodels' fdr_bh silently returns nan for
    every q-value in the batch once one nan is present, not just its own.
    """
    with pytest.raises(ValueError, match="nan"):
        benjamini_hochberg([0.01, float("nan"), 0.03])


def test_statsmodels_fdr_bh_is_corrupted_by_a_single_nan() -> None:
    """Documents why benjamini_hochberg rejects nan outright, with the
    reference library's own observed behaviour, not an assumption."""
    _, qvals, _, _ = multipletests([0.01, float("nan"), 0.03], method="fdr_bh")
    assert np.isnan(qvals).all()


# --------------------------------------------------------------------------
# benjamini_hochberg: reference-library agreement on seeded random inputs.
# --------------------------------------------------------------------------


def test_benjamini_hochberg_matches_statsmodels_on_random_inputs() -> None:
    rng = np.random.default_rng(seed=42)
    for _ in range(200):
        m = rng.integers(1, 50)
        pvals = rng.uniform(0, 1, size=m)
        q = benjamini_hochberg(pvals)
        _, ref_q, _, _ = multipletests(pvals, method="fdr_bh")
        assert q == pytest.approx(ref_q, abs=1e-12)


# --------------------------------------------------------------------------
# signed_qvalues.
# --------------------------------------------------------------------------


def test_signed_qvalues_negates_when_controls_higher() -> None:
    q = signed_qvalues(
        q_values=[0.01, 0.02], control_means=[5.0, 1.0], case_means=[1.0, 5.0]
    )
    assert q.tolist() == pytest.approx([-0.01, 0.02])


def test_signed_qvalues_exact_tie_is_documented_as_positive() -> None:
    """Undefined by the authors' README -- this project's explicit, documented
    choice is positive."""
    q = signed_qvalues(q_values=[0.05], control_means=[2.0], case_means=[2.0])
    assert q.tolist() == pytest.approx([0.05])
