"""Independent verification pass for mbhd.stats (M2 review).

Written by a separate verification pass, not the implementer: a different RNG
seed than tests/test_stats.py (which uses 42), a different sampling
distribution, and edge cases (p=0/p=1 exactly, single-element groups, very
lopsided group sizes) not already covered there. Calls scipy/statsmodels
directly as the oracle, exactly as tests/test_stats.py does -- this file does
not trust the existing suite, it re-derives agreement from scratch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from mbhd.stats import benjamini_hochberg, kruskal_wallis

# --------------------------------------------------------------------------
# kruskal_wallis: independent random-input agreement, seed=1337 (not 42).
# --------------------------------------------------------------------------


def test_kruskal_wallis_matches_scipy_seed_1337_zero_inflated_counts() -> None:
    rng = np.random.default_rng(seed=1337)
    n_checked = 0
    for _ in range(500):
        n_a = int(rng.integers(1, 50))
        n_b = int(rng.integers(1, 50))
        a = rng.choice([0, 0, 0, 1, 2, 3], size=n_a).astype(float)
        b = rng.choice([0, 0, 1, 2, 5, 10], size=n_b).astype(float)
        if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
            continue  # fully degenerate case, covered separately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = scipy_stats.kruskal(a, b)
        if np.isnan(ref.statistic) or np.isnan(ref.pvalue):
            continue  # scipy's own degenerate-input nan, not a disagreement
        h, p = kruskal_wallis(a, b)
        n_checked += 1
        assert h == pytest.approx(ref.statistic, abs=1e-8)
        assert p == pytest.approx(ref.pvalue, abs=1e-8)
    assert n_checked > 400  # sanity: the loop actually exercised real cases


def test_kruskal_wallis_matches_scipy_single_element_groups() -> None:
    """n=1 vs n=1 is not exercised by tests/test_stats.py's random sweep
    (which draws group sizes from [2, 30))."""
    rng = np.random.default_rng(seed=2024)
    checked = 0
    for _ in range(50):
        a = rng.integers(0, 5, size=1).astype(float)
        b = rng.integers(0, 5, size=1).astype(float)
        if a[0] == b[0]:
            continue
        ref = scipy_stats.kruskal(a, b)
        h, p = kruskal_wallis(a, b)
        checked += 1
        assert h == pytest.approx(ref.statistic, abs=1e-8)
        assert p == pytest.approx(ref.pvalue, abs=1e-8)
    assert checked > 0


def test_kruskal_wallis_matches_scipy_lopsided_group_sizes() -> None:
    """n=2 vs n=200: far more lopsided than the random sweep's [2, 30) range."""
    rng = np.random.default_rng(seed=99)
    a = rng.poisson(1.0, size=2).astype(float)
    b = rng.poisson(3.0, size=200).astype(float)
    ref = scipy_stats.kruskal(a, b)
    h, p = kruskal_wallis(a, b)
    assert h == pytest.approx(ref.statistic, abs=1e-8)
    assert p == pytest.approx(ref.pvalue, abs=1e-8)


# --------------------------------------------------------------------------
# benjamini_hochberg: independent random-input agreement, seed=777 (not 42),
# with exact 0.0 / 1.0 p-values deliberately injected.
# --------------------------------------------------------------------------


def test_benjamini_hochberg_matches_statsmodels_seed_777_with_boundary_pvalues() -> (
    None
):
    rng = np.random.default_rng(seed=777)
    for _ in range(500):
        m = int(rng.integers(1, 100))
        pvals = rng.uniform(0, 1, size=m)
        if rng.random() < 0.3:
            pvals[0] = 0.0
        if rng.random() < 0.3 and m > 1:
            pvals[-1] = 1.0
        q = benjamini_hochberg(pvals)
        _, ref_q, _, _ = multipletests(pvals, method="fdr_bh")
        assert q == pytest.approx(ref_q, abs=1e-9)


@pytest.mark.parametrize(
    "pvals",
    [
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.03],
    ],
)
def test_benjamini_hochberg_matches_statsmodels_on_boundary_vectors(
    pvals: list[float],
) -> None:
    q = benjamini_hochberg(pvals)
    _, ref_q, _, _ = multipletests(pvals, method="fdr_bh")
    assert q == pytest.approx(ref_q, abs=1e-12)


# --------------------------------------------------------------------------
# Claim: scipy.stats.kruskal is nan on heavily-tied/all-zero-variance input;
# mbhd.stats.kruskal_wallis is defined. Re-verified independently (same
# input as tests/test_stats.py, but also an all-identical-nonzero variant
# with unequal group sizes, which that file does not try).
# --------------------------------------------------------------------------


def test_scipy_kruskal_nan_vs_mbhd_defined_all_zero() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = scipy_stats.kruskal([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.isnan(ref.statistic)
    assert np.isnan(ref.pvalue)

    h, p = kruskal_wallis([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert h == 0.0
    assert p == 1.0


def test_scipy_kruskal_nan_vs_mbhd_defined_all_identical_nonzero_unequal_sizes() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = scipy_stats.kruskal([5.0, 5.0], [5.0, 5.0, 5.0])
    assert np.isnan(ref.statistic)
    assert np.isnan(ref.pvalue)

    h, p = kruskal_wallis([5.0, 5.0], [5.0, 5.0, 5.0])
    assert h == 0.0
    assert p == 1.0
