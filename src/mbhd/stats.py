"""From-scratch two-group Kruskal-Wallis and Benjamini-Hochberg FDR.

Deliberately does not import ``scipy`` or ``statsmodels`` -- those are
dev-only dependencies used in ``tests/test_stats.py`` to prove this module
agrees with them, never imported by ``mbhd`` itself
(``tests/test_no_reference_library_imports.py`` enforces the boundary).

Restricted to exactly two groups: every comparison in this project is a
case/control pair (``mbhd.labels.CASE_CONTROL``), so a general k-group
chi-square survival function would be speculative code this project does not
need -- ``kruskal_wallis`` raises rather than silently handling more.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class UnsupportedGroupCountError(ValueError):
    """Raised when ``kruskal_wallis`` isn't given exactly two non-empty groups."""


def kruskal_wallis(*groups: Sequence[float]) -> tuple[float, float]:
    """Two-group Kruskal-Wallis H test: average ranks, standard tie correction.

    ``H = (H_uncorrected) / C``, where ``H_uncorrected`` is the usual rank-sum
    statistic and ``C = 1 - sum(t_i**3 - t_i) / (N**3 - N)`` is the standard
    tie correction (``t_i`` = size of the i-th group of tied values, ``N`` =
    total observations). With exactly two groups, ``H`` is asymptotically
    chi-square distributed with 1 degree of freedom, which has the closed
    form survival function ``p = erfc(sqrt(H / 2))`` -- used directly here
    instead of a general chi-square implementation.

    **Degenerate/heavily-tied data.** When the data are heavily tied (e.g.
    zero-inflated relative abundances), floating-point cancellation in
    ``H_uncorrected / C`` can produce a tiny *negative* H (observed on real
    data: -1.96e-13 for one genus in ``par_scheperjans``) when the true value
    is exactly 0. ``scipy.stats.kruskal`` does not guard against this and
    returns ``nan`` for ``p`` in that case -- verified directly against the
    real downloaded ``par_scheperjans`` OTU table before implementing this; see
    ``tests/test_stats.py::test_scipy_kruskal_is_not_robust_to_this_but_ours_is``.
    A single ``nan`` p-value like that silently corrupts every other
    dataset's q-value once passed through Benjamini-Hochberg's sort-based
    procedure, not just its own -- so this function clamps ``H`` to be
    non-negative and always returns a defined, finite ``(H, p)``.

    When every value across both groups is tied to every other value (the
    fully degenerate case, e.g. both groups are all-zero), the tie
    correction's denominator ``N**3 - N`` still cancels, but ``H`` is 0 by
    construction (there is no rank variation to detect) -- this function
    returns ``(0.0, 1.0)`` rather than dividing zero by zero.

    Raises ``UnsupportedGroupCountError`` if not given exactly two groups, or
    if either group is empty -- every comparison in this project has exactly
    two groups (case, control), so a general k-group implementation would be
    speculative code this project does not need.
    """
    if len(groups) != 2:
        raise UnsupportedGroupCountError(
            f"kruskal_wallis supports exactly two groups, got {len(groups)}"
        )
    a = np.asarray(groups[0], dtype=float)
    b = np.asarray(groups[1], dtype=float)
    if a.size == 0 or b.size == 0:
        raise UnsupportedGroupCountError(
            f"kruskal_wallis requires a non-empty observation in both groups, "
            f"got sizes {a.size} and {b.size}"
        )

    all_values = np.concatenate([a, b])
    n_total = all_values.size

    ranks = _average_ranks(all_values)
    rank_sum_a = ranks[: a.size].sum()
    rank_sum_b = ranks[a.size :].sum()

    h_uncorrected = (12.0 / (n_total * (n_total + 1))) * (
        rank_sum_a**2 / a.size + rank_sum_b**2 / b.size
    ) - 3 * (n_total + 1)

    _, tie_sizes = np.unique(all_values, return_counts=True)
    tie_term = np.sum(tie_sizes**3 - tie_sizes)
    tie_denominator = n_total**3 - n_total
    correction = 1.0 - tie_term / tie_denominator if tie_denominator > 0 else 0.0

    h = 0.0 if correction <= 0 else max(h_uncorrected / correction, 0.0)
    p_value = math.erfc(math.sqrt(h / 2))
    return h, p_value


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` 1..N ascending, averaging ranks within tied groups."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    n = values.size
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        # positions i..j-1 (0-indexed) are ranks i+1..j (1-indexed); average them.
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (the standard step-up procedure).

    ``q_(i) = min_{k >= i} p_(k) * m / k``, computed via sort + reverse
    running-minimum + clip to [0, 1] -- matches
    ``statsmodels.stats.multitest.multipletests(method="fdr_bh")`` to
    floating-point precision (``tests/test_stats.py``).

    Raises ``ValueError`` on any ``nan`` input rather than propagating it:
    a single ``nan`` p-value, if allowed through, silently corrupts every
    other q-value in the same call once ``nan`` enters the sort order (see
    ``kruskal_wallis``'s docstring for the real case this was found from) --
    failing loudly here is cheaper than debugging a whole dataset's worth of
    ``nan`` q-values traced back to one degenerate genus.
    """
    p = np.asarray(p_values, dtype=float)
    if np.isnan(p).any():
        raise ValueError("benjamini_hochberg received a nan p-value")
    m = p.size
    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def signed_qvalues(
    q_values: Sequence[float],
    control_means: Sequence[float],
    case_means: Sequence[float],
) -> np.ndarray:
    """Apply the published sign convention to BH q-values.

    Per the authors' supp-files README: "Negative values indicate that the
    mean abundance in controls was higher than in cases." So a q-value is
    negated when ``control_mean > case_mean``, left positive otherwise.

    **Exact ties** (``control_mean == case_mean``) are not specified by the
    README and did not arise in any of the four M2 subset datasets --
    documented decision, not a discovery: this function leaves the q-value
    positive (as if case-associated) on an exact tie, an arbitrary but
    explicit choice made so the behaviour is defined rather than left to
    floating-point chance. If a future dataset (M3's all-30 appendix) hits
    this case, that is new evidence and should prompt revisiting the choice,
    not be silently absorbed by it.
    """
    q = np.asarray(q_values, dtype=float)
    control = np.asarray(control_means, dtype=float)
    case = np.asarray(case_means, dtype=float)
    sign = np.where(control > case, -1.0, 1.0)
    return q * sign
