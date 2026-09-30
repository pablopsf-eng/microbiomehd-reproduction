"""Centered log-ratio (CLR) transform for a genus x sample abundance table.

Microbiome relative abundances are compositional, not Euclidean.
This module is the M4 CLR extension: it operates on the abundance matrix
itself, never on the hypothesis test -- ``mbhd.stats`` (Kruskal-Wallis +
Benjamini-Hochberg) is unchanged and is reused as-is on whichever matrix it
is given.

**Orientation, matching ``mbhd.abundance``:** every table here is
genera x samples (rows = genus lineage, columns = sample ID). CLR's mean is
taken **per sample** (down each column, across that sample's own genera),
not per genus.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def multiplicative_replacement(relative_abundance: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative simple zero-replacement (Martin-Fernandez et al.).

    **Decided (2026-09-18, Pablo):** every exact ``0.0`` cell is replaced with
    ``delta = 0.5 * (the smallest nonzero value anywhere in this table)`` --
    one ``delta`` for the whole table, computed once, not per sample or per
    genus. Each sample's (column's) *nonzero* values are then shrunk by a
    constant factor so the column's own total is unchanged: the zeros are
    filled with real mass rather than an unaccounted-for constant being
    added, which is what the compositional-data literature's "multiplicative"
    convention means, as opposed to a flat additive pseudocount (rejected --
    a scale-blind additive constant is the wrong choice across datasets that
    already vary by orders of magnitude).

    Raises ``ValueError`` naming the column if that column has so many zero
    cells, relative to its own total, that shrinking its nonzero values by
    the required factor would make them negative -- a real data condition,
    not a bug, and must fail loudly rather than silently produce a negative
    "relative abundance".
    """
    values = relative_abundance.to_numpy(dtype=float)
    nonzero = values[values > 0]
    if nonzero.size == 0:
        raise ValueError(
            "relative_abundance has no nonzero values anywhere -- cannot "
            "derive a replacement delta"
        )
    delta = 0.5 * nonzero.min()

    result = relative_abundance.astype(float).copy()
    for column in result.columns:
        col = result[column]
        zero_mask = col == 0.0
        n_zero = int(zero_mask.sum())
        if n_zero == 0:
            continue
        total = col.sum()
        scale = (total - delta * n_zero) / total
        if scale <= 0:
            raise ValueError(
                f"column {column!r}: {n_zero} zero cell(s) at replacement "
                f"value {delta!r} would exceed the column's own total "
                f"{total!r} -- multiplicative replacement would produce a "
                "negative value"
            )
        result.loc[zero_mask, column] = delta
        result.loc[~zero_mask, column] = col.loc[~zero_mask] * scale
    return result


def clr(
    relative_abundance: pd.DataFrame,
    pseudocount_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Centered log-ratio transform, one sample (column) at a time.

    ``clr(x)_g = log(x_g) - mean_g'(log(x_g'))`` -- the per-sample geometric
    mean is subtracted from every genus's log-abundance in that same sample.
    ``pseudocount_fn`` is applied first (e.g. ``multiplicative_replacement``)
    since ``log(0)`` is undefined; it is a required argument, not a hidden
    default, because M4's zero-replacement convention is a deliberate,
    recorded decision, not one this function should make silently on a caller's behalf.

    Values can be negative (unlike a relative abundance) and, within each
    sample, sum to (numerically) zero by construction -- both are expected,
    not bugs. Kruskal-Wallis (``mbhd.stats.kruskal_wallis``) makes no
    Euclidean-geometry assumption and is a valid test on these values exactly
    as it is on raw relative abundances.
    """
    replaced = pseudocount_fn(relative_abundance)
    log_values = np.log(replaced)
    return log_values - log_values.mean(axis=0)
