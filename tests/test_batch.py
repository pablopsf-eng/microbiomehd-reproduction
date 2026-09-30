"""Tests for mbhd.batch. Fully synthetic -- no data or network dependency.

Hand-computed/known-answer values first, then agreement with ``pycombat``
(this project's dev-only ComBat oracle, `[dependency-groups] dev`, never
imported by ``mbhd`` itself -- ``tests/test_no_reference_library_imports.py``
enforces this) on seeded random inputs, matching ``tests/test_stats.py``'s
established from-scratch-then-oracle pattern for a from-scratch algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycombat import Combat

from mbhd.batch import (
    ComBatFit,
    EBNotConvergedError,
    ZeroPooledVarianceError,
    center_per_batch,
    combat,
    combat_for_fold,
    combat_new_batch,
)


def _two_batch_shifted_data(rng, n_per_batch=150, n_genes=8, seed_shift=1.0):
    """Two batches, a known additive shift and multiplicative scale applied
    to batch 'b' only, on top of shared standard-normal noise."""
    n = 2 * n_per_batch
    batch = np.array(["a"] * n_per_batch + ["b"] * n_per_batch)
    base = rng.normal(size=(n, n_genes))
    shift = rng.normal(scale=2.0, size=n_genes) * seed_shift
    scale = np.abs(rng.normal(loc=1.5, scale=0.4, size=n_genes)) + 0.3
    x = base.copy()
    x[n_per_batch:] = x[n_per_batch:] * scale + shift
    return x, batch, shift, scale


def test_combat_no_ties_exact_recovery_without_empirical_bayes() -> None:
    """empirical_bayes=False: gamma/delta are the raw method-of-moments
    estimates, not shrunk -- after correction, batch b's own per-gene mean
    and std must match batch a's own, to numerical precision (both batches
    drawn from the same underlying distribution before the shift/scale was
    applied)."""
    rng = np.random.default_rng(42)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng)
    corrected, fit = combat(x, batch, empirical_bayes=False)

    mean_a = corrected[batch == "a"].mean(axis=0)
    mean_b = corrected[batch == "b"].mean(axis=0)
    std_a = corrected[batch == "a"].std(axis=0, ddof=1)
    std_b = corrected[batch == "b"].std(axis=0, ddof=1)
    np.testing.assert_allclose(mean_a, mean_b, atol=1e-9)
    np.testing.assert_allclose(std_a, std_b, atol=1e-9)
    assert fit.n_eb_iterations["a"] == 0
    assert fit.n_eb_iterations["b"] == 0


def test_combat_empirical_bayes_still_aligns_batch_means_reasonably() -> None:
    """With empirical_bayes=True (shrinkage), correction is not exact, but
    must still bring batch b's mean much closer to batch a's than before
    correction -- a real, checkable direction, not just "runs"."""
    rng = np.random.default_rng(43)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=200)
    pre_gap = np.abs(x[batch == "a"].mean(axis=0) - x[batch == "b"].mean(axis=0))

    corrected, fit = combat(x, batch, empirical_bayes=True)
    post_gap = np.abs(
        corrected[batch == "a"].mean(axis=0) - corrected[batch == "b"].mean(axis=0)
    )
    assert np.all(post_gap < 0.5 * pre_gap)
    assert fit.n_eb_iterations["a"] > 0
    assert fit.n_eb_iterations["b"] > 0


def test_combat_single_batch_is_the_identity_transform() -> None:
    """A single distinct batch is mathematically degenerate for the general
    EB machinery (every gene's within-batch variance estimate collapses to
    the same constant, n/(n-1), after standardization -- zero cross-gene
    spread to fit a scale prior from) -- handled as a documented identity,
    not run through the general path."""
    rng = np.random.default_rng(44)
    x = rng.normal(size=(20, 5))
    batch = np.array(["only"] * 20)
    corrected, fit = combat(x, batch)
    np.testing.assert_array_equal(corrected, x)
    assert np.all(fit.gamma == 0.0)
    assert np.all(fit.delta == 1.0)


def test_combat_is_invariant_to_row_permutation() -> None:
    rng = np.random.default_rng(45)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=60)
    perm = rng.permutation(x.shape[0])

    corrected, _ = combat(x, batch, empirical_bayes=True)
    corrected_perm, _ = combat(x[perm], batch[perm], empirical_bayes=True)
    np.testing.assert_allclose(corrected_perm, corrected[perm], atol=1e-9)


def test_combat_is_invariant_to_batch_relabelling() -> None:
    rng = np.random.default_rng(46)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=60)
    relabelled = np.where(batch == "a", "x", "y")

    corrected, _ = combat(x, batch, empirical_bayes=True)
    corrected_relabelled, _ = combat(x, relabelled, empirical_bayes=True)
    np.testing.assert_allclose(corrected_relabelled, corrected, atol=1e-9)


def test_combat_handles_zero_within_batch_variance_without_producing_nan() -> None:
    """A genus wholly absent from one batch (filled a literal constant for
    every sample in that batch -- mbhd.classifier.pooled_feature_matrix's
    own real convention) must not crash or silently produce NaN/inf: that
    (batch, gene) cell gets delta=1 (no rescale), is excluded from that
    batch's own scale-prior fit, and is counted in n_zero_variance_cells."""
    rng = np.random.default_rng(47)
    n_a, n_b, n_genes = 30, 25, 10
    batch = np.array(["a"] * n_a + ["b"] * n_b)
    x = rng.normal(size=(n_a + n_b, n_genes))
    x[n_a:, 2] = 0.0  # genus absent from batch b
    x[:n_a, 5] = 0.0  # genus absent from batch a

    corrected, fit = combat(x, batch, empirical_bayes=True)
    assert np.all(np.isfinite(corrected))
    assert fit.n_zero_variance_cells == 2
    assert fit.delta[1, 2] == 1.0  # batch b, gene 2 (its own absent column)
    assert fit.delta[0, 5] == 1.0  # batch a, gene 5
    assert np.isfinite(fit.gamma[1, 2])  # location still shrunk normally


def test_combat_raises_zero_pooled_variance_for_a_globally_constant_gene() -> None:
    rng = np.random.default_rng(48)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=30)
    x[:, 3] = 5.0  # constant across every row, every batch
    with pytest.raises(ZeroPooledVarianceError, match=r"\[3\]"):
        combat(x, batch)


def test_combat_raises_value_error_when_a_batch_has_fewer_than_two_samples() -> None:
    x = np.random.default_rng(49).normal(size=(10, 4))
    batch = np.array(["a"] * 9 + ["b"] * 1)
    with pytest.raises(ValueError, match=">=2 samples"):
        combat(x, batch)


def test_combat_needs_at_least_one_batch_column_but_two_distinct_values_is_fine() -> (
    None
):
    """Sanity: exactly 2 batches (the minimum for the general, non-identity
    path) does not raise."""
    rng = np.random.default_rng(50)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=20)
    combat(x, batch, empirical_bayes=True)  # must not raise


def test_shrink_batch_raises_eb_not_converged_when_max_iter_is_too_small() -> None:
    rng = np.random.default_rng(51)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=100)
    with pytest.raises(EBNotConvergedError, match="did not converge"):
        combat(x, batch, empirical_bayes=True, max_iter=1, tol=1e-12)


def test_combat_with_covariate_preserves_the_covariate_effect_direction() -> None:
    """A real biological effect correlated with the covariate must survive
    correction (not be regressed away along with the batch effect) --
    checked by confirming the sign/rough magnitude of the covariate-vs-gene
    association is preserved, not just that the function runs."""
    rng = np.random.default_rng(52)
    n = 300
    batch = np.array(["a"] * 150 + ["b"] * 150)
    y = rng.integers(0, 2, size=n).astype(float)
    x = rng.normal(size=(n, 4))
    x[:, 0] += 3.0 * y  # a real, strong covariate-associated effect
    x[batch == "b"] += rng.normal(scale=1.5, size=4)  # batch shift

    corrected, fit = combat(x, batch, covariate=y, empirical_bayes=True)
    assert fit.beta is not None
    # The fitted covariate coefficient for gene 0 should be close to the
    # true effect size (3.0), not washed out by the batch correction.
    assert 2.0 < fit.beta[0] < 4.0
    # And the corrected data itself should still show cases > controls for
    # gene 0, on average.
    assert corrected[y == 1, 0].mean() > corrected[y == 0, 0].mean()


def test_combat_new_batch_aligns_an_unseen_batch_onto_the_training_reference() -> None:
    rng = np.random.default_rng(53)
    x, batch, shift, scale = _two_batch_shifted_data(rng, n_per_batch=150)
    _corrected, fit = combat(x, batch, empirical_bayes=True)

    # A third, held-out batch: standard-normal noise with the same known
    # shift/scale style already applied to batch "b" -- combat_new_batch
    # should pull its mean much closer to the fitted reference (fit.alpha)
    # than it was pre-correction, a directional check, not an exact one.
    held_out = rng.normal(size=(60, x.shape[1])) * scale + shift
    pre_gap = np.abs(held_out.mean(axis=0) - fit.alpha)

    corrected_held_out = combat_new_batch(fit, held_out)
    post_gap = np.abs(corrected_held_out.mean(axis=0) - fit.alpha)
    # Aggregate (mean-across-genes) comparison, not per-gene: a gene whose
    # random shift happened to be small to begin with can't reliably shrink
    # further and may even tick up slightly from EB sampling noise -- the
    # real, checkable claim is that correction helps overall, not on every
    # single gene individually.
    assert post_gap.mean() < 0.3 * pre_gap.mean()


def test_combat_new_batch_requires_at_least_two_samples() -> None:
    rng = np.random.default_rng(54)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=30)
    _corrected, fit = combat(x, batch)
    with pytest.raises(ValueError, match="at least 2|>=2"):
        combat_new_batch(fit, x[:1])


def test_combat_new_batch_never_takes_a_label_argument() -> None:
    """The label-leak this function exists to close (a held-out sample's
    own case/control label entering its own corrected feature value) is
    closed through the API, not by discipline: combat_new_batch has no
    parameter through which a label could even be passed."""
    import inspect

    params = inspect.signature(combat_new_batch).parameters
    assert not any(
        name in ("y", "y_new", "label", "labels", "covariate") for name in params
    )


def test_center_per_batch_zeros_out_each_batch_own_mean() -> None:
    rng = np.random.default_rng(55)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=40)
    train_mask = np.zeros(x.shape[0], dtype=bool)
    train_mask[:60] = True  # arbitrary split, some of each batch
    x_train, batch_train = x[train_mask], batch[train_mask]
    x_test = x[~train_mask]

    corrected_train, corrected_test = center_per_batch(
        x_train, None, batch_train, x_test
    )
    for b in np.unique(batch_train):
        np.testing.assert_allclose(
            corrected_train[batch_train == b].mean(axis=0), 0.0, atol=1e-9
        )
    np.testing.assert_allclose(corrected_test.mean(axis=0), 0.0, atol=1e-9)


def test_combat_for_fold_matches_manual_combat_plus_combat_new_batch() -> None:
    rng = np.random.default_rng(56)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=80)
    train_mask = batch == "a"
    x_train, y_train, batch_train = (
        x[train_mask],
        np.zeros(train_mask.sum()),
        batch[train_mask],
    )
    # "a" is a single batch here on purpose -- combat_for_fold must still
    # work through the identity path for a single training batch.
    x_test = x[~train_mask]

    corrected_train, corrected_test = combat_for_fold(
        x_train, y_train, batch_train, x_test, use_covariate=False
    )
    manual_train, fit = combat(x_train, batch_train, covariate=None)
    manual_test = combat_new_batch(fit, x_test)
    np.testing.assert_array_equal(corrected_train, manual_train)
    np.testing.assert_array_equal(corrected_test, manual_test)


def test_combat_for_fold_use_covariate_true_passes_y_train_as_covariate() -> None:
    rng = np.random.default_rng(57)
    n = 200
    batch = np.array(["a"] * 100 + ["b"] * 100)
    y = rng.integers(0, 2, size=n).astype(float)
    x = rng.normal(size=(n, 5))
    x[batch == "b"] += rng.normal(scale=1.0, size=5)

    corrected_cov, _ = combat_for_fold(x, y, batch, x[:5], use_covariate=True)
    corrected_nocov, _ = combat_for_fold(x, y, batch, x[:5], use_covariate=False)
    assert not np.allclose(corrected_cov, corrected_nocov)


# ---------------------------------------------------------------------
# Dev-only oracle cross-check (pycombat) -- same role scipy/statsmodels
# play for mbhd.stats, never imported by mbhd itself.
# ---------------------------------------------------------------------


def test_combat_matches_pycombat_without_covariate() -> None:
    rng = np.random.default_rng(58)
    n_a, n_b, n_c, n_genes = 40, 35, 25, 12
    batch = np.array(["a"] * n_a + ["b"] * n_b + ["c"] * n_c)
    n = n_a + n_b + n_c
    x = rng.normal(size=(n, n_genes))
    shift = rng.normal(scale=1.5, size=(3, n_genes))
    scale = np.abs(rng.normal(loc=1.5, scale=0.5, size=(3, n_genes))) + 0.3
    for i, b in enumerate(["a", "b", "c"]):
        mask = batch == b
        x[mask] = x[mask] * scale[i] + shift[i]

    mine, _fit = combat(x, batch, empirical_bayes=True)
    oracle = Combat(mode="p", conv=1e-4).fit_transform(x, batch)
    np.testing.assert_allclose(mine, oracle, atol=1e-4)


def test_combat_matches_pycombat_with_covariate() -> None:
    rng = np.random.default_rng(59)
    n_a, n_b, n_genes = 60, 55, 10
    batch = np.array(["a"] * n_a + ["b"] * n_b)
    n = n_a + n_b
    x = rng.normal(size=(n, n_genes))
    x[batch == "b"] += rng.normal(scale=1.2, size=n_genes)
    y = rng.integers(0, 2, size=n).astype(float)
    x[:, 0] += 2.0 * y

    mine, _fit = combat(x, batch, covariate=y, empirical_bayes=True)
    oracle = Combat(mode="p", conv=1e-4).fit_transform(x, batch, X=y.reshape(-1, 1))
    np.testing.assert_allclose(mine, oracle, atol=1e-4)


def test_combat_fit_dataclass_is_frozen() -> None:
    rng = np.random.default_rng(60)
    x, batch, _shift, _scale = _two_batch_shifted_data(rng, n_per_batch=20)
    _corrected, fit = combat(x, batch)
    assert isinstance(fit, ComBatFit)
    with pytest.raises(Exception):  # noqa: B017 -- FrozenInstanceError, dataclasses-internal
        fit.alpha = np.zeros_like(fit.alpha)
