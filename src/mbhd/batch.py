"""Parametric empirical-Bayes batch-effect correction (ComBat: Johnson, Li &
Rankin 2007, *Biostatistics*), implemented from scratch. Verified against
``pycombat`` (dev-only dependency, ``tests/test_batch_verification.py``) --
never imported here; ``tests/test_no_reference_library_imports.py`` enforces
this the same way it already does for ``mbhd.stats``/scipy/statsmodels.

Every array here is **samples x genes** (rows = samples), matching
``mbhd.classifier.pooled_feature_matrix``'s ``X`` orientation exactly --
never genes x samples (the opposite of ``mbhd.compositional``'s own
genus x sample convention; batch correction runs downstream of the
transpose ``pooled_feature_matrix`` already does).

**Model.** Each gene ``g`` is regressed on a batch design (one-hot per
batch) plus an optional single covariate column: ``x_g = alpha_g +
covariate*beta_g + batch_effects + noise``. ``alpha_g`` (grand mean),
``beta_g`` (covariate coefficient), and ``sigma_g`` (pooled residual
standard deviation) are fit once by OLS over every row passed in.
Standardized residuals are then grouped by batch; each batch's own location
(``gamma``) and scale (``delta``) adjustment is empirical-Bayes shrunk
toward a prior estimated from **that batch's own genes** (never pooled
across batches -- this is why the empirical-Bayes step itself does not leak
information between batches; only the upstream OLS fit does, which is why
``combat`` must only ever see training-fold rows in the leak-free per-fold
usage, ``mbhd.classifier``'s ``fold_transform`` hook).

**Zero within-batch variance.** A genus wholly absent from one study is
filled ``0.0`` for every sample in that study by ``mbhd.classifier.
pooled_feature_matrix``'s own convention (after CLR). Confirmed on the real
archive: this affects 27.9% of all (folder,
genus) cells and appears in every one of the 27 real folders -- not a rare
corner case. For such a cell, this batch's own scale is left unadjusted
(``delta = 1``) and the gene is excluded from that batch's own cross-gene
scale-prior fit (an inverse-gamma method-of-moments fit degenerates if fed
literal zeros); its location is still empirical-Bayes shrunk normally,
since the batch mean of a constant column is still a well-defined number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Threshold below which a variance is treated as exactly zero. Real
#: zero-within-batch-variance cells (a genus filled literal ``0.0`` for
#: every sample in a batch) can still compute to a tiny nonzero float
#: (observed: ~1e-34) rather than exact ``0.0``, because NumPy's vectorized
#: ``.var(axis=0)`` over a full 2D array does not use the same summation
#: order as reducing one column in isolation -- confirmed directly: the
#: same mathematically-exact-zero column gives literal ``0.0`` when sliced
#: out first, but ~1e-34 when reduced as part of the full batched array.
#: 1e-10 sits ~24 orders of magnitude above that observed noise floor and
#: ~9-11 orders of magnitude below any real CLR-scale variance -- a wide,
#: safe margin either way, not a tuned/fragile threshold.
_ZERO_VARIANCE_EPSILON = 1e-10


class ZeroPooledVarianceError(ValueError):
    """A gene has zero variance across every row passed to ``combat`` (not
    just within one batch) -- no pooled scale can be defined for it at all.
    Names the offending column index(es); never silently divides by zero."""


class EBNotConvergedError(RuntimeError):
    """The empirical-Bayes fixed-point iteration for one batch did not
    converge within ``max_iter`` steps, or produced a non-finite value.
    Never returns a partial/non-converged estimate."""


@dataclass(frozen=True)
class ComBatFit:
    """The fitted reference a held-out batch is corrected onto
    (``combat_new_batch``), and the full record of what ``combat`` did.

    ``alpha``/``sigma``: per-gene grand mean / pooled residual std, shape
    ``(n_genes,)``. ``beta``: per-gene covariate coefficient, shape
    ``(n_genes,)``, or ``None`` if no covariate was given. ``batches``: the
    distinct batch labels seen during fit, in the order ``gamma``/``delta``
    are indexed by. ``gamma``/``delta``: per-batch empirical-Bayes location/
    scale adjustment, shape ``(n_batches, n_genes)`` (``delta`` is a
    standard deviation, not a variance). ``n_eb_iterations``: iterations to
    convergence per batch (0 if ``empirical_bayes=False`` or the
    single-batch identity case). ``n_zero_variance_cells``: total
    (batch, gene) cells handled via the zero-variance policy above.
    """

    alpha: np.ndarray
    beta: np.ndarray | None
    sigma: np.ndarray
    batches: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    n_eb_iterations: dict[object, int]
    n_zero_variance_cells: int


def _prior_scale_moments(delta_hat_sq: np.ndarray) -> tuple[float, float]:
    """Inverse-gamma prior hyperparameters (lambda, theta) fit by method of
    moments across genes, from one batch's own per-gene variance estimates
    (Johnson et al. 2007, section 2.2). Caller guarantees at least 2 values
    and that they are not all identical (nonzero cross-gene spread)."""
    v = delta_hat_sq.mean()
    s2 = delta_hat_sq.var(ddof=1)
    lam = (2 * s2 + v**2) / s2
    theta = (v * s2 + v**3) / s2
    return lam, theta


def _shrink_batch(
    z_batch: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat_sq: np.ndarray,
    valid: np.ndarray,
    tol: float,
    max_iter: int,
    batch_label: object,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Empirical-Bayes shrinkage of one batch's per-gene (gamma_hat,
    delta_hat_sq) toward a prior fit from that same batch's own genes.
    ``valid`` marks genes with nonzero within-batch variance; invalid genes
    are pinned at ``delta = 1`` throughout (never updated) and excluded
    from the scale-prior fit, but still get their location shrunk
    normally. Returns (gamma_star, delta_star, n_iterations)."""
    n = z_batch.shape[0]
    gamma_bar = gamma_hat.mean()
    tau_sq = gamma_hat.var(ddof=1)
    if valid.sum() < 2 or delta_hat_sq[valid].var(ddof=1) <= _ZERO_VARIANCE_EPSILON:
        raise EBNotConvergedError(
            f"batch {batch_label!r}: fewer than 2 genes with distinct "
            "nonzero within-batch variance -- cannot fit a scale prior"
        )
    lam_bar, theta_bar = _prior_scale_moments(delta_hat_sq[valid])

    gamma = gamma_hat.copy()
    delta_sq = np.where(valid, delta_hat_sq, 1.0)
    for n_iter in range(1, max_iter + 1):
        new_gamma = (tau_sq * n * gamma_hat + delta_sq * gamma_bar) / (
            tau_sq * n + delta_sq
        )
        resid_sq_sum = np.sum((z_batch - new_gamma[np.newaxis, :]) ** 2, axis=0)
        new_delta_sq = (theta_bar + 0.5 * resid_sq_sum) / (n / 2.0 + lam_bar - 1)
        new_delta_sq = np.where(valid, new_delta_sq, 1.0)

        if not (np.all(np.isfinite(new_gamma)) and np.all(np.isfinite(new_delta_sq))):
            raise EBNotConvergedError(
                f"batch {batch_label!r}: EB iteration produced a non-finite value"
            )

        change = max(
            float(np.max(np.abs(new_gamma - gamma) / np.abs(gamma))),
            float(
                np.max(
                    np.abs(new_delta_sq[valid] - delta_sq[valid])
                    / np.abs(delta_sq[valid])
                )
            ),
        )
        gamma, delta_sq = new_gamma, new_delta_sq
        if change <= tol:
            return gamma, np.sqrt(delta_sq), n_iter

    raise EBNotConvergedError(
        f"batch {batch_label!r}: EB shrinkage did not converge within "
        f"{max_iter} iterations (last change={change:.2e}, tol={tol:.2e})"
    )


def center_per_batch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplest batch-correction lever: subtract each batch's own per-gene
    mean. ``x_test`` (the held-out batch) uses only its own rows -- leak-free
    by construction, zero cross-batch information sharing anywhere.
    ``y_train`` is unused (unsupervised); accepted only so this matches the
    ``FoldTransform`` signature every hook in this module shares."""
    del y_train
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    batch_train = np.asarray(batch_train)

    corrected_train = x_train.copy()
    for b in np.unique(batch_train):
        mask = batch_train == b
        corrected_train[mask] -= x_train[mask].mean(axis=0)

    corrected_test = x_test - x_test.mean(axis=0)
    return corrected_train, corrected_test


def combat(
    x: np.ndarray,
    batch: np.ndarray,
    covariate: np.ndarray | None = None,
    *,
    empirical_bayes: bool = True,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> tuple[np.ndarray, ComBatFit]:
    """Parametric ComBat on ``x`` (samples x genes). ``covariate``, if
    given, is one column (e.g. case/control 0/1) whose effect is preserved
    rather than removed alongside the batch effect (standard ComBat usage,
    Johnson et al. 2007 section 2.1) -- no multi-column design matrix,
    since no caller in this project needs more than one (YAGNI).

    A single distinct batch is a documented identity transform, not run
    through the general machinery below: with one batch, every gene's
    within-batch variance estimate collapses to the exact same constant
    (``n/(n-1)`` after standardization, algebraically, regardless of the
    data), giving zero cross-gene spread to fit a scale prior from --
    mathematically degenerate, not a bug to work around. ``pycombat``
    (this project's dev-only oracle) instead hard-requires >=2 batches;
    this is a deliberate, documented divergence, not an oversight.

    Raises ``ValueError`` if any batch has fewer than 2 samples (no
    within-batch variance is definable), ``ZeroPooledVarianceError`` if a
    gene has zero variance across every row (not just one batch), and
    ``EBNotConvergedError`` if empirical-Bayes shrinkage fails to converge
    for any batch.
    """
    x = np.asarray(x, dtype=float)
    batch = np.asarray(batch)
    n_samples, n_genes = x.shape
    batches = np.unique(batch)

    n_per_batch = np.array([(batch == b).sum() for b in batches])
    if np.any(n_per_batch < 2):
        sizes = dict(zip(batches.tolist(), n_per_batch.tolist(), strict=True))
        raise ValueError(
            f"every batch needs >=2 samples for a within-batch variance "
            f"estimate; batch sizes: {sizes}"
        )

    if batches.size == 1:
        return x.copy(), ComBatFit(
            alpha=x.mean(axis=0),
            beta=None,
            sigma=x.std(axis=0, ddof=0),
            batches=batches,
            gamma=np.zeros((1, n_genes)),
            delta=np.ones((1, n_genes)),
            n_eb_iterations={batches[0]: 0},
            n_zero_variance_cells=0,
        )

    has_covariate = covariate is not None
    design_blocks = [(batch == b).astype(float) for b in batches]
    if has_covariate:
        covariate_col = np.asarray(covariate, dtype=float).reshape(-1, 1)
        design = np.column_stack([*design_blocks, covariate_col])
    else:
        covariate_col = None
        design = np.column_stack(design_blocks)

    beta_hat, *_ = np.linalg.lstsq(design, x, rcond=None)
    alpha = (n_per_batch / n_samples) @ beta_hat[: batches.size, :]
    beta = beta_hat[batches.size :, :].reshape(-1) if has_covariate else None

    fitted = design @ beta_hat
    residual_sq_mean = np.mean((x - fitted) ** 2, axis=0)
    zero_pooled = residual_sq_mean <= _ZERO_VARIANCE_EPSILON
    if np.any(zero_pooled):
        raise ZeroPooledVarianceError(
            f"gene(s) at column index {np.flatnonzero(zero_pooled).tolist()} "
            "have zero variance across ALL rows -- cannot define a pooled scale"
        )
    sigma = np.sqrt(residual_sq_mean)

    covariate_term = covariate_col @ beta.reshape(1, -1) if has_covariate else 0.0
    z = (x - alpha[np.newaxis, :] - covariate_term) / sigma[np.newaxis, :]

    gamma = np.empty((batches.size, n_genes))
    delta = np.empty((batches.size, n_genes))
    n_iterations: dict[object, int] = {}
    n_zero_variance_cells = 0
    corrected = np.empty_like(x)

    for i, b in enumerate(batches):
        mask = batch == b
        z_batch = z[mask]
        gamma_hat = z_batch.mean(axis=0)
        delta_hat_sq = z_batch.var(axis=0, ddof=1)
        valid = delta_hat_sq > _ZERO_VARIANCE_EPSILON
        n_zero_variance_cells += int((~valid).sum())

        if empirical_bayes:
            gamma_star, delta_star, n_iter = _shrink_batch(
                z_batch, gamma_hat, delta_hat_sq, valid, tol, max_iter, b
            )
        else:
            gamma_star = gamma_hat
            delta_star = np.sqrt(np.where(valid, delta_hat_sq, 1.0))
            n_iter = 0
        gamma[i], delta[i] = gamma_star, delta_star
        n_iterations[b] = n_iter

        z_adjusted = (z_batch - gamma_star[np.newaxis, :]) / delta_star[np.newaxis, :]
        batch_corrected = z_adjusted * sigma[np.newaxis, :] + alpha[np.newaxis, :]
        if has_covariate:
            batch_corrected = batch_corrected + covariate_col[mask] @ beta.reshape(
                1, -1
            )
        corrected[mask] = batch_corrected

    fit = ComBatFit(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        batches=batches,
        gamma=gamma,
        delta=delta,
        n_eb_iterations=n_iterations,
        n_zero_variance_cells=n_zero_variance_cells,
    )
    return corrected, fit


def combat_new_batch(
    fit: ComBatFit,
    x_new: np.ndarray,
    *,
    empirical_bayes: bool = True,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> np.ndarray:
    """Corrects one unseen batch onto ``fit``'s training reference.

    Standardizes with the **training** ``alpha``/``sigma`` only -- no
    covariate term, since the new batch's own labels are never used, which
    is what makes this safe to call on a held-out LOFO fold (the label-leak
    path a naive "just include the test rows in ``combat``'s covariate
    fit" would open, closed here by construction, not by discipline). Then
    estimates this new batch's own gamma_hat/delta_hat and its own
    cross-gene prior from its own genes -- the same self-referential
    per-batch-prior pattern every batch in ``combat`` already uses, just
    for one more batch that was not part of the original fit -- and shrinks
    accordingly. A derivation for this project's specific train/holdout
    split, not a port of an existing "reference-batch" implementation
    (e.g. sva's ``ref.batch`` or neuroCombat's
    ``neuroCombatFromTraining``) -- documented as such, not asserted to
    match either without having checked.

    Raises ``ValueError`` if ``x_new`` has fewer than 2 samples (no
    within-batch variance is definable).
    """
    x_new = np.asarray(x_new, dtype=float)
    if x_new.shape[0] < 2:
        raise ValueError(
            f"combat_new_batch needs >=2 samples in the new batch, got {x_new.shape[0]}"
        )

    z_new = (x_new - fit.alpha[np.newaxis, :]) / fit.sigma[np.newaxis, :]
    gamma_hat = z_new.mean(axis=0)
    delta_hat_sq = z_new.var(axis=0, ddof=1)
    valid = delta_hat_sq > _ZERO_VARIANCE_EPSILON

    if empirical_bayes:
        gamma_star, delta_star, _ = _shrink_batch(
            z_new, gamma_hat, delta_hat_sq, valid, tol, max_iter, "<new batch>"
        )
    else:
        gamma_star = gamma_hat
        delta_star = np.sqrt(np.where(valid, delta_hat_sq, 1.0))

    z_adjusted = (z_new - gamma_star[np.newaxis, :]) / delta_star[np.newaxis, :]
    return z_adjusted * fit.sigma[np.newaxis, :] + fit.alpha[np.newaxis, :]


def combat_for_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_train: np.ndarray,
    x_test: np.ndarray,
    *,
    use_covariate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """``combat()`` on the training rows (``y_train`` as the covariate iff
    ``use_covariate``) followed by ``combat_new_batch()`` on ``x_test`` --
    the ``FoldTransform`` used via ``functools.partial(combat_for_fold,
    use_covariate=...)`` from ``scripts/classifier.py``. Matches
    ``center_per_batch``'s signature exactly so both can be passed to
    ``mbhd.classifier.leave_one_group_out_auc_dynamic_features``/
    ``permutation_null_auc_dynamic``'s ``fold_transform`` parameter
    interchangeably."""
    covariate = np.asarray(y_train, dtype=float) if use_covariate else None
    corrected_train, fit = combat(x_train, batch_train, covariate=covariate)
    corrected_test = combat_new_batch(fit, x_test)
    return corrected_train, corrected_test
