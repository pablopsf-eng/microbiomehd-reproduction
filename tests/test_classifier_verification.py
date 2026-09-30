"""Independent verification of the M4 model-tuning follow-up (2026-09-18):
``mbhd.classifier._tuned_pipeline`` + the ``pipeline_fn`` parameter on
``leave_one_group_out_auc_dynamic_features`` / ``permutation_null_auc_dynamic``.

Ground truth used:
- Nested-CV leakage: directly observed, via monkeypatching
  ``LogisticRegressionCV.fit``, exactly which rows reach the inner 5-fold CV
  that picks ``C`` for each outer LOFO fold. The held-out group's own row
  count must never appear.
- ``pipeline_fn`` default equivalence: two independent invocations (default
  omitted vs. ``pipeline_fn=_pipeline`` passed explicitly) must be
  byte-identical, checked here without reusing the existing regression test
  in ``tests/test_classifier.py``.

Also covers the M5 ``fold_transform`` hook's own leakage boundary (verification
cases T6/T7): a constant added to the held-out fold's own
features cannot change its AUC (a provable consequence of ``StandardScaler``
+ a linear classifier, not just a design intent); the hook is called with
exactly the training rows every fold; and -- the test that actually closes
the covariate label-leak ``mbhd.batch`` exists to prevent -- flipping the
held-out fold's own labels leaves that fold's ``predict_proba`` output
byte-identical under ``mbhd.batch.combat_for_fold(..., use_covariate=True)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from mbhd.batch import combat_for_fold
from mbhd.classifier import (
    SEED,
    _pipeline,
    _tuned_pipeline,
    leave_one_group_out_auc_dynamic_features,
)


def _synthetic_groups(n_groups: int, n_per_class: int) -> pd.Series:
    keys = [f"g{g}:s{i}" for g in range(n_groups) for i in range(2 * n_per_class)]
    values = [f"g{g}" for g in range(n_groups) for _ in range(2 * n_per_class)]
    return pd.Series(values, index=keys)


def test_tuned_pipeline_inner_cv_never_sees_held_out_fold_rows(monkeypatch) -> None:
    """Directly observes every row count passed to ``LogisticRegressionCV.fit``
    across a full LOFO run and asserts it always equals exactly that outer
    fold's *training* size (n_total - n_held_out), never n_total -- i.e. the
    inner 5-fold CV that picks C is structurally unable to see the held-out
    group's rows, since it never receives them."""
    n_groups, n_per_class = 5, 10
    groups = _synthetic_groups(n_groups, n_per_class)
    n_total = len(groups)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(
        {
            "f1": y.to_numpy(dtype=float) * 2.0 + rng.normal(size=n_total, scale=0.5),
            "f2": rng.normal(size=n_total),
        },
        index=groups.index,
    )

    seen_fit_sizes: list[int] = []
    original_fit = LogisticRegressionCV.fit

    def spy_fit(self, X, y_arg, *args, **kwargs):
        seen_fit_sizes.append(len(X))
        return original_fit(self, X, y_arg, *args, **kwargs)

    monkeypatch.setattr(LogisticRegressionCV, "fit", spy_fit)

    leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: ["f1", "f2"],
        pipeline_fn=_tuned_pipeline,
    )

    held_out_size = 2 * n_per_class
    expected_train_size = n_total - held_out_size
    assert len(seen_fit_sizes) == n_groups  # one fit() call per LOFO fold
    assert seen_fit_sizes == [expected_train_size] * n_groups
    assert all(size < n_total for size in seen_fit_sizes)


def test_pipeline_fn_default_matches_explicit_plain_pipeline() -> None:
    """Independent check (not reusing tests/test_classifier.py's own
    regression test) that omitting ``pipeline_fn`` gives byte-identical
    output to passing ``pipeline_fn=_pipeline`` explicitly."""
    n_groups, n_per_class = 4, 6
    groups = _synthetic_groups(n_groups, n_per_class)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    x = pd.DataFrame({"f": y.to_numpy(dtype=float)}, index=groups.index)

    default_result = leave_one_group_out_auc_dynamic_features(
        x, y, groups, seed=SEED, features_for_fold=lambda _g: ["f"]
    )
    explicit_result = leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: ["f"],
        pipeline_fn=_pipeline,
    )
    pd.testing.assert_frame_equal(default_result, explicit_result)


def _synthetic_dataset(n_groups: int, n_per_class: int, seed: int, n_features: int = 4):
    """A synthetic LOFO dataset with real per-fold group structure --
    ``x`` carries a genuine label signal on feature 0 plus a genuine,
    substantial per-group (batch) offset on every feature, so batch
    correction has something real to remove."""
    groups = _synthetic_groups(n_groups, n_per_class)
    n_total = len(groups)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "f0": y.to_numpy(dtype=float) * 1.5 + rng.normal(size=n_total, scale=0.6),
            **{f"f{i}": rng.normal(size=n_total) for i in range(1, n_features)},
        },
        index=groups.index,
    )
    group_offset = {g: rng.normal(scale=2.0, size=n_features) for g in groups.unique()}
    for g, offset in group_offset.items():
        mask = (groups == g).to_numpy()
        x.loc[mask, :] += offset
    return x, y, groups


def test_constant_shift_on_held_out_fold_features_never_changes_its_auc() -> None:
    """T6: with StandardScaler + a linear classifier, adding one constant
    vector to every held-out row's features shifts every test score by the
    same amount (w.c), so ranking -- and therefore AUC -- is unchanged.
    This is the concrete, checkable prediction that the M5 design
    makes about why test-side batch correction can't move AUC on its own."""
    x, y, groups = _synthetic_dataset(n_groups=5, n_per_class=12, seed=101)

    def add_constant_to_test_only(x_train, y_train, batch_train, x_test):
        del y_train, batch_train
        return x_train, x_test + np.array([3.0, -2.0, 1.5, 0.7])

    baseline = leave_one_group_out_auc_dynamic_features(
        x, y, groups, seed=SEED, features_for_fold=lambda _g: list(x.columns)
    )
    shifted = leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: list(x.columns),
        fold_transform=add_constant_to_test_only,
    )
    np.testing.assert_allclose(
        baseline["auc_roc"].to_numpy(), shifted["auc_roc"].to_numpy(), atol=1e-10
    )


def test_fold_transform_is_called_with_exactly_the_training_rows() -> None:
    """T7a: the hook must see len(y_train) == n_total - n_held_out every
    fold, and x_test must be exactly the held-out rows -- never the full
    matrix, never a mix."""
    n_groups, n_per_class = 4, 8
    x, y, groups = _synthetic_dataset(n_groups, n_per_class, seed=102)
    n_total = len(groups)
    held_out_size = 2 * n_per_class

    seen: list[tuple[int, int]] = []

    def spy(x_train, y_train, batch_train, x_test):
        assert len(y_train) == len(batch_train) == len(x_train)
        seen.append((len(y_train), len(x_test)))
        return x_train, x_test

    leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: list(x.columns),
        fold_transform=spy,
    )
    assert seen == [(n_total - held_out_size, held_out_size)] * n_groups


def test_flipping_held_out_labels_leaves_covariate_combat_predictions_unchanged() -> (
    None
):
    """T7c: the label-leak mbhd.batch.combat_new_batch exists to close --
    a held-out sample's own case/control label entering its own corrected
    feature value -- is closed through the fold_transform hook's own
    signature (it never receives y_test). Directly verified here: for each
    group in turn, flip *only that group's own* labels (every other
    group's labels held at their real values, so that group's own training
    set -- everyone else -- is byte-identical to the real run) and confirm
    that group's own fold_transform output is unchanged under
    mbhd.batch.combat_for_fold(use_covariate=True). Flipping every group's
    labels in one shot would also change every fold's *training* labels
    (since a fold's training set is every other group), contaminating the
    comparison with a real, expected training-side difference -- so each
    group is flipped in its own separate run, holding everyone else fixed."""
    x, y, groups = _synthetic_dataset(n_groups=6, n_per_class=15, seed=103)
    # leave_one_group_out_auc_dynamic_features iterates pd.unique(groups),
    # a deterministic order depending only on `groups` (unchanged across
    # every run below) -- so the i-th captured output always corresponds
    # to the i-th group in this same order, across all runs.
    group_order = list(pd.unique(groups.to_numpy()))

    def ordered_test_outputs(y_for_run: pd.Series) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []

        def spy_transform(x_train, y_train, batch_train, x_test):
            train_corrected, test_corrected = combat_for_fold(
                x_train, y_train, batch_train, x_test, use_covariate=True
            )
            outputs.append(test_corrected.copy())
            return train_corrected, test_corrected

        leave_one_group_out_auc_dynamic_features(
            x,
            y_for_run,
            groups,
            seed=SEED,
            features_for_fold=lambda _g: list(x.columns),
            fold_transform=spy_transform,
        )
        return outputs

    real_outputs = ordered_test_outputs(y)

    for i, target_group in enumerate(group_order):
        y_one_flipped = y.copy()
        mask = (groups == target_group).to_numpy()
        y_one_flipped.loc[mask] = 1 - y_one_flipped.loc[mask]
        # Only target_group's own training set (everyone else, whose
        # labels are untouched) is byte-identical to the real run -- every
        # OTHER fold's own training set now includes target_group with a
        # genuinely different label, so only position i is a valid check.
        flipped_outputs = ordered_test_outputs(y_one_flipped)
        np.testing.assert_array_equal(real_outputs[i], flipped_outputs[i])


def test_tuned_pipeline_and_fold_transform_compose_without_leaking(monkeypatch) -> None:
    """M5 follow-up (--tune combined with --batch-correct): no existing
    test exercised a non-default ``pipeline_fn`` and a non-default
    ``fold_transform`` in the same call. The composition is structurally
    trivial (``fold_transform`` runs first, its output feeds ``pipeline_fn``
    -- no shared state, src/mbhd/classifier.py's own per-fold loop), but
    this closes the gap directly: the tuned pipeline's inner 5-fold CV must
    still never see the held-out group's rows (the pre-existing M4
    guarantee) *and* the array reaching the pipeline's own first step
    (``StandardScaler``) must be exactly ``fold_transform``'s output, not
    the raw ``x`` -- checked by intercepting ``StandardScaler.fit`` itself
    (the earliest point inside the fitted ``Pipeline``), not by inspecting
    post-scaling values, since a constant offset added by a transform would
    otherwise be invisible after centering (``StandardScaler`` removes any
    location shift applied uniformly to every row) and prove nothing."""
    n_groups, n_per_class = 4, 8
    groups = _synthetic_groups(n_groups, n_per_class)
    n_total = len(groups)
    y = pd.Series(
        ([0] * n_per_class + [1] * n_per_class) * n_groups, index=groups.index
    )
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(
        {
            "f1": y.to_numpy(dtype=float) * 2.0 + rng.normal(size=n_total, scale=0.5),
            "f2": rng.normal(size=n_total),
        },
        index=groups.index,
    )
    held_out_size = 2 * n_per_class
    x_arr_full = x[["f1", "f2"]].to_numpy(dtype=float)
    groups_arr = groups.to_numpy()

    def marking_transform(x_train, y_train, batch_train, x_test):
        del y_train, batch_train
        return x_train * 3.0 + 7.0, x_test * 3.0 + 7.0

    seen_fit_sizes: list[int] = []
    seen_scaler_inputs: list[np.ndarray] = []
    original_cv_fit = LogisticRegressionCV.fit
    original_scaler_fit = StandardScaler.fit

    def spy_cv_fit(self, X, y_arg, *args, **kwargs):
        seen_fit_sizes.append(len(X))
        return original_cv_fit(self, X, y_arg, *args, **kwargs)

    def spy_scaler_fit(self, X, *args, **kwargs):
        seen_scaler_inputs.append(np.asarray(X).copy())
        return original_scaler_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(LogisticRegressionCV, "fit", spy_cv_fit)
    monkeypatch.setattr(StandardScaler, "fit", spy_scaler_fit)

    leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=lambda _g: ["f1", "f2"],
        pipeline_fn=_tuned_pipeline,
        fold_transform=marking_transform,
    )

    expected_train_size = n_total - held_out_size
    assert seen_fit_sizes == [expected_train_size] * n_groups
    assert len(seen_scaler_inputs) == n_groups
    for held_out_group, scaler_input in zip(
        pd.unique(groups_arr), seen_scaler_inputs, strict=True
    ):
        train_mask = groups_arr != held_out_group
        expected = x_arr_full[train_mask] * 3.0 + 7.0
        np.testing.assert_array_equal(scaler_input, expected)


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-q"]))
