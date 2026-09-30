"""Pooled, leave-one-folder-out shared-response classifier (M4).

Does a linear classifier trained on the shared-response genus signature
generalize to an entirely unseen study? Reuses the exact building blocks
``mbhd.differential.differential_abundance`` already uses to read one
study's data (``mbhd.abundance.analysis_samples`` / ``filter_table`` /
``to_genus_abundance``, ``mbhd.labels.resolved_case_control``) -- this module
adds only the new step of turning those per-study tables into one pooled
feature matrix and running scikit-learn's leave-one-group-out CV over it.

**Grouping is by ``DatasetInfo.folder``, never by ``dataset_id``.**
``cdi_schubert``/``noncdi_schubert`` and ``nash_zhu``/``ob_zhu`` share one
physical folder and can share the literal same control samples
-- leaving one dataset ID out while keeping
its folder-mate in the training fold would leak those samples across the
train/test boundary. Grouping by folder collapses each pair into one fold and
closes this without per-pair special-casing.

**A genus absent from a study's own surviving genus set is filled ``0.0``**
(a real floor-of-detection interpretation, not a missing-data placeholder),
so every fold sees the same fixed feature columns.

``scikit-learn`` is a deliberate, named departure from ``mbhd.stats``'s
from-scratch convention: there is no published number to reproduce
bit-for-bit here, so hand-rolling logistic regression would trade
verification benefit for real implementation risk.

**Two LOFO variants.** ``leave_one_group_out_auc`` takes one fixed feature
matrix for every fold -- correct when the feature *columns* were chosen
independently of any group's labels (e.g. "every genus ever reproduced"),
but a feature-selection leakage risk when they were not (e.g. a genus list
chosen by significance across all groups, which is what a fixed,
globally-selected shared-response pool does -- found during review,
`scripts/classifier.py`'s own docstring has the full account).
``leave_one_group_out_auc_dynamic_features`` closes that path by asking the
caller to recompute the feature list per fold from training data only.

**Two model variants.** ``_pipeline`` (plain ``LogisticRegression``, ``C=1.0``)
is the original M4 model. ``_tuned_pipeline`` (``class_weight="balanced"``,
``C`` chosen by nested cross-validation via ``LogisticRegressionCV``) is a
same-day follow-up Pablo asked for after seeing the plain model's modest
LOFO score. Both ``leave_one_group_out_auc_dynamic_features`` and
``permutation_null_auc_dynamic`` take an optional ``pipeline_fn`` to select
between them; default preserves the original plain-model behaviour exactly.

**M5: an optional per-fold ``fold_transform`` hook** (``mbhd.batch``'s
batch-effect correction functions) applied to each fold's already-
column-selected feature matrix, immediately before the pipeline is fit --
after feature *selection* (``features_for_fold``), before feature
*scaling* (``StandardScaler``, inside ``pipeline_fn``'s own ``Pipeline``).
The hook receives ``(x_train, y_train, batch_train, x_test)`` and must
return ``(x_train_corrected, x_test_corrected)`` -- it is never handed the
held-out group's own labels or identity beyond its own ``x_test`` rows, so
a label-leak on the held-out fold (``mbhd.batch``'s own module docstring
has the full account of why this matters for covariate-preserving ComBat)
cannot be expressed through this signature. Default ``None`` preserves
today's behaviour byte-for-byte -- regression-tested.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mbhd.abundance import analysis_samples, filter_table, to_genus_abundance
from mbhd.datasets import (
    DatasetInfo,
    entry_for,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.differential import NoSurvivingSamplesError
from mbhd.io import DISEASE_STATE_COLUMN, encoding_for, load_metadata, load_otu_table
from mbhd.labels import (
    resolved_case_control,
    restrict_to_condition,
    restrict_to_sample_type,
)

#: One project-wide constant (every stochastic step takes an explicit seed
#: argument; no global ``np.random.seed()``), threaded
#: explicitly into LogisticRegression and the permutation shuffles by every
#: caller -- never a bare global seed call anywhere in this module.
SEED = 42

#: M5 batch-correction hook. ``(x_train, y_train, batch_train, x_test) ->
#: (x_train_corrected, x_test_corrected)`` -- see the module docstring for
#: why the held-out fold's own labels/identity are never passed in.
FoldTransform = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


class DegenerateFoldError(ValueError):
    """Raised when a held-out fold has only one class present.

    Cannot arise on the real data as of this writing (every dataset that
    reaches this point already required >=1 control and >=1 case to produce
    a comparison at all -- ``NoSurvivingSamplesError`` upstream), but
    ``roc_auc_score`` on a single-class fold would otherwise raise its own
    less legible ``ValueError`` deep inside sklearn; this fails loudly with
    the held-out group named instead.
    """


def pooled_feature_matrix(
    dataset_ids: list[str],
    genera: list[str],
    index: dict[str, DatasetInfo],
    data_root: Path,
    *,
    abundance_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build one pooled ``(X, y, groups)`` across every dataset in ``dataset_ids``.

    ``X`` is samples x ``genera`` (relative abundance from ``to_genus_
    abundance``, reindexed to the fixed ``genera`` column list with missing
    genera filled ``0.0``). ``y`` is 0 (control) / 1 (case). ``groups`` is
    each sample's study ``DatasetInfo.folder`` -- the leave-one-group-out
    grouping key, not ``dataset_id`` (module docstring).

    ``abundance_transform``, if given, is applied to each study's own
    genus x sample abundance table immediately after ``to_genus_abundance``
    and before reindexing to the fixed ``genera`` columns -- the same
    insertion point and parameter name ``mbhd.differential.
    differential_abundance`` already uses (e.g.
    ``functools.partial(mbhd.compositional.clr, pseudocount_fn=mbhd.
    compositional.multiplicative_replacement)`` for a CLR-featured
    classifier run). Default ``None`` preserves exactly today's relative-
    abundance behaviour -- regression-tested byte-identical.

    **The absent-genus fill value stays ``0.0`` regardless of
    ``abundance_transform``** -- decided, not overlooked. For relative
    abundance, ``0.0`` is literally the bottom of the scale ("floor of
    detection"). Under a transform like CLR, ``0.0`` means something
    different (each sample's own geometric mean, i.e. "average," not
    "low") -- a real, known mismatch with the floor-of-detection rationale,
    kept anyway because it is the least assumption-laden choice: it does
    not invent a second, unverified "what does absent mean in transformed
    space" convention on top of the already-decided CLR zero-replacement
    one. A caller relying on this for a transformed run should not read
    ``0.0`` there as "low abundance."

    Row keys are ``f"{dataset_id}:{sample_id}"`` rather than the bare sample
    ID: a folder shared by two dataset IDs (e.g. ``cdi_schubert``/
    ``noncdi_schubert``) can contribute the *same* physical control sample
    once per dataset ID it is used as a control for, exactly as the paper's
    own Table 1 counts that shared control pool once per row -- this keeps
    each contribution uniquely addressable rather than colliding on one
    pandas index label.

    Raises ``NoSurvivingSamplesError`` (reused from ``mbhd.differential``,
    not a second definition) if a dataset has zero surviving controls or
    cases after filtering -- the same condition ``differential_abundance``
    already refuses to run on silently.
    """
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    group_parts: list[pd.Series] = []

    for dataset_id in dataset_ids:
        entry = entry_for(index, dataset_id)
        folder = study_dir(index, dataset_id, data_root)
        metadata = load_metadata(
            metadata_path(dataset_id, folder), encoding=encoding_for(dataset_id)
        )
        metadata = restrict_to_sample_type(dataset_id, metadata)
        metadata = restrict_to_condition(dataset_id, metadata)
        otu_table = load_otu_table(otu_table_path(dataset_id, folder))

        sample_ids = analysis_samples(metadata, otu_table.columns)
        filtered = filter_table(otu_table, sample_ids)
        genus_abundance = to_genus_abundance(filtered)
        if abundance_transform is not None:
            genus_abundance = abundance_transform(genus_abundance)

        spec = resolved_case_control(dataset_id)
        control_ids = [
            s
            for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.controls)]
            if s in genus_abundance.columns
        ]
        case_ids = [
            s
            for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.cases)]
            if s in genus_abundance.columns
        ]
        if not control_ids or not case_ids:
            raise NoSurvivingSamplesError(
                f"{dataset_id}: {len(control_ids)} control(s), {len(case_ids)} "
                "case(s) left after filtering -- need at least one of each"
            )

        ordered = control_ids + case_ids
        keys = [f"{dataset_id}:{s}" for s in ordered]
        features = genus_abundance.reindex(index=genera, fill_value=0.0)[ordered].T
        features.index = pd.Index(keys)
        x_parts.append(features)
        y_parts.append(
            pd.Series(
                [0] * len(control_ids) + [1] * len(case_ids), index=keys, dtype=int
            )
        )
        group_parts.append(pd.Series([entry.folder] * len(keys), index=keys))

    x = pd.concat(x_parts, axis=0)
    y = pd.concat(y_parts, axis=0)
    groups = pd.concat(group_parts, axis=0)
    return x, y, groups


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )


def _tuned_pipeline(seed: int) -> Pipeline:
    """Like ``_pipeline``, but with ``class_weight="balanced"`` and the
    regularization strength ``C`` chosen by nested cross-validation
    (``sklearn.linear_model.LogisticRegressionCV``, sklearn's own default
    10-value log-spaced grid, 5 inner folds) instead of held fixed at 1.0.

    The inner CV that picks ``C`` sees only whatever data ``.fit()`` is
    called with -- for a LOFO fold that is the outer training fold alone,
    so nested-CV correctness follows from the same train-fold-only ``fit``
    call ``StandardScaler`` already relies on, not from anything new here.

    ``scoring="roc_auc"`` is set explicitly rather than left at sklearn's
    default, which a live ``FutureWarning`` (observed when this was
    written) confirms is changing in a future sklearn version -- this project's own
    chosen evaluation metric should not silently drift with a dependency
    upgrade, and AUC-ROC is the metric this module reports on
    imbalanced data rather than accuracy.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    class_weight="balanced",
                    scoring="roc_auc",
                    cv=5,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def leave_one_group_out_auc(
    x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int
) -> pd.DataFrame:
    """One row per held-out group: AUC-ROC and AUC-PR from a fold fit fresh.

    ``StandardScaler`` is fit inside each fold's ``Pipeline``, on that fold's
    training data only -- it never sees the held-out group's own feature
    values (cross-validation must wrap the whole pipeline, not just the
    model; satisfied by construction via ``sklearn.pipeline.Pipeline`` rather
    than a hand-rolled loop that could get the fit/transform split wrong).

    AUC-PR (``average_precision_score``) is reported alongside AUC-ROC
    because several folders are heavily imbalanced (AUC-ROC alone is already fairly
    imbalance-robust, but not assumed sufficient on its own).

    Raises ``DegenerateFoldError`` naming the held-out group if that group's
    ``y`` has only one class present (module docstring).
    """
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    groups_arr = groups.to_numpy()

    rows: list[dict[str, object]] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(x_arr, y_arr, groups_arr):
        held_out_group = groups_arr[test_idx][0]
        y_test = y_arr[test_idx]
        if len(np.unique(y_test)) < 2:
            raise DegenerateFoldError(
                f"held-out group {held_out_group!r} has only one class in "
                f"its {len(y_test)} test sample(s) -- cannot compute AUC"
            )

        pipeline = _pipeline(seed)
        pipeline.fit(x_arr[train_idx], y_arr[train_idx])
        proba = pipeline.predict_proba(x_arr[test_idx])[:, 1]

        rows.append(
            {
                "group": held_out_group,
                "n_test": int(len(test_idx)),
                "n_control": int((y_test == 0).sum()),
                "n_case": int((y_test == 1).sum()),
                "auc_roc": float(roc_auc_score(y_test, proba)),
                "auc_pr": float(average_precision_score(y_test, proba)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_group_out_auc_dynamic_features(
    x_full: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    features_for_fold: Callable[[str], list[str]],
    *,
    pipeline_fn: Callable[[int], Pipeline] = _pipeline,
    fold_transform: FoldTransform | None = None,
) -> pd.DataFrame:
    """Like ``leave_one_group_out_auc``, but each fold's feature-column
    subset of ``x_full`` is (re)computed fresh by ``features_for_fold(
    held_out_group)``, which the caller must implement using only that
    fold's *training* data.

    Closes a feature-selection leakage path found during review (M4):
    if a caller instead selects features once, globally, from every
    group's own labels before ever calling ``leave_one_group_out_auc``,
    the held-out group's own labels can have influenced which features it
    is later scored on. Here, the caller is handed only the held-out
    group's name and is responsible for deriving that fold's feature list
    from training data alone -- this function does not see or pass along
    anything from the held-out group except its identity.

    ``pipeline_fn`` builds the ``Pipeline`` fit inside each fold; default
    ``_pipeline`` preserves today's plain-``LogisticRegression`` behaviour
    exactly. Pass ``_tuned_pipeline`` for the class-weighted, nested-CV-
    tuned variant (M4 follow-up). If the fitted classifier exposes a
    fitted regularization path (``C_``, as ``LogisticRegressionCV`` does),
    the selected value is recorded per fold as ``selected_C`` -- cheap
    transparency into what the tuning actually chose, ``None`` for a
    plain, untuned pipeline.

    ``fold_transform`` (M5), if given, is applied to each fold's already
    column-selected ``(x_train, x_test)`` arrays immediately before
    ``pipeline_fn`` is fit -- see the module docstring. Default ``None``
    is byte-identical to omitting it entirely.

    Raises ``DegenerateFoldError`` naming the held-out group if that
    group's ``y`` has only one class present (module docstring).
    """
    y_arr = y.to_numpy(dtype=int)
    groups_arr = groups.to_numpy()

    rows: list[dict[str, object]] = []
    for held_out_group in pd.unique(groups_arr):
        train_mask = groups_arr != held_out_group
        test_mask = groups_arr == held_out_group
        y_test = y_arr[test_mask]
        if len(np.unique(y_test)) < 2:
            raise DegenerateFoldError(
                f"held-out group {held_out_group!r} has only one class in "
                f"its {int(test_mask.sum())} test sample(s) -- cannot compute AUC"
            )

        features = features_for_fold(held_out_group)
        x_arr = x_full[features].to_numpy(dtype=float)

        x_train, x_test = x_arr[train_mask], x_arr[test_mask]
        if fold_transform is not None:
            x_train, x_test = fold_transform(
                x_train, y_arr[train_mask], groups_arr[train_mask], x_test
            )

        pipeline = pipeline_fn(seed)
        pipeline.fit(x_train, y_arr[train_mask])
        proba = pipeline.predict_proba(x_test)[:, 1]
        # LogisticRegressionCV.C_ is an array (one value per class; binary
        # classification here always gives a length-1 array) -- a live
        # FutureWarning (scikit-learn 1.9.1) says this attribute's shape is
        # slated to simplify in 1.10 (`use_legacy_attributes`); this reads
        # the current, installed-version shape, pinned in pyproject.toml,
        # not a forward guess.
        selected_c = getattr(pipeline.named_steps["clf"], "C_", None)

        rows.append(
            {
                "group": held_out_group,
                "n_features": len(features),
                "n_test": int(test_mask.sum()),
                "n_control": int((y_test == 0).sum()),
                "n_case": int((y_test == 1).sum()),
                "auc_roc": float(roc_auc_score(y_test, proba)),
                "auc_pr": float(average_precision_score(y_test, proba)),
                "selected_C": float(selected_c[0]) if selected_c is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _shuffle_within_groups(
    y: pd.Series, groups: pd.Series, rng: np.random.Generator
) -> pd.Series:
    """Permute ``y`` within each group independently -- never across groups.

    Shuffling across groups would leak group identity into the label
    (a sample's group membership would then predict its shuffled label);
    within-group shuffling preserves each fold's own class balance while
    destroying any real feature-label association.
    """
    shuffled = y.copy()
    for group in groups.unique():
        idx = np.flatnonzero((groups == group).to_numpy())
        permuted = rng.permutation(idx)
        shuffled.iloc[idx] = y.iloc[permuted].to_numpy()
    return shuffled


def permutation_null_auc(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    n_repeats: int = 200,
) -> np.ndarray:
    """Empirical null: mean LOFO AUC-ROC under ``n_repeats`` within-group label
    shuffles.

    One ``numpy.random.Generator`` seeded once with ``seed`` drives every
    shuffle across all repeats (one seed, passed explicitly, no
    bare global ``np.random.seed()``); the classifier's own ``random_state``
    stays fixed at ``seed`` for every repeat, matching the real run, so only
    the label permutation varies. Returns one mean AUC-ROC per repeat -- the
    real result is judged against this distribution, not a textbook p-value
    (there is no published ground truth for this question).
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_repeats)
    for i in range(n_repeats):
        y_shuffled = _shuffle_within_groups(y, groups, rng)
        fold_results = leave_one_group_out_auc(x, y_shuffled, groups, seed)
        means[i] = fold_results["auc_roc"].mean()
    return means


def permutation_null_auc_dynamic(
    x_full: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    features_for_fold: Callable[[str], list[str]],
    n_repeats: int = 200,
    *,
    pipeline_fn: Callable[[int], Pipeline] = _pipeline,
    fold_transform: FoldTransform | None = None,
) -> np.ndarray:
    """Like ``permutation_null_auc``, but each repeat's LOFO pass is
    ``leave_one_group_out_auc_dynamic_features`` rather than a fixed
    feature matrix -- so the null is judged under the same leak-free,
    per-fold feature-selection design as the real result it is compared
    against, not a fixed, globally-selected feature pool.

    ``features_for_fold`` is derived from the real, unshuffled labels
    (exactly as the real run's own feature selection is) and reused
    unchanged across every repeat -- only the labels fed to the classifier
    are shuffled, matching ``permutation_null_auc``'s own convention of
    holding feature selection fixed and varying only the label-feature
    association being tested.

    ``pipeline_fn`` (default ``_pipeline``) is forwarded to every repeat's
    ``leave_one_group_out_auc_dynamic_features`` call -- pass the same
    ``pipeline_fn`` used for the real result being tested (e.g.
    ``_tuned_pipeline``), so the null and the real result are judged under
    an identical model, not just identical features.

    ``fold_transform`` (M5), if given, is likewise forwarded on every
    repeat. Critically, ``y_train`` reaches the hook fresh from
    ``y_shuffled`` on every call (never closure-captured from the real,
    unshuffled labels) -- so a covariate-using correction (e.g.
    ``mbhd.batch.combat_for_fold(..., use_covariate=True)``) is correctly
    recomputed against the *shuffled* labels each repeat, exactly as
    ``pipeline_fn``'s own fit already is. Without this, a label-informed
    correction would bake a real signal into every "null" repeat and
    invalidate the whole comparison.
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_repeats)
    for i in range(n_repeats):
        y_shuffled = _shuffle_within_groups(y, groups, rng)
        fold_results = leave_one_group_out_auc_dynamic_features(
            x_full,
            y_shuffled,
            groups,
            seed,
            features_for_fold,
            pipeline_fn=pipeline_fn,
            fold_transform=fold_transform,
        )
        means[i] = fold_results["auc_roc"].mean()
    return means


def within_study_cv_auc(
    x: pd.DataFrame, y: pd.Series, seed: int, n_splits: int = 5
) -> tuple[float, int]:
    """Secondary, M3-Fig-1a-style comparison: stratified k-fold AUC-ROC within
    one study.

    Reported as context, not as the headline result: unlike
    ``leave_one_group_out_auc``, train and test samples here come from the
    *same* study, so a high score cannot distinguish a real, transferable
    biological signal from that one study's own technical/batch artifact.

    Returns ``(mean_auc_roc, actual_n_splits)``. ``n_splits`` is capped at
    the smaller class's own sample count when that is less than the
    requested value, since ``StratifiedKFold`` cannot form more folds than
    the minority class has members; the actual number used is returned
    rather than silently reduced without a record.

    Raises ``DegenerateFoldError`` if the minority class has fewer than 2
    samples -- not even 2 stratified folds can be formed.
    """
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    minority = int(np.bincount(y_arr).min())
    actual_n_splits = min(n_splits, minority)
    if actual_n_splits < 2:
        raise DegenerateFoldError(
            f"minority class has only {minority} sample(s) -- cannot form "
            "even 2 stratified folds"
        )

    splitter = StratifiedKFold(
        n_splits=actual_n_splits, shuffle=True, random_state=seed
    )
    scores = []
    for train_idx, test_idx in splitter.split(x_arr, y_arr):
        pipeline = _pipeline(seed)
        pipeline.fit(x_arr[train_idx], y_arr[train_idx])
        proba = pipeline.predict_proba(x_arr[test_idx])[:, 1]
        scores.append(roc_auc_score(y_arr[test_idx], proba))
    return float(np.mean(scores)), actual_n_splits
