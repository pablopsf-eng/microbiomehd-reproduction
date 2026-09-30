#!/usr/bin/env python3
"""M4 classifier question: does a linear classifier trained on the
shared-response genus signature generalize to an entirely unseen study?

**Decided (2026-09-18, Pablo):** standalone/manual, not wired into the
Snakemake DAG -- there is no
published ground truth for this question (unlike the rest of the DAG), and
this avoids lengthening the full pipeline's real, network-dependent run for
an exploratory analysis.

Primary result: leave-one-*folder*-out (LOFO) AUC-ROC/AUC-PR, restricted to
a **per-fold-recomputed** shared-response genus pool -- each fold's feature
list is derived only from that fold's own 26/27 training datasets
(``mbhd.classifier.leave_one_group_out_auc_dynamic_features``), never from
the held-out study, closing a feature-selection leakage path found during
this milestone's review (see "Known limitation" below, now fixed rather
than only documented). The **global-pool** version originally
specified (one fixed 59-genus list, taken from all 29 datasets at once,
matching ``docs/reproduction-m3.md``'s own already-committed shared-response
pool) is also still run and reported side by side, so the size of the
leakage effect is visible rather than merely asserted. Two further reported
(non-primary) comparison arms (full outer-join genus set, disease-specific
genus pool -- also per-fold-recomputed) and one reported secondary analysis
(within-study stratified k-fold, the M3-Fig-1a-style comparison) are also
run and written out.

``ob_zupancic`` is excluded up front, not caught as a per-dataset failure --
this is the same non-negotiable exclusion M2/M3 already document (it
hard-errors in ``to_genus_abundance`` before any feature can be computed at
all), not a judgment call made here.

**Known limitation, in its original (global-pool-only) form: fixed by this
revision, 2026-09-18.** The original design specified "the M3
full-29-vs-GitHub shared-response genus pool" as the classifier's feature
set -- one fixed, already-committed list computed from all 29 datasets at
once. A held-out fold's own case/control labels can have helped decide
whether a borderline genus crossed the "significant in >=2 distinct
diseases" threshold and made it into that fixed list, before that same
fold is scored against it -- a feature-selection leakage path (cross-validation
must wrap the entire pipeline, not just the model), found by both the
``verifier`` and ``code-reviewer`` passes and
confirmed by Pablo as worth fixing rather than only documenting. This
revision keeps the original fixed-pool numbers (now labelled
``*_global_pool``) alongside new per-fold-recomputed numbers (the arms
without that suffix), so the gap between the two is a real, reported
number rather than an unquantified caveat.

Usage:
    uv run scripts/classifier.py
    uv run scripts/classifier.py --out results/classifier --data-root data/raw \
        --n-repeats 200
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

from mbhd.batch import center_per_batch, combat, combat_for_fold
from mbhd.classifier import (
    SEED,
    DegenerateFoldError,
    FoldTransform,
    _pipeline,
    _tuned_pipeline,
    leave_one_group_out_auc,
    leave_one_group_out_auc_dynamic_features,
    permutation_null_auc_dynamic,
    pooled_feature_matrix,
    within_study_cv_auc,
)
from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import DatasetInfo, analysed_dataset_ids, load_dataset_info
from mbhd.differential import EXPECTED_DATASET_ERRORS, differential_abundance
from mbhd.shared_response import (
    PUBLISHED_EXCLUSIONS,
    QValueRows,
    is_significant,
    shared_response_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "classifier"
_DEFAULT_N_REPEATS = 200

#: Not a judgment call -- see module docstring.
_EXCLUDED = frozenset({"ob_zupancic"})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=_DEFAULT_N_REPEATS,
        help=(
            "permutation-null repeats (default: 200; if runtime makes this "
            "impractical, pass a smaller value -- the actual number used is "
            "always recorded in provenance.json, never silently reduced)"
        ),
    )
    parser.add_argument(
        "--features",
        choices=("relative_abundance", "clr"),
        default="relative_abundance",
        help=(
            "value representation for the pooled feature matrix (default: "
            "relative_abundance, byte-identical to earlier runs). 'clr' "
            "additionally runs the same leak-free shared-response LOFO + "
            "permutation null on CLR-transformed values (mbhd.compositional"
            ".clr, multiplicative-replacement zero handling) for comparison "
            "-- written to *_clr.csv / a _clr summary block, never "
            "overwriting the relative_abundance outputs"
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "additionally run the shared-response arm (for whichever "
            "representation(s) --features selects) through class_weight="
            "'balanced' + nested-CV-tuned C (mbhd.classifier._tuned_pipeline) "
            "-- written to *_tuned.csv / a _tuned summary block, never "
            "overwriting the plain-model outputs. Combined with "
            "--batch-correct, also tunes the adopted ComBat arm (row 3, "
            "no covariate, leak-free) -- written to "
            "*_clr_combat_tuned.csv / a matching summary block. Off by "
            "default: no behaviour change unless passed"
        ),
    )
    parser.add_argument(
        "--batch-correct",
        action="store_true",
        help=(
            "M5: additionally run the CLR shared-response arm through "
            "mbhd.batch's batch-effect correction (per-batch mean-centring, "
            "leak-free per-fold ComBat with/without a case/control "
            "covariate, plus two naive fit-once-globally ComBat comparisons "
            "that quantify what each leak would have cost) -- requires "
            "--features clr. Written to *_clr_{center,combat,combat_cov,"
            "combat_global,combat_cov_global}.csv / matching summary "
            "blocks, never overwriting the uncorrected CLR outputs. Off by "
            "default: no behaviour change unless passed"
        ),
    )
    args = parser.parse_args(argv)
    if args.batch_correct and args.features != "clr":
        parser.error("--batch-correct requires --features clr")
    return args


def _compute_per_dataset_q(
    dataset_ids: list[str], index: dict[str, DatasetInfo], data_root: Path
) -> dict[str, pd.Series]:
    """Each dataset's signed q-values, computed once (the expensive step --
    Kruskal-Wallis + Benjamini-Hochberg per genus) and reused by every
    per-fold genus-set recomputation below, instead of re-running
    differential_abundance once per fold per arm."""
    per_dataset_q: dict[str, pd.Series] = {}
    for dataset_id in dataset_ids:
        try:
            per_dataset_q[dataset_id] = differential_abundance(
                dataset_id, index, data_root
            )["signed_q_value"]
        except EXPECTED_DATASET_ERRORS as exc:
            print(
                f"differential_abundance failed for {dataset_id}: {exc}",
                file=sys.stderr,
            )
    return per_dataset_q


def _rows_from_per_dataset_q(
    per_dataset_q: dict[str, pd.Series], dataset_ids: list[str]
) -> QValueRows:
    """The same outer-join shape as mbhd.shared_response.build_qvalue_matrix,
    built from an already-computed cache restricted to dataset_ids -- lets a
    caller cheaply ask "what would the q-value matrix look like using only
    this subset of datasets" without re-running any statistics."""
    all_genera: set[str] = set()
    for dataset_id in dataset_ids:
        if dataset_id in per_dataset_q:
            all_genera.update(per_dataset_q[dataset_id].index)
    return {
        genus: {
            dataset_id: (
                float(per_dataset_q[dataset_id][genus])
                if dataset_id in per_dataset_q
                and genus in per_dataset_q[dataset_id].index
                else None
            )
            for dataset_id in dataset_ids
        }
        for genus in sorted(all_genera)
    }


def _genus_sets_from_rows(rows: QValueRows) -> dict[str, list[str]]:
    """shared_response/disease_specific column lists
    from an already-built q-value matrix -- the pool-splitting logic
    scripts/figures.py's fig3c-equivalent already uses (M3), reused
    unchanged, not reinvented."""
    pool = [g for g in rows if any(is_significant(q) for q in rows[g].values())]
    shared_labels = shared_response_labels(rows, PUBLISHED_EXCLUSIONS)
    return {
        "shared_response": sorted(g for g in pool if shared_labels.get(g, "") != ""),
        "disease_specific": sorted(g for g in pool if shared_labels.get(g, "") == ""),
    }


def _dynamic_features_for(
    arm_key: str,
    dataset_ids: list[str],
    index: dict[str, DatasetInfo],
    per_dataset_q: dict[str, pd.Series],
) -> Callable[[str], list[str]]:
    """A features_for_fold callback (mbhd.classifier.
    leave_one_group_out_auc_dynamic_features) that recomputes arm_key's
    genus list from only the training folders' own datasets -- the
    held-out folder's data never enters this computation, closing the
    feature-selection leakage path named in the module docstring."""

    def features_for_fold(held_out_group: str) -> list[str]:
        train_ids = [d for d in dataset_ids if index[d].folder != held_out_group]
        rows = _rows_from_per_dataset_q(per_dataset_q, train_ids)
        return _genus_sets_from_rows(rows)[arm_key]

    return features_for_fold


def _run_arm(
    label: str,
    null_filename: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    features_for_fold: Callable[[str], list[str]],
    n_repeats: int,
    out_dir: Path,
    summary: dict[str, object],
    *,
    pipeline_fn: Callable[[int], object] = _pipeline,
    fold_transform: FoldTransform | None = None,
) -> None:
    """LOFO + permutation null for ``label`` (the full, final label, e.g.
    "shared_response_tuned" or "shared_response_clr_combat" -- callers
    build the whole name, this function does not append a suffix of its
    own). Writes ``lofo_{label}.csv`` / ``{null_filename}`` and a
    ``{label}`` / ``permutation_null_{label}`` block in ``summary``,
    mutated in place.

    Generalizes the M4 tuning follow-up's original ``_run_tuned_arm``
    (pipeline_fn=_tuned_pipeline was its only option) to also support the
    M5 ``fold_transform`` batch-correction hook -- both are optional
    keywords with today's plain-model, no-correction defaults, so a caller
    passing neither reproduces the exact original behaviour."""
    lofo = leave_one_group_out_auc_dynamic_features(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=features_for_fold,
        pipeline_fn=pipeline_fn,
        fold_transform=fold_transform,
    )
    lofo.to_csv(out_dir / f"lofo_{label}.csv", index=False)
    block: dict[str, object] = {
        "n_features_min": int(lofo["n_features"].min()),
        "n_features_max": int(lofo["n_features"].max()),
        "n_folds": len(lofo),
        "mean_auc_roc": float(lofo["auc_roc"].mean()),
        "mean_auc_pr": float(lofo["auc_pr"].mean()),
    }
    if lofo["selected_C"].notna().any():
        block["mean_selected_C"] = float(lofo["selected_C"].mean())
    summary[label] = block
    print(f"{label}: mean AUC-ROC={block['mean_auc_roc']:.3f}")

    null = permutation_null_auc_dynamic(
        x,
        y,
        groups,
        seed=SEED,
        features_for_fold=features_for_fold,
        n_repeats=n_repeats,
        pipeline_fn=pipeline_fn,
        fold_transform=fold_transform,
    )
    pd.DataFrame({"repeat": range(len(null)), "mean_auc_roc": null}).to_csv(
        out_dir / null_filename, index=False
    )
    n_at_or_above = int((null >= block["mean_auc_roc"]).sum())
    p_value = (n_at_or_above + 1) / (len(null) + 1)
    summary[f"permutation_null_{label}"] = {
        "n_repeats": len(null),
        "null_mean_auc_roc": float(null.mean()),
        "null_std_auc_roc": float(null.std()),
        "empirical_p_value": p_value,
    }
    print(
        f"permutation null ({label}, {len(null)} repeats): "
        f"mean={null.mean():.3f}, std={null.std():.3f}, empirical p={p_value:.3f}"
    )


def _run_global_combat_arm(
    label: str,
    x_global: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    covariate: pd.Series | None,
    out_dir: Path,
    summary: dict[str, object],
) -> None:
    """Fits ``mbhd.batch.combat`` once on every folder together (including
    the labels of whichever folder LOFO will later hold out) and scores it
    with the plain, non-dynamic ``leave_one_group_out_auc`` -- the
    deliberately naive/leaky M5 comparison rows (5-6) that quantify what
    each leak (mbhd.batch's own module docstring: cross-batch information
    in the OLS standardization step; a covariate's direct label leak on
    held-out rows) would have cost, mirroring M4's own leaky-global-pool-
    vs-leak-free-per-fold reporting pattern. Never offered as a candidate
    result -- shown explicitly as a warning comparison."""
    corrected_array, _fit = combat(
        x_global.to_numpy(dtype=float),
        groups.to_numpy(),
        covariate=covariate.to_numpy(dtype=float) if covariate is not None else None,
    )
    corrected = pd.DataFrame(
        corrected_array, index=x_global.index, columns=x_global.columns
    )
    lofo = leave_one_group_out_auc(corrected, y, groups, seed=SEED)
    lofo.to_csv(out_dir / f"lofo_{label}.csv", index=False)
    summary[label] = {
        "n_features": x_global.shape[1],
        "n_folds": len(lofo),
        "mean_auc_roc": float(lofo["auc_roc"].mean()),
        "mean_auc_pr": float(lofo["auc_pr"].mean()),
    }
    mean_auc = summary[label]["mean_auc_roc"]
    print(f"{label} (naive, fit once globally): mean AUC-ROC={mean_auc:.3f}")


def _run_batch_correct_arms(
    x_full_clr: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    features_for_fold: Callable[[str], list[str]],
    shared_response_genera: list[str],
    n_repeats: int,
    out_dir: Path,
    summary: dict[str, object],
    *,
    tune: bool = False,
) -> None:
    """M5: the six-row batch-correction comparison table on top of the
    already-computed, uncorrected CLR shared-response arm (row 1,
    ``summary["shared_response_clr"]``, not recomputed here).

    Rows 2-4 are leak-free, per-fold-recomputed (same design as the
    uncorrected CLR arm, just with an ``mbhd.batch`` ``fold_transform``
    added): per-batch mean-centring, ComBat without a covariate, and
    ComBat with the case/control label as a covariate on the training side
    only (the held-out fold is corrected label-blind, ``mbhd.batch.
    combat_new_batch``'s own contract). Rows 5-6 are the deliberately naive
    fit-once-globally comparisons that quantify what each leak would have
    cost -- warnings, never candidate results.

    ``tune`` (M5 follow-up, ``--tune`` combined with ``--batch-correct``):
    additionally runs the adopted headline (row 3, ComBat, no covariate,
    leak-free) through ``_tuned_pipeline`` instead of the plain model --
    one more question ("does tuning help the adopted result further?"),
    not a wider grid over every row above.
    """
    _run_arm(
        "shared_response_clr_center",
        "permutation_null_clr_center.csv",
        x_full_clr,
        y,
        groups,
        features_for_fold,
        n_repeats,
        out_dir,
        summary,
        fold_transform=center_per_batch,
    )
    _run_arm(
        "shared_response_clr_combat",
        "permutation_null_clr_combat.csv",
        x_full_clr,
        y,
        groups,
        features_for_fold,
        n_repeats,
        out_dir,
        summary,
        fold_transform=partial(combat_for_fold, use_covariate=False),
    )
    _run_arm(
        "shared_response_clr_combat_cov",
        "permutation_null_clr_combat_cov.csv",
        x_full_clr,
        y,
        groups,
        features_for_fold,
        n_repeats,
        out_dir,
        summary,
        fold_transform=partial(combat_for_fold, use_covariate=True),
    )

    x_shared_global = x_full_clr[shared_response_genera]
    _run_global_combat_arm(
        "shared_response_clr_combat_global",
        x_shared_global,
        y,
        groups,
        covariate=None,
        out_dir=out_dir,
        summary=summary,
    )
    _run_global_combat_arm(
        "shared_response_clr_combat_cov_global",
        x_shared_global,
        y,
        groups,
        covariate=y,
        out_dir=out_dir,
        summary=summary,
    )

    print(
        "batch-correct summary: "
        f"no-correction={summary['shared_response_clr']['mean_auc_roc']:.3f}, "
        f"center={summary['shared_response_clr_center']['mean_auc_roc']:.3f}, "
        f"combat={summary['shared_response_clr_combat']['mean_auc_roc']:.3f}, "
        f"combat_cov={summary['shared_response_clr_combat_cov']['mean_auc_roc']:.3f}, "
        "combat_global(leaky)="
        f"{summary['shared_response_clr_combat_global']['mean_auc_roc']:.3f}, "
        "combat_cov_global(leaky+label-leak)="
        f"{summary['shared_response_clr_combat_cov_global']['mean_auc_roc']:.3f}"
    )

    if tune:
        _run_arm(
            "shared_response_clr_combat_tuned",
            "permutation_null_clr_combat_tuned.csv",
            x_full_clr,
            y,
            groups,
            features_for_fold,
            n_repeats,
            out_dir,
            summary,
            pipeline_fn=_tuned_pipeline,
            fold_transform=partial(combat_for_fold, use_covariate=False),
        )
        print(
            "batch-correct + tune: adopted headline (combat, plain)="
            f"{summary['shared_response_clr_combat']['mean_auc_roc']:.3f} vs. "
            "tuned="
            f"{summary['shared_response_clr_combat_tuned']['mean_auc_roc']:.3f}"
        )


def _provenance(
    n_repeats: int,
    dataset_ids: list[str],
    features: str,
    tune: bool,
    batch_correct: bool,
) -> dict[str, object]:
    versions = {}
    for package in (
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "pyyaml",
        "pycombat",
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "seed": SEED,
        "n_repeats": n_repeats,
        "features": features,
        "tune": tune,
        "batch_correct": batch_correct,
        "dataset_ids": dataset_ids,
        "excluded_dataset_ids": sorted(_EXCLUDED),
        **{f"{pkg}_version": v for pkg, v in versions.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)

    index = load_dataset_info(args.data_root / "dataset_info.yaml")
    dataset_ids = [d for d in analysed_dataset_ids(index) if d not in _EXCLUDED]

    per_dataset_q = _compute_per_dataset_q(dataset_ids, index, args.data_root)
    global_rows = _rows_from_per_dataset_q(per_dataset_q, dataset_ids)
    global_sets = _genus_sets_from_rows(global_rows)
    global_sets["full"] = sorted(global_rows.keys())
    print(
        f"global (all-29) feature sets: "
        f"shared_response={len(global_sets['shared_response'])}, "
        f"disease_specific={len(global_sets['disease_specific'])}, "
        f"full={len(global_sets['full'])}"
    )

    x_full, y, groups = pooled_feature_matrix(
        dataset_ids, global_sets["full"], index, args.data_root
    )
    print(f"pooled matrix: {x_full.shape[0]} samples, {groups.nunique()} folders")

    summary: dict[str, object] = {}

    # "full" arm: not leakage-sensitive (every genus ever reproduced,
    # regardless of significance -- no label-based selection at all), so it
    # is only ever run once, on the fixed global set.
    lofo_full = leave_one_group_out_auc(x_full, y, groups, seed=SEED)
    lofo_full.to_csv(args.out / "lofo_full.csv", index=False)
    summary["full"] = {
        "n_features": x_full.shape[1],
        "n_folds": len(lofo_full),
        "mean_auc_roc": float(lofo_full["auc_roc"].mean()),
        "mean_auc_pr": float(lofo_full["auc_pr"].mean()),
    }
    print(
        f"full: {x_full.shape[1]} features, {len(lofo_full)} folds, "
        f"mean AUC-ROC={summary['full']['mean_auc_roc']:.3f}"
    )

    # shared_response / disease_specific: run BOTH the original
    # fixed global-pool version (*_global_pool, kept for comparison) and the
    # per-fold-recomputed, leak-free version (the new primary/reported
    # arms) -- the difference between the two is the leakage effect's size.
    for arm_key in ("shared_response", "disease_specific"):
        x_global = x_full[global_sets[arm_key]]
        lofo_global = leave_one_group_out_auc(x_global, y, groups, seed=SEED)
        lofo_global.to_csv(args.out / f"lofo_{arm_key}_global_pool.csv", index=False)
        summary[f"{arm_key}_global_pool"] = {
            "n_features": x_global.shape[1],
            "n_folds": len(lofo_global),
            "mean_auc_roc": float(lofo_global["auc_roc"].mean()),
            "mean_auc_pr": float(lofo_global["auc_pr"].mean()),
        }

        lofo_dynamic = leave_one_group_out_auc_dynamic_features(
            x_full,
            y,
            groups,
            seed=SEED,
            features_for_fold=_dynamic_features_for(
                arm_key, dataset_ids, index, per_dataset_q
            ),
        )
        lofo_dynamic.to_csv(args.out / f"lofo_{arm_key}.csv", index=False)
        summary[arm_key] = {
            "n_features_min": int(lofo_dynamic["n_features"].min()),
            "n_features_max": int(lofo_dynamic["n_features"].max()),
            "n_folds": len(lofo_dynamic),
            "mean_auc_roc": float(lofo_dynamic["auc_roc"].mean()),
            "mean_auc_pr": float(lofo_dynamic["auc_pr"].mean()),
        }
        print(
            f"{arm_key}: global-pool mean AUC-ROC="
            f"{summary[f'{arm_key}_global_pool']['mean_auc_roc']:.3f} "
            f"({summary[f'{arm_key}_global_pool']['n_features']} features) vs. "
            f"per-fold (leak-free) mean AUC-ROC="
            f"{summary[arm_key]['mean_auc_roc']:.3f} "
            f"({summary[arm_key]['n_features_min']}-"
            f"{summary[arm_key]['n_features_max']} features)"
        )

    # Permutation null is run against the leak-free primary arm, using that
    # same per-fold feature recomputation inside every shuffle repeat.
    null = permutation_null_auc_dynamic(
        x_full,
        y,
        groups,
        seed=SEED,
        n_repeats=args.n_repeats,
        features_for_fold=_dynamic_features_for(
            "shared_response", dataset_ids, index, per_dataset_q
        ),
    )
    pd.DataFrame({"repeat": range(len(null)), "mean_auc_roc": null}).to_csv(
        args.out / "permutation_null.csv", index=False
    )
    # (b+1)/(m+1), the standard unbiased Monte Carlo permutation p-value
    # estimator (North, Curtis & Sham 2002) -- never the raw b/m fraction,
    # which can report an impossible exact 0.0 and overstates the certainty
    # a finite number of resamples can actually establish.
    n_at_or_above = int((null >= summary["shared_response"]["mean_auc_roc"]).sum())
    p_value = (n_at_or_above + 1) / (len(null) + 1)
    summary["permutation_null"] = {
        "n_repeats": len(null),
        "null_mean_auc_roc": float(null.mean()),
        "null_std_auc_roc": float(null.std()),
        "empirical_p_value": p_value,
    }
    print(
        f"permutation null ({len(null)} repeats, leak-free): "
        f"mean={null.mean():.3f}, std={null.std():.3f}, empirical p={p_value:.3f}"
    )

    if args.tune:
        _run_arm(
            "shared_response_tuned",
            "permutation_null_tuned.csv",
            x_full,
            y,
            groups,
            _dynamic_features_for("shared_response", dataset_ids, index, per_dataset_q),
            args.n_repeats,
            args.out,
            summary,
            pipeline_fn=_tuned_pipeline,
        )

    if args.features == "clr":
        # Same leak-free per-fold feature *selection* as the relative-
        # abundance primary arm (it depends only on per_dataset_q's signed
        # q-values, never on x's own values) -- only the VALUES fed to the
        # classifier change, isolating that one variable.
        x_full_clr, y_clr, groups_clr = pooled_feature_matrix(
            dataset_ids,
            global_sets["full"],
            index,
            args.data_root,
            abundance_transform=partial(clr, pseudocount_fn=multiplicative_replacement),
        )
        if not (y_clr.equals(y) and groups_clr.equals(groups)):
            raise ValueError(
                "CLR pooled matrix has different samples/labels/groups than "
                "the relative-abundance one -- the two runs would not be "
                "comparable"
            )

        clr_features_for_fold = _dynamic_features_for(
            "shared_response", dataset_ids, index, per_dataset_q
        )
        lofo_clr = leave_one_group_out_auc_dynamic_features(
            x_full_clr, y, groups, seed=SEED, features_for_fold=clr_features_for_fold
        )
        lofo_clr.to_csv(args.out / "lofo_shared_response_clr.csv", index=False)
        summary["shared_response_clr"] = {
            "n_features_min": int(lofo_clr["n_features"].min()),
            "n_features_max": int(lofo_clr["n_features"].max()),
            "n_folds": len(lofo_clr),
            "mean_auc_roc": float(lofo_clr["auc_roc"].mean()),
            "mean_auc_pr": float(lofo_clr["auc_pr"].mean()),
        }
        print(
            f"shared_response (CLR features): mean AUC-ROC="
            f"{summary['shared_response_clr']['mean_auc_roc']:.3f} vs. "
            f"relative-abundance mean AUC-ROC="
            f"{summary['shared_response']['mean_auc_roc']:.3f}"
        )

        null_clr = permutation_null_auc_dynamic(
            x_full_clr,
            y,
            groups,
            seed=SEED,
            n_repeats=args.n_repeats,
            features_for_fold=clr_features_for_fold,
        )
        pd.DataFrame({"repeat": range(len(null_clr)), "mean_auc_roc": null_clr}).to_csv(
            args.out / "permutation_null_clr.csv", index=False
        )
        n_at_or_above_clr = int(
            (null_clr >= summary["shared_response_clr"]["mean_auc_roc"]).sum()
        )
        p_value_clr = (n_at_or_above_clr + 1) / (len(null_clr) + 1)
        summary["permutation_null_clr"] = {
            "n_repeats": len(null_clr),
            "null_mean_auc_roc": float(null_clr.mean()),
            "null_std_auc_roc": float(null_clr.std()),
            "empirical_p_value": p_value_clr,
        }
        print(
            f"permutation null (CLR, {len(null_clr)} repeats): "
            f"mean={null_clr.mean():.3f}, std={null_clr.std():.3f}, "
            f"empirical p={p_value_clr:.3f}"
        )

        if args.tune:
            _run_arm(
                "shared_response_clr_tuned",
                "permutation_null_clr_tuned.csv",
                x_full_clr,
                y,
                groups,
                clr_features_for_fold,
                args.n_repeats,
                args.out,
                summary,
                pipeline_fn=_tuned_pipeline,
            )

        if args.batch_correct:
            _run_batch_correct_arms(
                x_full_clr,
                y,
                groups,
                clr_features_for_fold,
                global_sets["shared_response"],
                args.n_repeats,
                args.out,
                summary,
                tune=args.tune,
            )

    within_study_rows = []
    x_shared_global = x_full[global_sets["shared_response"]]
    for dataset_id in dataset_ids:
        keys = [k for k in x_shared_global.index if k.startswith(f"{dataset_id}:")]
        x_d, y_d = x_shared_global.loc[keys], y.loc[keys]
        try:
            mean_auc, actual_n_splits = within_study_cv_auc(x_d, y_d, seed=SEED)
        except DegenerateFoldError as exc:
            within_study_rows.append(
                {"dataset_id": dataset_id, "error": str(exc), "mean_auc_roc": None}
            )
            continue
        within_study_rows.append(
            {
                "dataset_id": dataset_id,
                "n_splits": actual_n_splits,
                "mean_auc_roc": mean_auc,
                "error": None,
            }
        )
    within_study_df = pd.DataFrame(within_study_rows)
    within_study_df.to_csv(args.out / "within_study.csv", index=False)
    n_within_ok = within_study_df["mean_auc_roc"].notna().sum()
    summary["within_study"] = {
        "n_datasets": len(within_study_df),
        "n_computed": int(n_within_ok),
        "mean_auc_roc": float(within_study_df["mean_auc_roc"].dropna().mean())
        if n_within_ok
        else None,
    }
    print(
        f"within-study k-fold: {n_within_ok}/{len(within_study_df)} datasets, "
        f"mean AUC-ROC={summary['within_study']['mean_auc_roc']}"
    )

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "provenance.json").write_text(
        json.dumps(
            _provenance(
                args.n_repeats,
                dataset_ids,
                args.features,
                args.tune,
                args.batch_correct,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"\noutput written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
