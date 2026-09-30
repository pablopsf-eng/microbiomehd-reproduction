"""Independent verifier check (not part of the M5 implementation): does
``docs/reproduction-m5.md``'s row 5 really measure what it claims to?

Real result, computed directly against ``data/raw`` (skips cleanly if that
directory is absent -- this repo never commits data; `data/` is
gitignored): an earlier version of ``docs/reproduction-m5.md`` reported
row 5 ("ComBat, no
covariate, fit once globally") as "0.651 (leaky) vs. 0.647 (leak-free row 3)
-- a +0.004 inflation ... quantif[ying] the cross-batch-fit leak ...
small, as predicted."

That is not what the row-5 experiment actually isolates. Row 5 differs from
row 3 in *two* independent ways at once: (a) the fixed, globally-selected
55-genus feature set vs. row 3's per-fold, leak-free dynamically-selected
feature set (the *same* feature-selection leak M4 already quantified as
"0.607 vs. 0.603" for relative-abundance features -- unrelated to ComBat),
and (b) fitting ``combat()`` once on all 27 folders together vs. per-fold
leak-free fitting (the actual cross-batch-fit leak).

Decomposing the two (below, and independently confirmed by the verifying
agent from both decomposition orders): holding the feature set fixed and
varying only whether ComBat is fit once globally or per-fold leak-free
changes the LOFO mean AUC-ROC by <0.0005 in either direction -- i.e. the
ComBat cross-batch-fit leak is, on this data, indistinguishable
from zero. Essentially the entire reported "+0.004" gap between row 3 and
row 5 is the pre-existing, already-documented feature-selection leak
carried over unchanged (rows 5-6 reuse the fixed global feature set), not a
new confirmation of ComBat's own leak mechanism. This does not change
row 3's adoption (row 1 vs. row 3 never involves the global feature set)
or row 6's headline label-leak finding (an order of magnitude larger, and
independently confirmed to be robust to this same confound -- see the
verifier's report). It does mean row 5's own causal narrative in
``docs/reproduction-m5.md`` is not well-supported as written.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mbhd.batch import combat, combat_for_fold
from mbhd.classifier import SEED, _pipeline, pooled_feature_matrix
from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import analysed_dataset_ids, load_dataset_info

DATA_ROOT = Path("data/raw")


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _lofo_from_precomputed(x_corrected, y, groups, features_for_fold):
    """Bare LOFO scoring loop over an *already corrected* matrix -- no
    fold_transform, since the correction under test here was deliberately
    applied once, globally, before this loop (that is exactly what row 5/6
    do, and what this test is decomposing)."""
    y_arr = y.to_numpy(dtype=int)
    groups_arr = groups.to_numpy()
    aucs = []
    for held_out_group in pd.unique(groups_arr):
        test_mask = groups_arr == held_out_group
        train_mask = ~test_mask
        features = features_for_fold(held_out_group)
        x_sub = x_corrected[features].to_numpy(dtype=float)
        pipeline = _pipeline(SEED)
        pipeline.fit(x_sub[train_mask], y_arr[train_mask])
        proba = pipeline.predict_proba(x_sub[test_mask])[:, 1]
        aucs.append(roc_auc_score(y_arr[test_mask], proba))
    return float(np.mean(aucs))


def test_row5_gap_is_the_feature_selection_leak_not_the_combat_fit_leak() -> None:
    _require(DATA_ROOT / "dataset_info.yaml")
    import classifier as clf_script  # scripts/classifier.py, path-inserted above

    index = load_dataset_info(DATA_ROOT / "dataset_info.yaml")
    dataset_ids = [
        d for d in analysed_dataset_ids(index) if d not in clf_script._EXCLUDED
    ]
    per_dataset_q = clf_script._compute_per_dataset_q(dataset_ids, index, DATA_ROOT)
    global_rows = clf_script._rows_from_per_dataset_q(per_dataset_q, dataset_ids)
    global_sets = clf_script._genus_sets_from_rows(global_rows)
    global_sets["full"] = sorted(global_rows.keys())

    x_full_clr, y, groups = pooled_feature_matrix(
        dataset_ids,
        global_sets["full"],
        index,
        DATA_ROOT,
        abundance_transform=partial(clr, pseudocount_fn=multiplicative_replacement),
    )
    shared = global_sets["shared_response"]
    x_shared_global = x_full_clr[shared]
    dynamic_features_for_fold = clf_script._dynamic_features_for(
        "shared_response", dataset_ids, index, per_dataset_q
    )
    fixed_features_for_fold = lambda _g: shared  # noqa: E731

    # Leaky global ComBat fit, no covariate (row 5's own correction step).
    corrected_leaky, _fit = combat(
        x_shared_global.to_numpy(dtype=float), groups.to_numpy()
    )
    corrected_leaky_df = pd.DataFrame(
        corrected_leaky, index=x_shared_global.index, columns=x_shared_global.columns
    )

    # A: fixed global feature set + LEAK-FREE per-fold ComBat -- isolates
    # the feature-selection leak alone (ComBat itself is leak-free here).
    from mbhd.classifier import leave_one_group_out_auc_dynamic_features

    lofo_a = leave_one_group_out_auc_dynamic_features(
        x_full_clr,
        y,
        groups,
        seed=SEED,
        features_for_fold=fixed_features_for_fold,
        fold_transform=partial(combat_for_fold, use_covariate=False),
    )
    mean_a = float(lofo_a["auc_roc"].mean())

    # Row 5 itself: fixed global feature set + LEAKY global ComBat fit.
    mean_row5 = _lofo_from_precomputed(
        corrected_leaky_df, y, groups, fixed_features_for_fold
    )

    # B: per-fold feature set (row 3's own, leak-free) + LEAKY global ComBat
    # fit -- isolates the ComBat cross-batch-fit leak alone, feature
    # selection held fixed at row 3's own convention.
    mean_b = _lofo_from_precomputed(
        corrected_leaky_df, y, groups, dynamic_features_for_fold
    )

    # Row 3 reference: both leak-free (recomputed fresh here, not read from
    # a results/ file).
    lofo_row3 = leave_one_group_out_auc_dynamic_features(
        x_full_clr,
        y,
        groups,
        seed=SEED,
        features_for_fold=dynamic_features_for_fold,
        fold_transform=partial(combat_for_fold, use_covariate=False),
    )
    mean_row3 = float(lofo_row3["auc_roc"].mean())

    leak_global_feat = mean_row5 - mean_a
    leak_per_fold_feat = mean_b - mean_row3
    feature_selection_leak = mean_a - mean_row3

    print(
        f"row3={mean_row3:.4f} row5={mean_row5:.4f} A={mean_a:.4f} B={mean_b:.4f} "
        f"combat-fit-leak(global feat)={leak_global_feat:+.4f} "
        f"combat-fit-leak(per-fold feat)={leak_per_fold_feat:+.4f} "
        f"feature-selection-leak={feature_selection_leak:+.4f}"
    )

    # The ComBat-specific cross-batch-fit leak, isolated either way, is an
    # order of magnitude smaller than the reported row3-vs-row5 gap
    # (~0.0037) -- i.e. not what's actually driving that number.
    assert abs(leak_global_feat) < 0.0015
    assert abs(leak_per_fold_feat) < 0.0015
    # The feature-selection difference alone accounts for essentially the
    # whole reported gap.
    assert abs(feature_selection_leak - (mean_row5 - mean_row3)) < 0.0015
