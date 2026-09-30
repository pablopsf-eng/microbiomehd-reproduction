"""Independent re-verification of the M4/M5 classifier headline (mean
AUC-ROC 0.635 CLR-featured, 0.647 with ComBat correction) and the leak-free
design of ``mbhd.batch.combat_for_fold``.

Two independent checks, neither reusing ``tests/test_classifier_verification.py``:

1. **Leak-free ComBat, from scratch.** Synthetic 3-batch data (not the real
   study data), a minimal stand-in for ``mbhd.classifier``'s own LOFO
   train/test slicing pattern, and -- as a check on the check itself -- a
   deliberately leaky alternative (fit ``combat()`` once on everything with
   every row's own true label as covariate) that is shown to actually leak,
   so the leak-free assertion is not vacuous.
2. **End-to-end determinism.** Re-runs ``scripts/classifier.py --features
   clr --batch-correct`` with far fewer permutation-null repeats than the
   committed 200 and asserts every LOFO mean AUC-ROC matches
   ``results/classifier/summary.json`` to full float precision (bit-for-bit)
   -- LOFO means depend only on ``SEED`` and the data, never on
   ``n_repeats`` (which only changes the permutation null). Slow (~2-3
   minutes: differential abundance + LOFO + a full ComBat batch-correction
   sweep over all 30 datasets) and gated behind ``MBHD_RUN_SLOW_TESTS=1``
   so the default ``pytest`` run stays fast; not skipped silently -- the
   skip reason states exactly what to set to run it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbhd.batch import combat, combat_for_fold

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw"
COMMITTED_SUMMARY = REPO_ROOT / "results" / "classifier" / "summary.json"


def test_combat_for_fold_is_leak_free_and_the_leak_it_avoids_is_real(tmp_path):
    """From-scratch synthetic probe (not a reuse of test_classifier_verification.py)."""
    n_per_batch = 12
    n_genes = 8

    def make_batch(batch_mean_shift, seed):
        r = np.random.default_rng(seed)
        y = np.array([0] * (n_per_batch // 2) + [1] * (n_per_batch // 2))
        covariate_effect = y[:, None] * r.normal(1.0, 0.1, size=n_genes)[None, :]
        x = (
            r.normal(0, 1, size=(n_per_batch, n_genes))
            + batch_mean_shift
            + covariate_effect
        )
        return x, y

    x_a, y_a = make_batch(0.0, 1)
    x_b, y_b = make_batch(3.0, 2)
    x_c, y_c = make_batch(-2.0, 3)  # held-out batch

    x_full = np.vstack([x_a, x_b, x_c])
    y_full_real = np.concatenate([y_a, y_b, y_c])
    batch_full = np.array(
        ["A"] * n_per_batch + ["B"] * n_per_batch + ["C"] * n_per_batch
    )
    train_mask = batch_full != "C"
    test_mask = ~train_mask

    y_full_flipped_c = y_full_real.copy()
    y_full_flipped_c[test_mask] = 1 - y_full_flipped_c[test_mask]

    def lofo_fold_leak_free(y_full: np.ndarray) -> np.ndarray:
        """mbhd.classifier.leave_one_group_out_auc_dynamic_features's own
        slicing pattern: y_train = y_arr[train_mask], fold_transform never
        sees y_arr[test_mask]."""
        y_train = y_full[train_mask]
        _, corrected_test = combat_for_fold(
            x_full[train_mask],
            y_train,
            batch_full[train_mask],
            x_full[test_mask],
            use_covariate=True,
        )
        return corrected_test

    corrected_real = lofo_fold_leak_free(y_full_real)
    corrected_flipped = lofo_fold_leak_free(y_full_flipped_c)
    np.testing.assert_array_equal(corrected_real, corrected_flipped)

    # Sanity check on the check itself: the deliberately leaky alternative
    # (fit combat() once on everything, covariate populated for every row
    # including the held-out batch's own label) must actually leak, or this
    # test would be proving nothing.
    corrected_leaky_real, _ = combat(x_full, batch_full, covariate=y_full_real)
    corrected_leaky_flipped, _ = combat(x_full, batch_full, covariate=y_full_flipped_c)
    leaky_test_real = corrected_leaky_real[test_mask]
    leaky_test_flipped = corrected_leaky_flipped[test_mask]
    assert not np.array_equal(leaky_test_real, leaky_test_flipped)
    assert np.max(np.abs(leaky_test_real - leaky_test_flipped)) > 1e-3


@pytest.mark.skipif(
    os.environ.get("MBHD_RUN_SLOW_TESTS") != "1",
    reason=(
        "slow (~2-3 min, full 30-dataset differential abundance + LOFO + "
        "ComBat sweep) -- set MBHD_RUN_SLOW_TESTS=1 to run"
    ),
)
def test_lofo_means_are_deterministic_and_match_committed_summary(tmp_path):
    if not COMMITTED_SUMMARY.exists():
        pytest.skip("data not fetched: results/classifier/summary.json")
    out_dir = tmp_path / "classifier_rerun"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "classifier.py"),
            "--features",
            "clr",
            "--batch-correct",
            "--n-repeats",
            "3",  # far fewer than the committed 200; LOFO means don't depend on this
            "--data-root",
            str(DATA_ROOT),
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fresh = json.loads((out_dir / "summary.json").read_text())
    main = json.loads(COMMITTED_SUMMARY.read_text())

    lofo_keys = [
        "full",
        "shared_response_global_pool",
        "shared_response",
        "disease_specific_global_pool",
        "disease_specific",
        "shared_response_clr",
        "shared_response_clr_center",
        "shared_response_clr_combat",
        "shared_response_clr_combat_cov",
    ]
    for key in lofo_keys:
        assert fresh[key]["mean_auc_roc"] == main[key]["mean_auc_roc"], (
            f"{key}: fresh={fresh[key]['mean_auc_roc']!r} "
            f"main={main[key]['mean_auc_roc']!r}"
        )

    # The headline numbers, stated explicitly (not just "matches main"):
    assert main["shared_response_clr"]["mean_auc_roc"] == pytest.approx(0.635, abs=5e-4)
    assert main["shared_response_clr_combat"]["mean_auc_roc"] == pytest.approx(
        0.647, abs=5e-4
    )
