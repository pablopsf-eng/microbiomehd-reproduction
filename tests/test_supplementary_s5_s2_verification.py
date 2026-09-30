"""Independent numerical re-derivation of file-S5.effects.txt and
file-S2.disease_specific_genera.txt against file-S1.qvalues.txt -- these two
files were only read/checksum-verified by a prior review, never re-derived
numerically.

file-S5: each cell is
``log2(mean_abundance_in_cases / mean_abundance_in_controls)``, with the
table's own global max/min used as a fill value when one group's mean is
exactly 0. ``docs/reproduction-m2.md``/``mbhd.abundance.to_genus_abundance``'s
own docstring claim this project's from-scratch genus-abundance table
reproduces file-S5's finite, non-fill cdi_schubert values to ~1e-12. This
module re-derives that independently: it builds the genus x sample relative
abundance table with ``mbhd.abundance`` (already independently verified
against file-S1 in ``test_differential_m2_verification.py``), then computes
log2(case_mean / control_mean) itself -- never importing
``mbhd.differential`` at all, so the log2 fold-change formula under test is
never borrowed from the module whose claim is being checked.

file-S2: a genus is listed for a disease
(>= 3 independent datasets) if it is significant in the same direction in at
least 2 of that disease's datasets, net of cancellation. Spot-checked here
against every genus file-S2 lists for "ob" (5 independent datasets), using
only file-S1's real signed q-values and the significance rule
(abs(q) < 0.05, exact 0.0 not significant).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mbhd.abundance import analysis_samples, filter_table, to_genus_abundance
from mbhd.datasets import load_dataset_info, metadata_path, otu_table_path, study_dir
from mbhd.io import DISEASE_STATE_COLUMN, encoding_for, load_metadata, load_otu_table
from mbhd.labels import (
    resolved_case_control,
    restrict_to_condition,
    restrict_to_sample_type,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _genus_means(dataset_id: str) -> tuple[pd.Series, pd.Series]:
    """control_mean, case_mean per genus, via mbhd.abundance only (no
    mbhd.differential import)."""
    index = load_dataset_info(DATA_ROOT / "dataset_info.yaml")
    folder = study_dir(index, dataset_id, DATA_ROOT)
    metadata = load_metadata(
        metadata_path(dataset_id, folder), encoding=encoding_for(dataset_id)
    )
    metadata = restrict_to_sample_type(dataset_id, metadata)
    metadata = restrict_to_condition(dataset_id, metadata)
    otu_table = load_otu_table(otu_table_path(dataset_id, folder))
    sample_ids = analysis_samples(metadata, otu_table.columns)
    filtered = filter_table(otu_table, sample_ids)
    genus_abundance = to_genus_abundance(filtered)

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
    return genus_abundance[control_ids].mean(axis=1), genus_abundance[case_ids].mean(
        axis=1
    )


def test_file_s5_log2_fold_change_matches_published_finite_non_fill_cells():
    _require(DATA_ROOT / "file-S1.qvalues.txt", DATA_ROOT / "file-S5.effects.txt")
    dataset_id = "cdi_schubert"
    control_mean, case_mean = _genus_means(dataset_id)
    my_log2fc = np.log2(
        case_mean / control_mean
    )  # own formula, inf/-inf/nan at zero means

    s5 = pd.read_csv(DATA_ROOT / "file-S5.effects.txt", sep="\t", index_col=0)
    published = s5[dataset_id].dropna()
    fill_max, fill_min = (
        s5.to_numpy(dtype=float)
        .flatten()[~np.isnan(s5.to_numpy(dtype=float).flatten())]
        .max(),
        s5.to_numpy(dtype=float)
        .flatten()[~np.isnan(s5.to_numpy(dtype=float).flatten())]
        .min(),
    )

    common = my_log2fc.index.intersection(published.index)
    assert set(published.index) <= set(
        my_log2fc.index
    )  # every published genus is one we computed

    diffs = []
    for genus in common:
        mine, pub = my_log2fc[genus], published[genus]
        if np.isfinite(mine) and pub not in (fill_max, fill_min):
            diffs.append(abs(mine - pub))
    diffs = np.array(diffs)
    assert (
        len(diffs) == 92
    )  # matches mbhd.abundance.to_genus_abundance's own docstring claim
    assert diffs.max() < 1e-10


def test_file_s5_significant_genera_have_correct_sign_where_real():
    """For every genus file-S1 calls significant for cdi_schubert: if
    file-S5 has a real (non-fill) value, its sign must agree with the
    q-value's sign; if it has a fill value instead, that must be because the
    corresponding group mean is genuinely 0 (own computation, not S5's own
    claim), and the fill's sign must still point the documented direction."""
    _require(DATA_ROOT / "file-S1.qvalues.txt", DATA_ROOT / "file-S5.effects.txt")
    dataset_id = "cdi_schubert"
    control_mean, case_mean = _genus_means(dataset_id)

    s1 = pd.read_csv(DATA_ROOT / "file-S1.qvalues.txt", sep="\t", index_col=0)
    s5 = pd.read_csv(DATA_ROOT / "file-S5.effects.txt", sep="\t", index_col=0)
    q = s1[dataset_id].dropna()
    significant = q[(q != 0.0) & (q.abs() < 0.05)]

    flat = s5.to_numpy(dtype=float).flatten()
    flat = flat[~np.isnan(flat)]
    fill_max, fill_min = flat.max(), flat.min()

    fc = s5[dataset_id]
    n_real_checked = 0
    n_fill = 0
    for genus, qval in significant.items():
        assert genus in fc.index and not pd.isna(fc[genus]), (
            f"{genus}: significant in file-S1 but missing/blank in file-S5"
        )
        value = fc[genus]
        if value in (fill_max, fill_min):
            n_fill += 1
            # Fill direction must match the q-value's direction, and the
            # underlying group mean it substitutes for must genuinely be 0.
            if value == fill_max:
                assert qval > 0
                assert control_mean[genus] == 0.0 and case_mean[genus] > 0.0
            else:
                assert qval < 0
                assert case_mean[genus] == 0.0 and control_mean[genus] > 0.0
        else:
            n_real_checked += 1
            assert (qval > 0) == (value > 0), f"{genus}: sign disagreement"

    assert n_real_checked == 63
    assert n_fill == 10


def test_file_s2_ob_genera_agree_with_file_s1_direction_in_at_least_two_datasets():
    """Every genus file-S2 lists under "ob" must be significant in the same
    direction in >= 2 of the 5 ob_* datasets, per file-S1's own signed
    q-values -- computed here from scratch, mbhd.shared_response never
    imported."""
    _require(
        DATA_ROOT / "file-S1.qvalues.txt",
        DATA_ROOT / "file-S2.disease_specific_genera.txt",
    )
    ob_datasets = ["ob_goodrich", "ob_ross", "ob_turnbaugh", "ob_zhu", "ob_zupancic"]
    s1 = pd.read_csv(DATA_ROOT / "file-S1.qvalues.txt", sep="\t", index_col=0)
    s2 = pd.read_csv(
        DATA_ROOT / "file-S2.disease_specific_genera.txt", sep="\t", index_col=0
    )

    ob = s2["ob"].dropna()
    ob = ob[ob.astype(str).str.strip() != ""]
    assert len(ob) > 0

    for genus, label in ob.items():
        assert label in ("health", "disease")
        row = s1.loc[genus, ob_datasets]
        significant = row[(row != 0.0) & (row.abs() < 0.05)]
        n_health = int(
            (significant < 0).sum()
        )  # negative = higher in controls = health
        n_disease = int((significant > 0).sum())
        if label == "health":
            assert n_health >= 2
            assert (
                n_disease == 0
            )  # net-association rule: no mixed direction survives labelling
        else:
            assert n_disease >= 2
            assert n_health == 0
