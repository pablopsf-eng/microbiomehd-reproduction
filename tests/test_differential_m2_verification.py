"""Independent re-verification of the M2 headline claim: for the four
subset datasets, ``mbhd.differential.differential_abundance`` reproduces
``file-S1.qvalues.txt`` exactly, and its from-scratch Kruskal-Wallis +
Benjamini-Hochberg agrees with scipy/statsmodels on real data.

Ground truth:
- ``data/raw/file-S1.qvalues.txt`` (published, GitHub supp-files).
- ``scipy.stats.kruskal`` / ``statsmodels.stats.multitest.multipletests``
  as an independent reference implementation of the same statistics,
  applied directly to the same filtered genus-abundance table
  ``mbhd.differential.differential_abundance`` itself operates on.
- A fully hand-computable 3-vs-3, no-ties synthetic case for
  ``mbhd.stats.kruskal_wallis`` in isolation.

Skips (does not fail) if ``data/raw`` is not present -- matches the
convention already used by ``tests/test_supplementary_files.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

from mbhd.abundance import analysis_samples, filter_table, to_genus_abundance
from mbhd.datasets import load_dataset_info, metadata_path, otu_table_path, study_dir
from mbhd.differential import differential_abundance
from mbhd.io import DISEASE_STATE_COLUMN, encoding_for, load_metadata, load_otu_table
from mbhd.labels import (
    resolved_case_control,
    restrict_to_condition,
    restrict_to_sample_type,
)
from mbhd.stats import benjamini_hochberg, kruskal_wallis

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
TARGET_DATASETS = ["cdi_schubert", "crc_baxter", "par_scheperjans", "t1d_mejialeon"]


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def test_hand_computable_kruskal_wallis_matches_scipy_and_mbhd():
    """3 vs 3, no ties: H and p worked out by hand, cross-checked against
    scipy.stats.kruskal, then against mbhd.stats.kruskal_wallis."""
    control = [1, 2, 3]
    case = [4, 5, 6]
    # ranks 1..6, no ties -> R_control=6, R_case=15, N=6
    h_by_hand = (12 / (6 * 7)) * (6**2 / 3 + 15**2 / 3) - 3 * 7
    assert h_by_hand == pytest.approx(3.857142857142857, abs=1e-12)
    p_by_hand = math.erfc(math.sqrt(h_by_hand / 2))

    stat_scipy, p_scipy = kruskal(control, case)
    assert stat_scipy == pytest.approx(h_by_hand, abs=1e-12)
    assert p_scipy == pytest.approx(p_by_hand, abs=1e-12)

    h_mbhd, p_mbhd = kruskal_wallis(control, case)
    assert h_mbhd == pytest.approx(h_by_hand, abs=1e-12)
    assert p_mbhd == pytest.approx(p_scipy, abs=1e-12)


def test_benjamini_hochberg_matches_statsmodels_hand_vector():
    ps = [0.01, 0.02, 0.03, 0.5, 0.9]
    q_mbhd = benjamini_hochberg(ps)
    _, q_sm, _, _ = multipletests(ps, method="fdr_bh")
    assert np.max(np.abs(q_mbhd - q_sm)) < 1e-15


@pytest.mark.parametrize("dataset_id", TARGET_DATASETS)
def test_reproduces_file_s1_exactly(dataset_id: str):
    _require(DATA_ROOT / "dataset_info.yaml", DATA_ROOT / "file-S1.qvalues.txt")
    index = load_dataset_info(DATA_ROOT / "dataset_info.yaml")
    s1 = pd.read_csv(DATA_ROOT / "file-S1.qvalues.txt", sep="\t", index_col=0)
    published = s1[dataset_id].dropna()

    reproduced = differential_abundance(dataset_id, index, DATA_ROOT)
    repro_signed = reproduced["signed_q_value"]

    assert set(repro_signed.index) == set(published.index), (
        f"{dataset_id}: genus set mismatch"
    )
    diffs = (repro_signed.loc[published.index] - published).abs()
    assert diffs.max() < 1e-10, f"{dataset_id}: max signed-q diff {diffs.max():.3e}"

    def sig(q: pd.Series) -> pd.Series:
        return (q != 0.0) & (q.abs() < 0.05)

    assert (sig(repro_signed.loc[published.index]) == sig(published)).all(), (
        f"{dataset_id}: significance-call disagreement"
    )


@pytest.mark.parametrize("dataset_id", TARGET_DATASETS)
def test_kruskal_wallis_and_bh_agree_with_scipy_statsmodels_on_real_data(
    dataset_id: str,
):
    """Independent re-derivation, bypassing mbhd.stats entirely: rebuild the
    same filtered genus x sample abundance table mbhd.differential uses
    (pure I/O + filtering, not the statistics under test), then run
    scipy.stats.kruskal per genus and statsmodels' BH across the dataset,
    and compare against mbhd's own p/q values for every genus."""
    _require(DATA_ROOT / "dataset_info.yaml", DATA_ROOT / "file-S1.qvalues.txt")
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

    reproduced = differential_abundance(dataset_id, index, DATA_ROOT)
    genera = genus_abundance.index.tolist()

    p_scipy = np.array(
        [
            kruskal(
                genus_abundance.loc[g, control_ids].to_numpy(),
                genus_abundance.loc[g, case_ids].to_numpy(),
            )[1]
            for g in genera
        ]
    )
    p_mbhd = reproduced.loc[genera, "p_value"].to_numpy()

    finite = ~np.isnan(p_scipy)
    # scipy.stats.kruskal is documented (mbhd.stats module docstring) to
    # return nan on heavily-tied/degenerate real data; mbhd.stats clamps H
    # to be non-negative instead. Every scipy-finite p-value must still
    # agree with mbhd's own to float precision.
    assert np.max(np.abs(p_scipy[finite] - p_mbhd[finite])) < 1e-10

    # BH agreement using mbhd's own p-vector (isolates the BH step).
    _, q_sm, _, _ = multipletests(p_mbhd, method="fdr_bh")
    q_mbhd = reproduced.loc[genera, "q_value"].to_numpy()
    assert np.max(np.abs(q_sm - q_mbhd)) < 1e-10
