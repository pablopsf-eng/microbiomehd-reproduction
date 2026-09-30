"""Tests for mbhd.differential.differential_abundance.

Two layers: a real end-to-end check
against the published file-S1/file-S5 ground truth for the four chosen
subset datasets (skip-guarded, matching the project's established
_require()/skip convention -- these numbers ARE the regression target, a
drift from them is a failure to investigate, not a tolerance to widen), and
fully synthetic tests of differential_abundance's own contract (error
handling, output shape) with no data dependency.

M4 adds ``abundance_transform``: its default ``None`` must reproduce today's numbers
byte-identically before
anything CLR-related is trusted, and a non-``None`` transform must visibly
change the output.
"""

from __future__ import annotations

import pathlib
from functools import partial

import numpy as np
import pandas as pd
import pytest
import yaml

from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import load_dataset_info
from mbhd.differential import NoSurvivingSamplesError, differential_abundance

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"
FILE_S1 = DATA_RAW / "file-S1.qvalues.txt"
FILE_S5 = DATA_RAW / "file-S5.effects.txt"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


# --------------------------------------------------------------------------
# The real reproduction check -- M2's four subset datasets.
#
# Regression targets independently re-derived in this session directly from
# the real downloaded archive (not copied from an earlier plan
# without re-checking): every genus in file-S1 reproduced, every significance
# call and sign agreeing, q-values matching to floating-point noise, and
# file-S5's finite log2 fold-changes matching to ~1e-12.
# --------------------------------------------------------------------------

_SUBSET = {
    "cdi_schubert": {"n_genera": 120, "n_pub_sig": 73, "n_finite_s5": 92},
    "crc_baxter": {"n_genera": 163, "n_pub_sig": 16, "n_finite_s5": 113},
    "par_scheperjans": {"n_genera": 75, "n_pub_sig": 2, "n_finite_s5": 63},
    "t1d_mejialeon": {"n_genera": 66, "n_pub_sig": 0, "n_finite_s5": 52},
}


@pytest.mark.parametrize("dataset_id", sorted(_SUBSET))
def test_reproduces_file_s1_significance_and_sign_exactly(dataset_id: str) -> None:
    _require(DATASET_INFO, FILE_S1, FILE_S5)
    expected = _SUBSET[dataset_id]

    index = load_dataset_info(DATASET_INFO)
    result = differential_abundance(dataset_id, index, DATA_RAW)

    published = pd.read_csv(FILE_S1, sep="\t", index_col=0)[dataset_id].dropna()

    assert set(result.index) == set(published.index), (
        "genus set does not match file-S1 -- filtering or genus collapse has regressed"
    )
    assert len(result) == expected["n_genera"]

    def is_significant(signed_q: float) -> bool:
        return abs(signed_q) < 0.05 and signed_q != 0.0

    n_pub_sig = 0
    for genus in published.index:
        ours = result.loc[genus, "signed_q_value"]
        pub = published[genus]
        our_sig, pub_sig = is_significant(ours), is_significant(pub)
        assert our_sig == pub_sig, (
            f"{dataset_id}/{genus}: significance disagrees "
            f"(ours q={ours!r}, published q={pub!r})"
        )
        if pub_sig:
            n_pub_sig += 1
            assert np.sign(ours) == np.sign(pub), (
                f"{dataset_id}/{genus}: sign disagrees "
                f"(ours q={ours!r}, published q={pub!r})"
            )
            # Regression floor, not the achievable precision (~5e-13): a
            # generous margin so this test tracks a real drift, not noise.
            assert abs(ours - pub) < 1e-6

    assert n_pub_sig == expected["n_pub_sig"]


@pytest.mark.parametrize("dataset_id", sorted(_SUBSET))
def test_reproduces_file_s5_effect_sizes_where_finite(dataset_id: str) -> None:
    """Effect sizes are REPORTED, not required for the pass/fail gate
    (the pass/fail criterion is significance-call and sign agreement; effect
    sizes are reported) -- but where
    both means are nonzero (the log2 fold-change is finite on both sides),
    the two independent computations (ours from scratch, the authors'
    published) must agree to numeric precision."""
    _require(DATASET_INFO, FILE_S1, FILE_S5)
    expected = _SUBSET[dataset_id]

    index = load_dataset_info(DATASET_INFO)
    result = differential_abundance(dataset_id, index, DATA_RAW)
    published = pd.read_csv(FILE_S5, sep="\t", index_col=0)[dataset_id].dropna()

    finite = result[np.isfinite(result["log2_fold_change"])]
    common = sorted(set(finite.index) & set(published.index))
    assert len(common) == expected["n_finite_s5"]

    diffs = [abs(finite.loc[g, "log2_fold_change"] - published[g]) for g in common]
    assert max(diffs) < 1e-6


def test_t1d_mejialeon_is_a_null_control() -> None:
    """A genuine null (8 controls vs 21 cases): the paper reports zero
    significant genera, and so must this implementation -- a reproduction
    that only ever confirms positives proves much less."""
    _require(DATASET_INFO, FILE_S1)
    index = load_dataset_info(DATASET_INFO)
    result = differential_abundance("t1d_mejialeon", index, DATA_RAW)
    n_significant = (
        (result["signed_q_value"].abs() < 0.05) & (result["signed_q_value"] != 0.0)
    ).sum()
    assert n_significant == 0


def test_output_reports_class_imbalance() -> None:
    """n_controls/n_cases must be visible in the output (class
    imbalance must be reported, not hidden) -- t1d_mejialeon is the most
    imbalanced subset dataset."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    result = differential_abundance("t1d_mejialeon", index, DATA_RAW)
    assert (result["n_controls"] == 8).all()
    assert (result["n_cases"] == 21).all()


# --------------------------------------------------------------------------
# M4: abundance_transform's default must not move M2's numbers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_id", sorted(_SUBSET))
def test_abundance_transform_default_matches_omitting_it_on_real_data(
    dataset_id: str,
) -> None:
    """Regression gate (M4): calling with the new keyword
    explicitly set to its own default must not move a single one of M2's
    already-published-value-matched numbers, on the real archive."""
    _require(DATASET_INFO, FILE_S1)
    index = load_dataset_info(DATASET_INFO)
    without_kwarg = differential_abundance(dataset_id, index, DATA_RAW)
    with_default = differential_abundance(
        dataset_id, index, DATA_RAW, abundance_transform=None
    )
    pd.testing.assert_frame_equal(without_kwarg, with_default)


# --------------------------------------------------------------------------
# Synthetic contract tests -- no real data dependency.
# --------------------------------------------------------------------------


def _build_synthetic_archive(
    tmp_path: pathlib.Path,
    dataset_id: str,
    folder: str,
    otu_table: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[dict, pathlib.Path]:
    """A minimal fake data_root with the real archive's directory shape,
    matching the pattern already used in tests/test_datasets.py."""
    data_root = tmp_path / "data_root"
    study_dir = data_root / folder
    (study_dir / "RDP").mkdir(parents=True)

    otu_filename = f"{dataset_id}.otu_table.100.denovo.rdp_assigned"
    otu_table.to_csv(study_dir / "RDP" / otu_filename, sep="\t")
    (study_dir / "summary_file.txt").write_text(
        f"DATASET_ID\t{dataset_id}\n"
        "#16S_start\n"
        f"OTU_TABLE_RDP\t{otu_filename}\n"
        "PROCESSED\tTrue\n"
        "#16S_end\n"
    )
    metadata.to_csv(study_dir / f"{dataset_id}.metadata.txt", sep="\t")

    dataset_info_path = data_root / "dataset_info.yaml"
    dataset_info_path.write_text(yaml.dump({dataset_id: {"folder": folder}}))
    index = load_dataset_info(dataset_info_path)
    return index, data_root


def test_raises_when_a_group_has_no_surviving_samples(tmp_path: pathlib.Path) -> None:
    """par_scheperjans's real CASE_CONTROL entry is H vs PAR -- a synthetic
    archive with only H samples must raise, not silently run
    Kruskal-Wallis on one group or crash inside it."""
    otu = pd.DataFrame(
        {"S1": [50, 5], "S2": [40, 6]},
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    metadata = pd.DataFrame({"DiseaseState": {"S1": "H", "S2": "H"}})
    index, data_root = _build_synthetic_archive(
        tmp_path, "par_scheperjans", "par_scheperjans_results", otu, metadata
    )

    with pytest.raises(NoSurvivingSamplesError, match="par_scheperjans"):
        differential_abundance(
            "par_scheperjans",
            index,
            data_root,
            min_sample_reads=0,
            min_otu_reads=0,
            min_prevalence=0.0,
        )


def test_output_shape_and_columns(tmp_path: pathlib.Path) -> None:
    otu = pd.DataFrame(
        {
            "S1": [10, 0],
            "S2": [12, 0],
            "S3": [0, 10],
            "S4": [0, 12],
        },
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    metadata = pd.DataFrame(
        {"DiseaseState": {"S1": "H", "S2": "H", "S3": "PAR", "S4": "PAR"}}
    )
    index, data_root = _build_synthetic_archive(
        tmp_path, "par_scheperjans", "par_scheperjans_results", otu, metadata
    )

    result = differential_abundance(
        "par_scheperjans",
        index,
        data_root,
        min_sample_reads=0,
        min_otu_reads=0,
        min_prevalence=0.0,
    )

    assert list(result.columns) == [
        "p_value",
        "q_value",
        "signed_q_value",
        "control_mean",
        "case_mean",
        "log2_fold_change",
        "n_controls",
        "n_cases",
    ]
    assert result.index.name == "genus"
    assert len(result) == 2
    assert (result["n_controls"] == 2).all()
    assert (result["n_cases"] == 2).all()
    # g__Foo is entirely absent from the PAR group and entirely present in H
    # -- a clean, hand-verifiable case: control_mean=1.0, case_mean=0.0.
    foo = "k__B;p__X;c__X;o__X;f__X;g__Foo"
    assert result.loc[foo, "control_mean"] == pytest.approx(1.0)
    assert result.loc[foo, "case_mean"] == pytest.approx(0.0)
    assert result.loc[foo, "log2_fold_change"] == float("-inf")


# --------------------------------------------------------------------------
# M4: abundance_transform actually changes the matrix fed to
# Kruskal-Wallis, using the CLR path as the concrete example.
# --------------------------------------------------------------------------


def test_abundance_transform_applies_clr_before_kruskal_wallis(
    tmp_path: pathlib.Path,
) -> None:
    """Same synthetic archive as test_output_shape_and_columns (which has a
    real exact-zero relative-abundance cell for g__Bar in the H group), but
    with the CLR path applied. With exactly two genera, every sample's CLR
    values are exact negatives of each other (n=2 centered log-ratio) --
    control_mean/case_mean for the two genera must sum to ~0, a
    hand-verifiable signature that CLR (not the Euclidean untransformed
    matrix) is what fed Kruskal-Wallis."""
    otu = pd.DataFrame(
        {
            "S1": [10, 0],
            "S2": [12, 0],
            "S3": [0, 10],
            "S4": [0, 12],
        },
        index=[
            "k__B;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__B;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    metadata = pd.DataFrame(
        {"DiseaseState": {"S1": "H", "S2": "H", "S3": "PAR", "S4": "PAR"}}
    )
    index, data_root = _build_synthetic_archive(
        tmp_path, "par_scheperjans", "par_scheperjans_results", otu, metadata
    )

    result = differential_abundance(
        "par_scheperjans",
        index,
        data_root,
        min_sample_reads=0,
        min_otu_reads=0,
        min_prevalence=0.0,
        abundance_transform=partial(clr, pseudocount_fn=multiplicative_replacement),
    )

    foo = "k__B;p__X;c__X;o__X;f__X;g__Foo"
    bar = "k__B;p__X;c__X;o__X;f__X;g__Bar"
    assert result.loc[foo, "control_mean"] + result.loc[bar, "control_mean"] == (
        pytest.approx(0.0, abs=1e-9)
    )
    assert result.loc[foo, "case_mean"] + result.loc[bar, "case_mean"] == (
        pytest.approx(0.0, abs=1e-9)
    )
    # The untransformed (Euclidean) control_mean(g__Foo) would be exactly
    # 1.0 (g__Foo is all of the H group's relative abundance); CLR's own
    # zero-replacement + log-ratio centering makes it 0.0 instead --
    # a direct, hand-verifiable signature that the transform actually ran.
    assert result.loc[foo, "control_mean"] == pytest.approx(0.0, abs=1e-9)
