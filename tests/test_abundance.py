"""Tests for mbhd.abundance. Fully synthetic -- no data or network dependency.

The real end-to-end check (filtering + collapse reproducing file-S1's actual
genus sets) lives in tests/test_differential.py. This file covers each
function's own contract in isolation, with hand-computed expected values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbhd.abundance import (
    ZeroReadSampleError,
    analysis_samples,
    filter_table,
    genus_of,
    to_genus_abundance,
)

# --------------------------------------------------------------------------
# analysis_samples
# --------------------------------------------------------------------------


def _metadata(rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"DiseaseState": rows})


def test_analysis_samples_includes_blank_disease_state() -> None:
    """Mirrors crc_zeller's real evidence (module docstring): a blank
    DiseaseState sample still has read counts that belong in the
    prevalence/read-count denominator, even though it contributes no group
    membership -- excluding it was tried and disproven against real data."""
    metadata = _metadata({"S1": "H", "S2": "CDI", "S3": ""})
    assert analysis_samples(metadata, otu_sample_ids={"S1", "S2", "S3"}) == [
        "S1",
        "S2",
        "S3",
    ]


def test_analysis_samples_includes_a_third_arm_not_in_any_comparison() -> None:
    """Mirrors crc_baxter's real 'nonCRC' arm: a non-blank DiseaseState value
    that no dataset ID's CASE_CONTROL entry ever refers to must still count
    -- it is part of the study's sample set for filtering purposes, not the
    two-group comparison."""
    metadata = _metadata({"S1": "H", "S2": "CRC", "S3": "nonCRC"})
    assert analysis_samples(metadata, otu_sample_ids={"S1", "S2", "S3"}) == [
        "S1",
        "S2",
        "S3",
    ]


def test_analysis_samples_only_counts_the_intersection() -> None:
    """S3 has a real label but is missing from the OTU table -- excluded."""
    metadata = _metadata({"S1": "H", "S2": "CDI", "S3": "CDI"})
    assert analysis_samples(metadata, otu_sample_ids={"S1", "S2"}) == ["S1", "S2"]


# --------------------------------------------------------------------------
# filter_table
# --------------------------------------------------------------------------


def test_filter_table_drops_low_read_samples() -> None:
    """S3 has 5 total reads, under the 100 default -- its whole column drops,
    which must not affect the surviving samples' prevalence denominator."""
    table = pd.DataFrame(
        {"S1": [50, 60], "S2": [50, 60], "S3": [2, 3]},
        index=["otuA", "otuB"],
    )
    result = filter_table(table, ["S1", "S2", "S3"], min_sample_reads=100)
    assert list(result.columns) == ["S1", "S2"]


def test_filter_table_drops_low_read_otus() -> None:
    """otuB has 5 total reads across surviving samples, under the 10 default."""
    table = pd.DataFrame(
        {"S1": [50, 3], "S2": [60, 2]},
        index=["otuA", "otuB"],
    )
    result = filter_table(table, ["S1", "S2"], min_sample_reads=0, min_otu_reads=10)
    assert list(result.index) == ["otuA"]


def test_filter_table_drops_low_prevalence_otus() -> None:
    """otuB is present (nonzero) in only 1 of 4 samples: 25% for min_otu_reads
    but the default min_prevalence=0.01 threshold is what actually matters
    here -- raise it to 0.5 to exercise the drop."""
    table = pd.DataFrame(
        {"S1": [10, 10], "S2": [10, 0], "S3": [10, 0], "S4": [10, 0]},
        index=["otuA", "otuB"],
    )
    result = filter_table(
        table, ["S1", "S2", "S3", "S4"], min_sample_reads=0, min_prevalence=0.5
    )
    assert list(result.index) == ["otuA"]


def test_filter_table_boundary_is_inclusive() -> None:
    """Exactly at each threshold must be kept, not dropped ('fewer than X
    removed' means 'X or more kept')."""
    table = pd.DataFrame({"S1": [10], "S2": [10]}, index=["otuA"])
    result = filter_table(
        table, ["S1", "S2"], min_sample_reads=10, min_otu_reads=20, min_prevalence=1.0
    )
    assert list(result.columns) == ["S1", "S2"]
    assert list(result.index) == ["otuA"]


def test_filter_table_ignores_sample_ids_absent_from_the_table() -> None:
    table = pd.DataFrame({"S1": [50]}, index=["otuA"])
    result = filter_table(table, ["S1", "S2_not_in_table"], min_sample_reads=0)
    assert list(result.columns) == ["S1"]


# --------------------------------------------------------------------------
# genus_of
# --------------------------------------------------------------------------


def test_genus_of_drops_species_and_denovo_fields() -> None:
    lineage = (
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;"
        "f__Lachnospiraceae;g__Blautia;s__;d__denovo5393"
    )
    assert genus_of(lineage) == (
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;"
        "f__Lachnospiraceae;g__Blautia"
    )


def test_genus_of_returns_none_for_unannotated() -> None:
    lineage = (
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;"
        "f__Ruminococcaceae;g__;s__;d__denovo15568"
    )
    assert genus_of(lineage) is None


def test_genus_of_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        genus_of("k__Bacteria;p__Firmicutes")


# --------------------------------------------------------------------------
# to_genus_abundance
# --------------------------------------------------------------------------


def test_to_genus_abundance_normalises_before_dropping_unannotated() -> None:
    """The regression fixture for a central M2 finding: an
    unannotated OTU's reads must stay in the per-sample denominator.

    S1 has otuA=30 (g__Foo), otuB=70 (unannotated) -> total 100.
    otuA's relative abundance is 30/100 = 0.3, NOT 30/30 = 1.0 (which is
    what dropping the unannotated OTU before normalising would give).
    """
    table = pd.DataFrame(
        {"S1": [30, 70]},
        index=[
            "k__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__Bacteria;p__X;c__X;o__X;f__X;g__;s__;d__denovo2",
        ],
    )
    result = to_genus_abundance(table)
    assert list(result.index) == ["k__Bacteria;p__X;c__X;o__X;f__X;g__Foo"]
    assert result.loc["k__Bacteria;p__X;c__X;o__X;f__X;g__Foo", "S1"] == pytest.approx(
        0.3
    )


def test_to_genus_abundance_collapses_multiple_otus_to_one_genus() -> None:
    """Two distinct de-novo OTUs assigned the same genus must sum, not
    overwrite one another."""
    table = pd.DataFrame(
        {"S1": [10, 20]},
        index=[
            "k__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo2",
        ],
    )
    result = to_genus_abundance(table)
    assert result.shape[0] == 1
    assert result.iloc[0, 0] == pytest.approx(30 / 30)


def test_to_genus_abundance_raises_on_zero_read_sample() -> None:
    table = pd.DataFrame(
        {"S1": [10, 0], "S2": [0, 0]},
        index=[
            "k__Bacteria;p__X;c__X;o__X;f__X;g__Foo;s__;d__denovo1",
            "k__Bacteria;p__X;c__X;o__X;f__X;g__Bar;s__;d__denovo2",
        ],
    )
    with pytest.raises(ZeroReadSampleError, match="S2"):
        to_genus_abundance(table)
