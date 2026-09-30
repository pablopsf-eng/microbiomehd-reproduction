"""Independent verification pass for mbhd.abundance (M2 review).

Claim under test (mbhd.abundance's module docstring): crc_zeller has 13
blank-``DiseaseState`` samples in the
metadata/OTU-table intersection, and including them (rather than the
seemingly-more-conservative "only labelled samples" rule) is required to
reproduce all 193 of file-S1's published crc_zeller genera -- excluding them
loses exactly 7 genera.

This file re-derives the filtering and genus collapse from the raw archive
files directly with plain pandas, deliberately NOT calling
mbhd.abundance.analysis_samples/filter_table/to_genus_abundance, so that a
bug shared between the implementation and this check could not hide the
disagreement.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
STUDY = DATA_RAW / "crc_zeller_results"
METADATA_PATH = STUDY / "crc_zeller.metadata.txt"
OTU_PATH = STUDY / "RDP" / "crc_zeller.otu_table.100.denovo.rdp_assigned"
FILE_S1 = DATA_RAW / "file-S1.qvalues.txt"


def _require(*paths: pathlib.Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {[str(p) for p in missing]}")


def _genus_of(lineage: str) -> str | None:
    fields = lineage.split(";")
    if fields[-3] == "g__":
        return None
    return ";".join(fields[:-2])


def _filter_and_collapse(
    otu_table: pd.DataFrame,
    sample_ids: list[str],
    *,
    min_sample_reads: int = 100,
    min_otu_reads: int = 10,
    min_prevalence: float = 0.01,
) -> pd.DataFrame:
    present = [s for s in sample_ids if s in otu_table.columns]
    table = otu_table[present]

    table = table.loc[:, table.sum(axis=0) >= min_sample_reads]
    table = table.loc[table.sum(axis=1) >= min_otu_reads]
    n_samples = table.shape[1]
    table = table.loc[(table > 0).sum(axis=1) / n_samples >= min_prevalence]

    relative = table.div(table.sum(axis=0), axis=1)
    genus = pd.Series(table.index, index=table.index).map(_genus_of)
    annotated = relative.loc[genus.notna()].copy()
    annotated.index = genus.loc[genus.notna()].to_numpy()
    return annotated.groupby(level=0).sum()


def test_crc_zeller_has_exactly_13_blank_disease_state_samples_in_intersection() -> (
    None
):
    _require(METADATA_PATH, OTU_PATH)
    metadata = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        index_col=0,
        encoding="latin-1",
        dtype={0: str},
        keep_default_na=False,
    )
    otu_table = pd.read_csv(OTU_PATH, sep="\t", index_col=0)
    intersected = metadata.loc[metadata.index.isin(set(otu_table.columns))]
    blank = intersected.loc[intersected["DiseaseState"] == ""]
    assert len(blank) == 13


def test_crc_zeller_excluding_blank_disease_state_loses_exactly_7_genera() -> None:
    _require(METADATA_PATH, OTU_PATH, FILE_S1)
    metadata = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        index_col=0,
        encoding="latin-1",
        dtype={0: str},
        keep_default_na=False,
    )
    otu_table = pd.read_csv(OTU_PATH, sep="\t", index_col=0)
    intersected = metadata.loc[metadata.index.isin(set(otu_table.columns))]

    scope_including_blanks = intersected.index.tolist()
    scope_excluding_blanks = intersected.loc[
        intersected["DiseaseState"] != ""
    ].index.tolist()

    genus_all = _filter_and_collapse(otu_table, scope_including_blanks)
    genus_nonblank = _filter_and_collapse(otu_table, scope_excluding_blanks)

    lost = set(genus_all.index) - set(genus_nonblank.index)
    assert len(lost) == 7

    published = pd.read_csv(FILE_S1, sep="\t", index_col=0)["crc_zeller"].dropna()
    assert set(genus_all.index) == set(published.index)
    assert set(genus_nonblank.index) != set(published.index)
