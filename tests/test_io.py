"""Tests for mbhd.io.

Split by dependency, matching the tests/test_manifest.py convention:
synthetic fixtures (tmp_path) cover the loaders' contracts and edge cases with
no dependency; real-data tests (data/raw/, guarded by _require()) confirm the
loaders actually work against every analysed study's real files, not just a
hand-built example.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from mbhd.datasets import (
    analysed_dataset_ids,
    entry_for,
    load_dataset_info,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.io import (
    DISEASE_STATE_COLUMN,
    DuplicateSampleIdError,
    EmptyFileError,
    MissingColumnError,
    encoding_for,
    load_metadata,
    load_otu_table,
)

DATA_RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


# --------------------------------------------------------------------------
# load_otu_table -- synthetic, hand-computed.
# --------------------------------------------------------------------------


def test_load_otu_table_hand_computed(tmp_path: pathlib.Path) -> None:
    """3 OTUs x 4 samples, values chosen so per-sample sums are computable by
    hand: S1=1+0+2=3, S2=0+5+0=5, S3=2+2+2=6, S4=0+0+0=0."""
    path = tmp_path / "otu_table.100.denovo.rdp_assigned"
    path.write_text(
        "OTU_ID\tS1\tS2\tS3\tS4\n"
        "k__Bacteria;g__A;d__denovo1\t1\t0\t2\t0\n"
        "k__Bacteria;g__B;d__denovo2\t0\t5\t2\t0\n"
        "k__Bacteria;g__;d__denovo3\t2\t0\t2\t0\n"
    )
    table = load_otu_table(path)

    assert table.shape == (3, 4)
    assert list(table.columns) == ["S1", "S2", "S3", "S4"]
    assert table.sum(axis=0).to_dict() == {"S1": 3, "S2": 5, "S3": 6, "S4": 0}
    assert table.loc["k__Bacteria;g__;d__denovo3", "S1"] == 2


def test_load_otu_table_handles_blank_header_cell(tmp_path: pathlib.Path) -> None:
    """The real rdp_assigned variant's header starts with a tab, not a
    column name (unlike the .100.denovo variant's 'OTU_ID')."""
    path = tmp_path / "otu_table.rdp_assigned"
    path.write_text("\tS1\tS2\nk__Bacteria;g__A;d__denovo1\t3\t4\n")
    table = load_otu_table(path)
    assert list(table.columns) == ["S1", "S2"]
    assert table.loc["k__Bacteria;g__A;d__denovo1", "S2"] == 4


def test_load_otu_table_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_otu_table(tmp_path / "does_not_exist.txt")


def test_load_otu_table_raises_on_empty_data(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "otu_table.txt"
    path.write_text("OTU_ID\tS1\tS2\n")
    with pytest.raises(EmptyFileError):
        load_otu_table(path)


def test_load_otu_table_raises_on_duplicate_sample_id(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "otu_table.txt"
    path.write_text("OTU_ID\tS1\tS1\nk__Bacteria;g__A;d__1\t1\t2\n")
    with pytest.raises(DuplicateSampleIdError, match="S1"):
        load_otu_table(path)


# --------------------------------------------------------------------------
# load_metadata -- synthetic, hand-computed.
# --------------------------------------------------------------------------


def test_load_metadata_hand_computed(tmp_path: pathlib.Path) -> None:
    """4 samples, 2 case (CASE) and 2 control (H) -- counted by hand."""
    path = tmp_path / "metadata.txt"
    path.write_text(
        "sample_id\tage\tDiseaseState\n"
        "S1\t30\tH\n"
        "S2\t40\tH\n"
        "S3\t50\tCASE\n"
        "S4\t60\tCASE\n"
    )
    metadata = load_metadata(path)
    assert metadata.shape == (4, 2)
    counts = metadata[DISEASE_STATE_COLUMN].value_counts().to_dict()
    assert counts == {"H": 2, "CASE": 2}


def test_load_metadata_index_is_first_column_whatever_its_name(
    tmp_path: pathlib.Path,
) -> None:
    """Real files vary the ID column's header (sample_id, #SampleID, blank,
    Sample_Name_s, ...) -- the loader must use "first column", not a fixed name."""
    path = tmp_path / "metadata.txt"
    path.write_text("#SampleID\tDiseaseState\nM1\tT1D\n")
    metadata = load_metadata(path)
    assert list(metadata.index) == ["M1"]


def test_load_metadata_index_stays_string_for_all_digit_sample_ids(
    tmp_path: pathlib.Path,
) -> None:
    """Regression test: crc_baxter's sample IDs are all-digit (e.g.
    2005650). Left to pandas' own type inference, the index parses as int64
    while an OTU table's columns (always strings, taken from a header row)
    do not -- so every ID silently fails to intersect despite being
    identical. Caught only by cross-checking against Table 1 (0 of 490
    samples intersected), not by any exception."""
    path = tmp_path / "metadata.txt"
    path.write_text("sample_id\tDiseaseState\n2005650\tH\n2003650\tCASE\n")
    metadata = load_metadata(path)
    assert all(isinstance(sample_id, str) for sample_id in metadata.index)
    assert set(metadata.index) == {"2005650", "2003650"}


def test_load_metadata_decodes_with_the_given_encoding(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "metadata.txt"
    # A real trap: a latin-1 registered-trademark byte (0xae, "®") that is not
    # valid UTF-8 -- exactly what cdi_schubert's real file contains.
    content = "sample_id\tnote\tDiseaseState\nS1\tPowerSoil\xae kit\tH\n"
    path.write_bytes(content.encode("latin-1"))

    with pytest.raises(UnicodeDecodeError):
        load_metadata(path, encoding="utf-8")

    metadata = load_metadata(path, encoding="latin-1")
    assert metadata.loc["S1", "note"] == "PowerSoil\xae kit"


def test_load_metadata_raises_on_missing_disease_state_column(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "metadata.txt"
    path.write_text("sample_id\tdisease_stat\nS1\tCase\n")
    with pytest.raises(MissingColumnError, match="DiseaseState"):
        load_metadata(path)


def test_load_metadata_raises_on_duplicate_sample_id(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "metadata.txt"
    path.write_text("sample_id\tDiseaseState\nS1\tH\nS1\tCASE\n")
    with pytest.raises(DuplicateSampleIdError, match="S1"):
        load_metadata(path)


def test_load_metadata_raises_on_empty_data(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "metadata.txt"
    path.write_text("sample_id\tDiseaseState\n")
    with pytest.raises(EmptyFileError):
        load_metadata(path)


def test_load_metadata_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "does_not_exist.txt")


# --------------------------------------------------------------------------
# encoding_for
# --------------------------------------------------------------------------


def test_encoding_for_defaults_to_utf8() -> None:
    assert encoding_for("some_dataset_never_seen") == "utf-8"


def test_encoding_for_known_overrides() -> None:
    assert encoding_for("cdi_schubert") == "latin-1"
    assert encoding_for("noncdi_schubert") == "latin-1"
    assert encoding_for("crc_zeller") == "latin-1"


# --------------------------------------------------------------------------
# Real data: an independent cross-check on one study, and a full sweep.
# --------------------------------------------------------------------------

_T1D_DIR = DATA_RAW / "t1d_mejialeon_results"


def test_t1d_mejialeon_sample_count_matches_wc_l() -> None:
    """Independent cross-check: the loader's sample count must match a count
    taken a different way, so a loader bug can't be masked by two errors
    cancelling out."""
    _require(_T1D_DIR)
    metadata_file = metadata_path("t1d_mejialeon", _T1D_DIR)
    df = load_metadata(metadata_file, encoding=encoding_for("t1d_mejialeon"))

    # wc -l counts the header too, so real sample rows = line count - 1.
    line_count = sum(1 for _ in open(metadata_file, encoding="latin-1"))
    assert df.shape[0] == line_count - 1


def test_all_30_analysed_datasets_load_without_error() -> None:
    """Every analysed dataset's metadata and OTU table load cleanly end to
    end through the real, resolved paths -- not just the two studies used
    while designing the loaders."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    failures = []
    for dataset_id in analysed_dataset_ids(index):
        folder = DATA_RAW / entry_for(index, dataset_id).folder
        if not folder.is_dir():
            pytest.skip(f"data not fetched: {folder.name}")
        try:
            sd = study_dir(index, dataset_id, DATA_RAW)
            metadata = load_metadata(
                metadata_path(dataset_id, sd), encoding=encoding_for(dataset_id)
            )
            otu = load_otu_table(otu_table_path(dataset_id, sd))
        except Exception as exc:  # broad on purpose: collecting per-dataset failures
            failures.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            continue
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(otu, pd.DataFrame)

    assert not failures, "loader failed on:\n" + "\n".join(failures)
