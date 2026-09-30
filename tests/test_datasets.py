"""Tests for mbhd.datasets.

Split by dependency, matching the tests/test_manifest.py convention: the real
committed dataset_info.yaml (data/raw/, gitignored, guarded by _require())
covers the count reconciliation and the folder-is-not-the-id distinction;
fully synthetic fixtures (tmp_path) cover malformed input and edge cases with
no dependency at all.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from mbhd.datasets import (
    DatasetFileError,
    DatasetInfo,
    UnknownDatasetError,
    analysed_dataset_ids,
    entry_for,
    load_dataset_info,
    load_summary_file,
    metadata_path,
    otu_table_path,
    study_dir,
)

DATA_RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


# --------------------------------------------------------------------------
# Against the real, fetched dataset_info.yaml.
# --------------------------------------------------------------------------


def test_load_dataset_info_finds_33_active_datasets() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert len(index) == 33


def test_hiv_dubourg_is_not_in_the_parsed_result() -> None:
    """Commented out in the yaml -- must not appear as an active dataset."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert "hiv_dubourg" not in index


def test_analysed_dataset_ids_is_30_and_excludes_the_documented_three() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    analysed = analysed_dataset_ids(index)
    assert len(analysed) == 30
    for excluded in ("crc_wu", "crc_zackular", "ob_escobar"):
        assert excluded not in analysed
        assert excluded in index  # still active, just not analysed


def test_folder_differs_from_dataset_id_for_documented_cases() -> None:
    """Folder names follow the senior author, not the dataset ID's first author."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert entry_for(index, "art_scher").folder == "ra_littman_results"
    assert entry_for(index, "ibd_morgan").folder == "ibd_huttenhower_results"
    assert entry_for(index, "ob_turnbaugh").folder == "ob_gordon_2008_v2_results"


def test_schubert_and_zhu_ids_share_one_folder_each() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert (
        entry_for(index, "cdi_schubert").folder
        == entry_for(index, "noncdi_schubert").folder
        == "cdi_schubert_results"
    )
    assert (
        entry_for(index, "nash_zhu").folder
        == entry_for(index, "ob_zhu").folder
        == "nash_ob_baker_results"
    )


def test_folders_are_an_exact_bijection_with_the_31_archives() -> None:
    """No folder is orphaned, and every folder maps back to a real dataset."""
    _require(DATASET_INFO)
    manifest_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "zenodo_840333_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    archive_stems = {
        e["key"][: -len(".tar.gz")]
        for e in manifest["files"]
        if e["key"].endswith(".tar.gz")
    }

    index = load_dataset_info(DATASET_INFO)
    folders = {entry.folder for entry in index.values()}

    assert folders == archive_stems


def test_entry_for_unknown_id_raises_named_error() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    with pytest.raises(UnknownDatasetError, match="not_a_real_dataset"):
        entry_for(index, "not_a_real_dataset")


def test_sample_size_in_yaml_does_not_match_table1_for_cdi_schubert() -> None:
    """Documents the trap this module's docstring warns about: sample_size is
    the source publication's own count, not the processed-data count."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    reported = entry_for(index, "cdi_schubert").raw["sample_size"]
    # yaml says H: 155; Table 1 / the metadata-OTU intersection says 154.
    assert reported["H"] == 155


# --------------------------------------------------------------------------
# study_dir / metadata_path / otu_table_path against a real extracted study.
# --------------------------------------------------------------------------

_T1D_DIR = DATA_RAW / "t1d_mejialeon_results"


def test_study_dir_resolves_through_folder_not_dataset_id() -> None:
    _require(DATASET_INFO, _T1D_DIR)
    index = load_dataset_info(DATASET_INFO)
    resolved = study_dir(index, "t1d_mejialeon", DATA_RAW)
    assert resolved == _T1D_DIR


def test_study_dir_raises_named_error_when_not_extracted() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    with pytest.raises(DatasetFileError, match="crc_wang"):
        study_dir(index, "crc_wang", DATA_RAW / "nonexistent_root")


def test_metadata_path_finds_the_real_file() -> None:
    _require(_T1D_DIR)
    path = metadata_path("t1d_mejialeon", _T1D_DIR)
    assert path.name == "t1d_mejialeon.metadata.txt"
    assert path.exists()


def test_otu_table_path_resolves_via_summary_file_not_string_building() -> None:
    _require(_T1D_DIR)
    path = otu_table_path("t1d_mejialeon", _T1D_DIR)
    assert path.name == "t1d_mejialeon.otu_table.100.denovo.rdp_assigned"
    assert path.parent.name == "RDP"
    assert path.exists()


# --------------------------------------------------------------------------
# load_dataset_info -- synthetic malformed input, no dependency on fetched data.
# --------------------------------------------------------------------------


def test_load_dataset_info_raises_on_non_mapping(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="expected a non-empty mapping"):
        load_dataset_info(path)


def test_load_dataset_info_raises_on_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ValueError, match="expected a non-empty mapping"):
        load_dataset_info(path)


def test_load_dataset_info_raises_on_missing_folder_field(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "no_folder.yaml"
    path.write_text(yaml.dump({"some_id": {"region": "V4"}}))
    with pytest.raises(ValueError, match="some_id.*folder"):
        load_dataset_info(path)


def test_load_dataset_info_handles_a_commented_block(tmp_path: pathlib.Path) -> None:
    """Mirrors the real file's hiv_dubourg pattern: a block commented out
    with '#' must not appear in the parsed result, and must not break
    parsing of the real entries around it."""
    path = tmp_path / "with_comment.yaml"
    path.write_text(
        "real_id:\n"
        "    folder: real_id_results\n"
        "#commented_id:\n"
        "#    folder: commented_id_results\n"
        "#    notes: free text with a colon: like this\n"
        "another_real_id:\n"
        "    folder: another_real_id_results\n"
    )
    index = load_dataset_info(path)
    assert set(index) == {"real_id", "another_real_id"}


# --------------------------------------------------------------------------
# load_summary_file -- synthetic fixtures covering the real format's quirks.
# --------------------------------------------------------------------------


def test_load_summary_file_parses_tab_delimited_keys(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "summary_file.txt"
    path.write_text(
        "DATASET_ID\tsome_study\n"
        "\n"
        "#16S_start\n"
        "OTU_TABLE_RDP\tsome_study.otu_table.100.denovo.rdp_assigned\n"
        "PROCESSED\tTrue\n"
        "#16S_end\n"
    )
    result = load_summary_file(path)
    assert result["DATASET_ID"] == "some_study"
    assert result["OTU_TABLE_RDP"] == "some_study.otu_table.100.denovo.rdp_assigned"
    assert result["PROCESSED"] == "True"
    assert "#16S_start" not in result


def test_load_summary_file_tolerates_duplicate_key_with_same_value(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "summary_file.txt"
    path.write_text("DATASET_ID\tsome_study\nDATASET_ID\tsome_study\n")
    result = load_summary_file(path)
    assert result["DATASET_ID"] == "some_study"


def test_load_summary_file_raises_on_duplicate_key_with_different_values(
    tmp_path: pathlib.Path,
) -> None:
    """A key repeating with disagreeing values *before* #ITS_start (i.e. not
    explained by the known 16S/ITS split below) must raise, not silently take
    whichever line came last."""
    path = tmp_path / "summary_file.txt"
    path.write_text("PROCESSED\tTrue\nPROCESSED\tN/A\n")
    with pytest.raises(ValueError, match="PROCESSED.*twice"):
        load_summary_file(path)


def test_load_summary_file_stops_before_the_its_section(tmp_path: pathlib.Path) -> None:
    """All 31 real archives repeat PROCESSED across #16S and #ITS with
    different values (True, then N/A) -- this project has no ITS data
    (the project works from OTU tables only), so parsing stops at #ITS_start
    rather than raising on an
    expected, universal, out-of-scope duplication."""
    path = tmp_path / "summary_file.txt"
    path.write_text(
        "DATASET_ID\tsome_study\n"
        "\n"
        "#16S_start\n"
        "PROCESSED\tTrue\n"
        "#16S_end\n"
        "#ITS_start\n"
        "PROCESSED\tN/A\n"
        "#ITS_end\n"
    )
    result = load_summary_file(path)
    assert result["PROCESSED"] == "True"


def test_load_summary_file_raises_on_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetFileError, match="summary_file.txt"):
        load_summary_file(tmp_path / "missing" / "summary_file.txt")


# --------------------------------------------------------------------------
# otu_table_path / metadata_path -- synthetic study directories.
# --------------------------------------------------------------------------


def test_otu_table_path_raises_when_key_missing_from_summary(
    tmp_path: pathlib.Path,
) -> None:
    study = tmp_path / "study_results"
    study.mkdir()
    (study / "summary_file.txt").write_text("DATASET_ID\tstudy\n")
    with pytest.raises(DatasetFileError, match="OTU_TABLE_RDP"):
        otu_table_path("study", study)


def test_otu_table_path_raises_when_named_file_absent(tmp_path: pathlib.Path) -> None:
    study = tmp_path / "study_results"
    study.mkdir()
    (study / "summary_file.txt").write_text(
        "DATASET_ID\tstudy\nOTU_TABLE_RDP\tstudy.otu_table.100.denovo.rdp_assigned\n"
    )
    with pytest.raises(DatasetFileError, match="does not exist"):
        otu_table_path("study", study)


def test_metadata_path_raises_when_no_match(tmp_path: pathlib.Path) -> None:
    study = tmp_path / "study_results"
    study.mkdir()
    with pytest.raises(DatasetFileError, match="found 0"):
        metadata_path("study", study)


def test_metadata_path_raises_when_multiple_matches(tmp_path: pathlib.Path) -> None:
    study = tmp_path / "study_results"
    study.mkdir()
    (study / "a.metadata.txt").write_text("x")
    (study / "b.metadata.txt").write_text("x")
    with pytest.raises(DatasetFileError, match="found 2"):
        metadata_path("study", study)


def test_metadata_path_ignores_macos_appledouble_sidecar(
    tmp_path: pathlib.Path,
) -> None:
    """Regression test: three real archives (cdi_youngster, hiv_lozupone,
    hiv_noguerajulian) were packaged on a Mac and ship a
    ._<name>.metadata.txt resource-fork stub matching the same glob as the
    real file. scripts/download.py already filters these at extraction time;
    this is defense in depth for any other extraction path."""
    study = tmp_path / "study_results"
    study.mkdir()
    (study / "study.metadata.txt").write_text("sample_id\tDiseaseState\nS1\tH\n")
    (study / "._study.metadata.txt").write_bytes(b"\x00\x05\x16\x07junk")

    path = metadata_path("study", study)

    assert path.name == "study.metadata.txt"


def test_real_previously_affected_studies_resolve_to_one_metadata_file() -> None:
    """cdi_youngster, hiv_lozupone and hiv_noguerajulian are the three real
    archives that shipped AppleDouble sidecars -- confirms the fix holds
    against the actual re-extracted data, not just a synthetic fixture."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    for dataset_id in ("cdi_youngster", "hiv_lozupone", "hiv_noguerajulian"):
        folder = DATA_RAW / entry_for(index, dataset_id).folder
        _require(folder)
        path = metadata_path(dataset_id, folder)
        assert not path.name.startswith("._")


def test_dataset_info_namedtuple_fields_are_accessible() -> None:
    info = DatasetInfo(dataset_id="x", folder="x_results", raw={"region": "V4"})
    assert info.dataset_id == "x"
    assert info.folder == "x_results"
    assert info.raw["region"] == "V4"
