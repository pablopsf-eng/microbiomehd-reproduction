"""Tests for mbhd.manifest.

Split by dependency: real committed data (docs/zenodo_840333_manifest.json is
tracked by git, no skip guard needed), gitignored fetched data (data/raw/,
guarded by _require() in the tests/test_supplementary_files.py style), and
fully synthetic malformed manifests (tmp_path, no dependency at all).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from mbhd.checksums import verify_checksum
from mbhd.manifest import (
    ManifestEntry,
    UnknownManifestEntryError,
    entry_for,
    load_manifest,
)

MANIFEST = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "zenodo_840333_manifest.json"
)
DATA_RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


# --------------------------------------------------------------------------
# Against the real, committed manifest.
# --------------------------------------------------------------------------


def test_real_manifest_has_33_entries():
    # 31 study archives + dataset_info.yaml + file-S3.core_genera.txt.
    index = load_manifest(MANIFEST)
    assert len(index) == 33


def test_real_manifest_dataset_info_entry():
    index = load_manifest(MANIFEST)
    entry = entry_for(index, "dataset_info.yaml")
    assert entry == ManifestEntry(
        key="dataset_info.yaml",
        size=29073,
        algorithm="md5",
        expected_hex="dccf3ab65b17260f1d83ca96b4e3ed9f",
        url="https://zenodo.org/api/records/840333/files/dataset_info.yaml/content",
    )


def test_unknown_filename_raises_and_names_the_filename():
    index = load_manifest(MANIFEST)
    with pytest.raises(UnknownManifestEntryError) as exc_info:
        entry_for(index, "nope.txt")
    assert "nope.txt" in str(exc_info.value)


# --------------------------------------------------------------------------
# Against real fetched files -- verifies the two modules compose correctly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["dataset_info.yaml", "file-S3.core_genera.txt"])
def test_real_zenodo_files_verify_against_the_manifest(filename):
    path = DATA_RAW / filename
    _require(path)
    index = load_manifest(MANIFEST)
    entry = entry_for(index, filename)
    verify_checksum(path, entry.expected_hex, entry.algorithm)


# --------------------------------------------------------------------------
# Synthetic malformed manifests -- no data or network dependency.
# --------------------------------------------------------------------------


def test_missing_files_key_raises_and_names_keys_found(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"id": 840333, "title": "not a real manifest"}))

    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)

    message = str(exc_info.value)
    assert "files" in message
    assert "id" in message
    assert "title" in message


def test_checksum_without_colon_raises_and_names_filename(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "key": "bad.txt",
                        "size": 1,
                        "checksum": "not-a-valid-checksum-string",
                        "links": {"self": "https://example.invalid/bad.txt"},
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)

    assert "bad.txt" in str(exc_info.value)


def test_duplicate_key_raises(tmp_path):
    path = tmp_path / "manifest.json"
    entry = {
        "key": "dup.txt",
        "size": 1,
        "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
        "links": {"self": "https://example.invalid/dup.txt"},
    }
    path.write_text(json.dumps({"files": [entry, entry]}))

    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)

    assert "dup.txt" in str(exc_info.value)


def test_entry_missing_key_raises_and_shows_the_entry(tmp_path):
    path = tmp_path / "manifest.json"
    entry = {
        "size": 1,
        "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
        "links": {"self": "https://example.invalid/x"},
    }
    path.write_text(json.dumps({"files": [entry]}))

    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)

    message = str(exc_info.value)
    assert "key" in message
    # Without a filename to name, the raw entry is shown instead.
    assert "example.invalid" in message


@pytest.mark.parametrize(
    "missing_field, entry",
    [
        pytest.param(
            "checksum",
            {"key": "no_checksum.txt", "size": 1, "links": {"self": "u"}},
            id="missing-checksum",
        ),
        pytest.param(
            "size",
            {
                "key": "no_size.txt",
                "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
                "links": {"self": "u"},
            },
            id="missing-size",
        ),
        pytest.param(
            "links",
            {
                "key": "no_links.txt",
                "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
                "size": 1,
            },
            id="missing-links",
        ),
        pytest.param(
            "self",
            {
                "key": "no_self.txt",
                "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
                "size": 1,
                "links": {},
            },
            id="missing-links-self",
        ),
    ],
)
def test_entry_missing_a_required_field_raises_and_names_filename_and_field(
    tmp_path, missing_field, entry
):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"files": [entry]}))

    with pytest.raises(ValueError) as exc_info:
        load_manifest(path)

    message = str(exc_info.value)
    assert entry["key"] in message
    assert missing_field in message
