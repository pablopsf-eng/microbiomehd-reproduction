"""Tests for mbhd.download.

Split by dependency, matching the tests/test_manifest.py convention: pure logic
(no I/O) runs unconditionally; file-transfer logic uses file:// URLs so it stays
offline and fast; a small real-network smoke test is guarded by _require_network()
and skips cleanly with no internet.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import tarfile

import pytest

from mbhd.checksums import ChecksumMismatchError, verify_checksum
from mbhd.download import (
    GITHUB_SUPP_FILES,
    DownloadError,
    all_entries,
    destination_for,
    download_entry,
    extract_needed,
    fetch,
    is_archive,
    is_verified,
)
from mbhd.manifest import ManifestEntry, entry_for, load_manifest

MANIFEST = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "zenodo_840333_manifest.json"
)


# --------------------------------------------------------------------------
# GITHUB_SUPP_FILES / all_entries -- pure logic, no I/O.
# --------------------------------------------------------------------------


def test_github_supp_files_has_the_five_expected_keys() -> None:
    assert set(GITHUB_SUPP_FILES) == {
        "file-S1.qvalues.txt",
        "file-S2.disease_specific_genera.txt",
        "file-S3.nonspecific_genera.txt",
        "file-S5.effects.txt",
        "supp-files-README.md",
    }


def test_github_supp_files_are_sha256_and_never_archives() -> None:
    for entry in GITHUB_SUPP_FILES.values():
        assert entry.algorithm == "sha256"
        assert is_archive(entry.key) is False


def test_github_supp_files_urls_point_at_the_real_supp_files_directory() -> None:
    for key, entry in GITHUB_SUPP_FILES.items():
        assert entry.url.startswith(
            "https://raw.githubusercontent.com/cduvallet/microbiomeHD/master/"
            "final/supp-files/"
        )
        # The GitHub-side README.md is renamed supp-files-README.md at the
        # destination so it never collides with this project's own README.md.
        expected_basename = "README.md" if key == "supp-files-README.md" else key
        assert entry.url.rsplit("/", 1)[-1] == expected_basename


def test_all_entries_merges_zenodo_manifest_and_github_supp_files() -> None:
    index = all_entries(MANIFEST)
    zenodo_only = load_manifest(MANIFEST)

    assert len(index) == len(zenodo_only) + len(GITHUB_SUPP_FILES)
    assert set(GITHUB_SUPP_FILES) <= set(index)
    assert index["file-S1.qvalues.txt"] == GITHUB_SUPP_FILES["file-S1.qvalues.txt"]
    # A real Zenodo entry is still reachable through the same combined index.
    assert index["dataset_info.yaml"] == zenodo_only["dataset_info.yaml"]


def test_all_entries_raises_on_a_colliding_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future Zenodo manifest update that happens to reuse one of
    GITHUB_SUPP_FILES' own keys must fail loudly, not silently pick one
    source over the other."""
    colliding_key = next(iter(GITHUB_SUPP_FILES))
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text(
        '{"files": [{"key": "'
        + colliding_key
        + '", "size": 1, "checksum": "md5:'
        + "0" * 32
        + '", "links": {"self": "file:///x"}}]}'
    )

    with pytest.raises(ValueError, match=colliding_key):
        all_entries(fake_manifest)


def test_all_entries_real_github_entry_downloads_and_verifies_via_file_url(
    tmp_path: pathlib.Path,
) -> None:
    """Exercises one real GITHUB_SUPP_FILES entry through the exact same
    download_entry()/destination_for() path the Zenodo entries use, proving
    the sha256 + loose-file placement actually work end to end for this
    entry shape -- offline, via a file:// override of its real url."""
    real_entry = GITHUB_SUPP_FILES["supp-files-README.md"]
    source = tmp_path / "source.md"
    source.write_bytes(b"anything")
    entry = real_entry._replace(
        expected_hex=hashlib.sha256(b"anything").hexdigest(),
        url=source.as_uri(),
    )
    dest = tmp_path / "data" / "raw"

    result = download_entry(entry, dest, timeout=5)

    assert result == "downloaded"
    # A GitHub supplementary file is a loose file, not an archive: it lands
    # directly at dest/<key>, never under dest/archives/.
    assert destination_for(entry, dest) == dest / "supp-files-README.md"
    assert destination_for(entry, dest).read_bytes() == b"anything"


def _require_network() -> None:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("https://zenodo.org/", timeout=5).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        pytest.skip("no network access")


# --------------------------------------------------------------------------
# is_archive / destination_for -- pure logic, no I/O.
# --------------------------------------------------------------------------


def test_is_archive_true_for_tar_gz() -> None:
    assert is_archive("cdi_schubert_results.tar.gz") is True


def test_is_archive_false_for_loose_files() -> None:
    assert is_archive("dataset_info.yaml") is False
    assert is_archive("file-S3.core_genera.txt") is False


def test_destination_for_archive_goes_under_archives_subdir(
    tmp_path: pathlib.Path,
) -> None:
    entry = ManifestEntry(
        key="cdi_schubert_results.tar.gz",
        size=1,
        algorithm="md5",
        expected_hex="x",
        url="file:///x",
    )
    assert destination_for(entry, tmp_path) == tmp_path / "archives" / entry.key


def test_destination_for_loose_file_goes_at_dest_root(tmp_path: pathlib.Path) -> None:
    entry = ManifestEntry(
        key="dataset_info.yaml",
        size=1,
        algorithm="md5",
        expected_hex="x",
        url="file:///x",
    )
    assert destination_for(entry, tmp_path) == tmp_path / entry.key


# --------------------------------------------------------------------------
# Every real manifest entry's key must round-trip through destination_for
# without any two entries landing on the same path.
# --------------------------------------------------------------------------


def test_all_real_manifest_destinations_are_distinct() -> None:
    index = load_manifest(MANIFEST)
    dest = pathlib.Path("data/raw")
    paths = [destination_for(entry, dest) for entry in index.values()]
    assert len(paths) == len(set(paths))


# --------------------------------------------------------------------------
# is_verified -- checksum-based skip logic.
# --------------------------------------------------------------------------


def test_is_verified_false_when_file_missing(tmp_path: pathlib.Path) -> None:
    entry = ManifestEntry(
        key="x",
        size=1,
        algorithm="sha256",
        expected_hex="doesnotmatter",
        url="file:///x",
    )
    assert is_verified(tmp_path / "absent.txt", entry) is False


def test_is_verified_true_when_checksum_matches(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    entry = ManifestEntry(
        key="f.txt", size=11, algorithm="sha256", expected_hex=expected, url="file:///x"
    )
    assert is_verified(path, entry) is True


def test_is_verified_false_when_checksum_mismatches(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"corrupted")
    entry = ManifestEntry(
        key="f.txt", size=11, algorithm="sha256", expected_hex="0" * 64, url="file:///x"
    )
    assert is_verified(path, entry) is False


# --------------------------------------------------------------------------
# fetch() -- offline, against file:// URLs.
# --------------------------------------------------------------------------


def test_fetch_happy_path_via_file_url(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"the actual bytes")
    target = tmp_path / "dest" / "out.bin"

    fetch(source.as_uri(), target, timeout=5, retries=1)

    assert target.read_bytes() == b"the actual bytes"
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_fetch_raises_download_error_after_exhausting_retries(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mbhd.download.time.sleep", lambda _seconds: None)
    missing = (tmp_path / "does_not_exist.bin").as_uri()
    target = tmp_path / "dest" / "out.bin"

    with pytest.raises(DownloadError, match="does_not_exist.bin"):
        fetch(missing, target, timeout=5, retries=2)

    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".part").exists()


# --------------------------------------------------------------------------
# download_entry() -- fetch + verify + skip-if-clean, offline via file://.
# --------------------------------------------------------------------------


def test_download_entry_downloads_and_verifies(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source" / "dataset_info.yaml"
    source.parent.mkdir()
    source.write_bytes(b"hello world")
    entry = ManifestEntry(
        key="dataset_info.yaml",
        size=11,
        algorithm="sha256",
        expected_hex="b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        url=source.as_uri(),
    )
    dest = tmp_path / "data" / "raw"

    result = download_entry(entry, dest, timeout=5)

    assert result == "downloaded"
    target = destination_for(entry, dest)
    assert target.read_bytes() == b"hello world"


def test_download_entry_skips_when_already_verified(tmp_path: pathlib.Path) -> None:
    entry = ManifestEntry(
        key="dataset_info.yaml",
        size=11,
        algorithm="sha256",
        expected_hex="b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        # Deliberately unreachable: if download_entry tried to fetch, this would
        # raise, proving the skip path was actually taken rather than a fetch
        # that happened to succeed.
        url="file:///nonexistent/path/should/not/be/read.bin",
    )
    dest = tmp_path / "data" / "raw"
    target = destination_for(entry, dest)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"hello world")

    result = download_entry(entry, dest, timeout=5)

    assert result == "skipped"


def test_download_entry_raises_and_deletes_on_checksum_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source" / "dataset_info.yaml"
    source.parent.mkdir()
    source.write_bytes(b"not what the manifest expects")
    entry = ManifestEntry(
        key="dataset_info.yaml",
        size=11,
        algorithm="sha256",
        expected_hex="b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        url=source.as_uri(),
    )
    dest = tmp_path / "data" / "raw"

    with pytest.raises(ChecksumMismatchError):
        download_entry(entry, dest, timeout=5)

    assert not destination_for(entry, dest).exists()


def test_download_entry_force_redownloads_even_if_verified(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source" / "dataset_info.yaml"
    source.parent.mkdir()
    source.write_bytes(b"hello world")
    entry = ManifestEntry(
        key="dataset_info.yaml",
        size=11,
        algorithm="sha256",
        expected_hex="b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        url=source.as_uri(),
    )
    dest = tmp_path / "data" / "raw"
    target = destination_for(entry, dest)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"hello world")

    result = download_entry(entry, dest, force=True, timeout=5)

    assert result == "downloaded"


# --------------------------------------------------------------------------
# extract_needed() -- synthetic tar fixtures, no network.
# --------------------------------------------------------------------------


def _make_archive(tmp_path: pathlib.Path, members: dict[str, bytes]) -> pathlib.Path:
    archive_path = tmp_path / "study_results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, fileobj=io.BytesIO(content))
    return archive_path


def test_extract_needed_extracts_only_matching_members(tmp_path: pathlib.Path) -> None:
    archive = _make_archive(
        tmp_path,
        {
            "study_results/summary_file.txt": b"DATASET_ID\tstudy\n",
            "study_results/study.metadata.txt": b"sample_id\tDiseaseState\n",
            "study_results/RDP/study.otu_table.100.denovo.rdp_assigned": b"OTU_ID\n",
            "study_results/study.raw_dereplicated.fasta": b">not needed\nACGT\n",
            "study_results/quality_control/read_lengths_distribution.png": b"\x89PNG",
        },
    )
    dest = tmp_path / "extracted"

    extracted_names = extract_needed(archive, dest)

    assert len(extracted_names) == 3
    assert (
        dest / "study_results" / "summary_file.txt"
    ).read_bytes() == b"DATASET_ID\tstudy\n"
    assert (dest / "study_results" / "study.metadata.txt").exists()
    assert (
        dest / "study_results" / "RDP" / "study.otu_table.100.denovo.rdp_assigned"
    ).exists()
    assert not (dest / "study_results" / "study.raw_dereplicated.fasta").exists()
    assert not (dest / "study_results" / "quality_control").exists()


def test_extract_needed_skips_macos_appledouble_sidecars(
    tmp_path: pathlib.Path,
) -> None:
    """Regression test: cdi_youngster, hiv_lozupone and hiv_noguerajulian were
    packaged on a Mac and each ship a ._<name>.metadata.txt resource-fork stub
    alongside the real metadata file. Left unfiltered, both match the
    NEEDED_MEMBERS suffix and metadata_path() sees two candidates."""
    archive = _make_archive(
        tmp_path,
        {
            "study_results/study.metadata.txt": b"sample_id\tDiseaseState\nS1\tH\n",
            "study_results/._study.metadata.txt": b"\x00\x05\x16\x07AppleDouble junk",
        },
    )
    dest = tmp_path / "extracted"

    extracted_names = extract_needed(archive, dest)

    assert extracted_names == ["study_results/study.metadata.txt"]
    assert not (dest / "study_results" / "._study.metadata.txt").exists()


def test_extract_needed_raises_when_nothing_matches(tmp_path: pathlib.Path) -> None:
    archive = _make_archive(tmp_path, {"study_results/only_a_readme.txt": b"hi"})
    dest = tmp_path / "extracted"

    with pytest.raises(ValueError, match="no members matching"):
        extract_needed(archive, dest)


def test_extract_needed_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    archive = _make_archive(tmp_path, {"../escape.metadata.txt": b"malicious"})
    dest = tmp_path / "extracted"
    dest.mkdir()

    with pytest.raises(tarfile.TarError):
        extract_needed(archive, dest)

    # Nothing was written outside dest.
    assert not (tmp_path / "escape.metadata.txt").exists()


# --------------------------------------------------------------------------
# Real network smoke test -- skips cleanly offline.
# --------------------------------------------------------------------------


def test_download_entry_against_real_zenodo_smallest_loose_file(
    tmp_path: pathlib.Path,
) -> None:
    """_require_network() only probes zenodo.org's root domain, which cannot
    detect the file-content endpoint's own documented, separate outage mode
    (extended 504s specific to
    /files/.../content have been observed). So a
    root-domain-reachable-but-file-endpoint-down
    day must still skip cleanly here, not fail CI on a real Zenodo outage
    this test has no way to distinguish from a genuine bug -- caught by a
    2026-09-24 review reading this test's own guard against what it actually
    exercises."""
    _require_network()
    index = load_manifest(MANIFEST)
    entry = entry_for(index, "file-S3.core_genera.txt")
    dest = tmp_path / "data" / "raw"

    try:
        result = download_entry(entry, dest, timeout=60)
    except DownloadError as exc:
        pytest.skip(f"Zenodo file endpoint unreachable: {exc}")

    assert result == "downloaded"
    target = destination_for(entry, dest)
    verify_checksum(target, entry.expected_hex, entry.algorithm)  # raises if wrong
