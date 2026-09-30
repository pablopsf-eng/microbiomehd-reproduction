"""Fetch and selectively extract the MicrobiomeHD archive from Zenodo record 840333,
plus the paper's supplementary data files hosted on GitHub (``GITHUB_SUPP_FILES``).

Two different provenances, two different checksum conventions (Zenodo publishes
md5; the GitHub files publish nothing), unified into one
``{key: ManifestEntry}`` index by ``all_entries()``
so the rest of this module and ``scripts/download.py`` do not need to know which
source a given key came from:

- **Zenodo** files are verified against the upstream md5 recorded in the pinned
  manifest (docs/zenodo_840333_manifest.json). Zenodo has been observed to answer
  a failed fetch with a 92-byte HTML error body under a success status, so
  neither a 200 nor a clean exit from urllib is treated as evidence that the
  bytes are right -- the
  checksum is the only gate that counts.
- **GitHub** files (``cduvallet/microbiomeHD``, ``final/supp-files/``) carry no
  published checksum at all, so ``GITHUB_SUPP_FILES``' sha256 values are
  self-generated at first fetch, not an upstream digest: a mismatch on a later
  run means the file changed on GitHub since, not that this download is corrupt.

Study archives expand about twentyfold, almost all of it raw FASTA that nothing
in this project reads. Only the three member patterns in ``NEEDED_MEMBERS`` are
extracted, which keeps data/raw near 1 GB instead of roughly 3 GB. The verified
.tar.gz files are kept so the checksums stay re-checkable after extraction.
"""

from __future__ import annotations

import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from mbhd.checksums import ChecksumMismatchError, verify_checksum
from mbhd.manifest import ManifestEntry, load_manifest

_CHUNK_SIZE = 1024 * 1024

#: Suffixes of the tar members this project actually reads. Everything else in a
#: study archive (raw/dereplicated FASTA, QC PNGs, the non-RDP OTU variants) is
#: left inside the .tar.gz.
NEEDED_MEMBERS = ("/summary_file.txt", ".metadata.txt", ".rdp_assigned")

_GITHUB_SUPP_BASE = (
    "https://raw.githubusercontent.com/cduvallet/microbiomeHD/master/final/supp-files"
)

#: The paper's supplementary data files (Duvallet et al. 2017), hosted on GitHub
#: rather than Zenodo -- no licence, no published checksum. Every checksum here
#: is a sha256 this project generated itself at first fetch, byte-for-byte
#: confirmed against the real files during the 2026-09-24 review. Never part of
#: docs/zenodo_840333_manifest.json, which is the real, committed Zenodo API
#: response and must stay exactly that -- kept as a separate, small, hardcoded
#: index instead, merged in by ``all_entries()``. ``README.md`` on GitHub is
#: renamed ``supp-files-README.md`` at the destination so it never collides
#: with this project's own top-level ``README.md``.
GITHUB_SUPP_FILES: dict[str, ManifestEntry] = {
    entry.key: entry
    for entry in (
        ManifestEntry(
            key="file-S1.qvalues.txt",
            size=79577,
            algorithm="sha256",
            expected_hex=(
                "06addebe7e09aacad1b21f7c9b21d1d64e28e4a88c70d8f4d1ca34e132b184d4"
            ),
            url=f"{_GITHUB_SUPP_BASE}/file-S1.qvalues.txt",
        ),
        ManifestEntry(
            key="file-S2.disease_specific_genera.txt",
            size=32087,
            algorithm="sha256",
            expected_hex=(
                "5a68f7d78122f7ed39b7f61ebd0fa629adbb6ee0195404faf906e7b4010623d3"
            ),
            url=f"{_GITHUB_SUPP_BASE}/file-S2.disease_specific_genera.txt",
        ),
        ManifestEntry(
            key="file-S3.nonspecific_genera.txt",
            size=14911,
            algorithm="sha256",
            expected_hex=(
                "0da62d7f479aba1c593a5f28ccf4d0ed18dd6cf18fc03172f7fc962dccec4dc8"
            ),
            url=f"{_GITHUB_SUPP_BASE}/file-S3.nonspecific_genera.txt",
        ),
        ManifestEntry(
            key="file-S5.effects.txt",
            size=52044,
            algorithm="sha256",
            expected_hex=(
                "a2e1facf698953aab424e255707627bcd270d41f31c926521a27af04f5d45390"
            ),
            url=f"{_GITHUB_SUPP_BASE}/file-S5.effects.txt",
        ),
        ManifestEntry(
            key="supp-files-README.md",
            size=2034,
            algorithm="sha256",
            expected_hex=(
                "668a0e74590385c6f7803b1c4bc281ec6daccf53cf40c3f47ce2bc31e8d25d0a"
            ),
            url=f"{_GITHUB_SUPP_BASE}/README.md",
        ),
    )
}


def all_entries(manifest_path: Path) -> dict[str, ManifestEntry]:
    """Every fetchable entry: the pinned Zenodo manifest plus ``GITHUB_SUPP_FILES``.

    Raises ``ValueError`` naming the key if the two sources ever collide --
    never a silent overwrite -- though this cannot happen today (the 33 Zenodo
    keys and the 5 GitHub keys are disjoint, checked here rather than assumed
    to stay that way forever).
    """
    zenodo = load_manifest(manifest_path)
    collisions = sorted(set(zenodo) & set(GITHUB_SUPP_FILES))
    if collisions:
        raise ValueError(
            f"{manifest_path}: key(s) {collisions} present in both the Zenodo "
            "manifest and GITHUB_SUPP_FILES -- cannot merge unambiguously"
        )
    return {**zenodo, **GITHUB_SUPP_FILES}


class DownloadError(Exception):
    """Raised when a file could not be fetched after every retry."""


def is_archive(key: str) -> bool:
    """True if this manifest key is a per-study .tar.gz rather than a loose file.

    The Zenodo record holds 31 study archives plus two loose files
    (dataset_info.yaml and file-S3.core_genera.txt); ``GITHUB_SUPP_FILES``
    adds five more loose files. Every loose file is stored at the root of the
    destination rather than under archives/, and none is ever extracted.
    """
    return key.endswith(".tar.gz")


def destination_for(entry: ManifestEntry, dest: Path) -> Path:
    """Where ``entry`` is stored.

    ``dest/archives/<key>`` for a study archive, ``dest/<key>`` for a loose file.
    """
    return dest / "archives" / entry.key if is_archive(entry.key) else dest / entry.key


def is_verified(path: Path, entry: ManifestEntry) -> bool:
    """True if ``path`` already exists and matches the manifest's checksum.

    Used to make the downloader resumable: an interrupted run can be repeated
    and will re-fetch only what is missing or corrupt.
    """
    if not path.exists():
        return False
    try:
        verify_checksum(path, entry.expected_hex, entry.algorithm)
    except ChecksumMismatchError:
        return False
    return True


def fetch(url: str, dest: Path, timeout: float = 60.0, retries: int = 5) -> None:
    """Stream ``url`` to ``dest``, retrying with exponential backoff.

    Writes to a sibling ``.part`` file and renames only on success, so an
    interrupted transfer can never be mistaken for a complete one. Raises
    ``DownloadError`` naming the url and the last error once ``retries``
    attempts are exhausted. This function does not verify contents -- the
    caller must checksum the result.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(retries):
        if attempt:
            time.sleep(2.0**attempt)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with open(partial, "wb") as out:
                    while chunk := response.read(_CHUNK_SIZE):
                        out.write(chunk)
            partial.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)

    raise DownloadError(
        f"{url}: failed after {retries} attempts; last error: {last_error!r}"
    )


def download_entry(
    entry: ManifestEntry, dest: Path, force: bool = False, timeout: float = 60.0
) -> str:
    """Fetch one manifest entry into ``dest`` and verify it.

    Returns "skipped" if the file was already present and checksum-clean (unless
    ``force``), otherwise "downloaded". Raises ``ChecksumMismatchError`` if the
    fetched bytes do not match the manifest, having first deleted the bad file so
    a later run cannot mistake it for a good one.
    """
    target = destination_for(entry, dest)
    if not force and is_verified(target, entry):
        return "skipped"

    fetch(entry.url, target, timeout=timeout)
    try:
        verify_checksum(target, entry.expected_hex, entry.algorithm)
    except ChecksumMismatchError:
        target.unlink(missing_ok=True)
        raise
    return "downloaded"


def _needed_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    """Yield the regular-file members matching ``NEEDED_MEMBERS``.

    Skips macOS AppleDouble sidecar files (basename starting with ``._``):
    three archives (cdi_youngster, hiv_lozupone, hiv_noguerajulian) were
    packaged on a Mac and each ship a ``._<name>.metadata.txt`` resource-fork
    stub alongside the real metadata file, matching the same suffix. Left
    unfiltered, this doubles the metadata match for those three studies.
    """
    for member in archive.getmembers():
        basename = member.name.rsplit("/", 1)[-1]
        if member.isfile() and not basename.startswith("._"):
            if member.name.endswith(NEEDED_MEMBERS):
                yield member


def extract_needed(archive_path: Path, dest: Path) -> list[str]:
    """Extract only the members this project reads, returning their names.

    Uses tarfile's "data" filter, which rejects absolute paths, parent-directory
    traversal, links pointing outside the destination, and device files -- so a
    malicious or malformed archive cannot write outside ``dest``.
    """
    with tarfile.open(archive_path, "r:gz") as archive:
        members = list(_needed_members(archive))
        if not members:
            raise ValueError(
                f"{archive_path}: no members matching {NEEDED_MEMBERS}; "
                f"archive holds {len(archive.getmembers())} entries"
            )
        archive.extractall(dest, members=members, filter="data")
    return [member.name for member in members]
