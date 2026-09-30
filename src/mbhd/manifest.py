"""Load the pinned Zenodo record manifest for MicrobiomeHD (record 840333).

The manifest (docs/zenodo_840333_manifest.json) is a full Zenodo API record
response, fetched once and committed so filenames, sizes and checksums stay
available even while the record is unreachable (the Zenodo record has been
observed returning 504 Gateway Time-out for days). This module is the single
place that knows its shape -- two ad-hoc, uncommitted scripts previously
disagreed with each other about it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class ManifestEntry(NamedTuple):
    """One file listed in the manifest.

    ``algorithm`` and ``expected_hex`` come from splitting the manifest's
    ``"algorithm:hex"`` checksum string (e.g. ``"md5:1c765176..."``) -- the
    manifest never gives a bare hex digest.
    """

    key: str
    size: int
    algorithm: str
    expected_hex: str
    url: str


class UnknownManifestEntryError(LookupError):
    """Raised when a requested filename is not present in the manifest.

    Subclasses ``LookupError`` rather than ``KeyError``: ``KeyError.__str__``
    reprs a single-argument message, which would wrap this class's already
    -formatted message in an extra pair of quotes.
    """


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """Parse a Zenodo record manifest into ``{filename: ManifestEntry}``.

    Raises ``ValueError`` if the top-level ``"files"`` key is absent, if a
    file entry is missing ``"key"``, ``"checksum"``, ``"size"`` or
    ``"links"."self"``, if a checksum string has no ``"algorithm:hex"``
    separator, or if two entries share the same ``key`` -- each names what
    was wrong rather than silently guessing or overwriting.
    """
    record = json.loads(path.read_text())
    if "files" not in record:
        raise ValueError(
            f"{path}: manifest has no 'files' key; top-level keys found: "
            f"{sorted(record.keys())}"
        )

    index: dict[str, ManifestEntry] = {}
    for file_entry in record["files"]:
        try:
            filename = file_entry["key"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: file entry has no 'key': {file_entry!r}"
            ) from exc

        try:
            checksum = file_entry["checksum"]
            size = file_entry["size"]
            url = file_entry["links"]["self"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: entry {filename!r} is missing required field {exc}"
            ) from exc

        if ":" not in checksum:
            raise ValueError(
                f"{path}: checksum for {filename!r} has no 'algorithm:hex' "
                f"separator: {checksum!r}"
            )
        algorithm, expected_hex = checksum.split(":", 1)

        if filename in index:
            raise ValueError(
                f"{path}: duplicate manifest entry for filename {filename!r}"
            )
        index[filename] = ManifestEntry(
            key=filename,
            size=size,
            algorithm=algorithm,
            expected_hex=expected_hex,
            url=url,
        )
    return index


def entry_for(index: dict[str, ManifestEntry], filename: str) -> ManifestEntry:
    """Look up ``filename`` in a loaded manifest index.

    Raises ``UnknownManifestEntryError`` naming ``filename`` if it is absent,
    rather than a bare ``KeyError`` with no context.
    """
    try:
        return index[filename]
    except KeyError:
        raise UnknownManifestEntryError(
            f"{filename!r} is not in the manifest"
        ) from None
