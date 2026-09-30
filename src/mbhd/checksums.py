"""Verify a file's contents against a known checksum.

Used to confirm that a fetched file matches the checksum recorded for it (Zenodo's
manifest, or a self-generated record for sources that publish none).
Two provenances are involved: Zenodo publishes md5 checksums, while the
GitHub supplementary files publish none.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


class ChecksumMismatchError(Exception):
    """Raised when a file's computed digest does not match the expected one."""

    def __init__(
        self, path: Path, algorithm: str, expected_hex: str, actual_hex: str
    ) -> None:
        super().__init__(
            f"{path}: {algorithm} mismatch (expected {expected_hex}, got {actual_hex})"
        )
        self.path = path
        self.algorithm = algorithm
        self.expected_hex = expected_hex
        self.actual_hex = actual_hex


def verify_checksum(path: Path, expected_hex: str, algorithm: str = "sha256") -> None:
    """Raise if the file at ``path`` does not hash to ``expected_hex``.

    ``algorithm`` is any name accepted by ``hashlib.new`` (e.g. "sha256", "md5").
    An unsupported algorithm name raises ``ValueError`` and a missing file raises
    ``FileNotFoundError`` -- both propagate unchanged, since each already names the
    problem. Only a computed-but-disagreeing digest is this function's own concern,
    raised as ``ChecksumMismatchError``. Reads the file in fixed-size chunks so a
    large archive is never held fully in memory.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual_hex = digest.hexdigest()
    if actual_hex != expected_hex:
        raise ChecksumMismatchError(path, algorithm, expected_hex, actual_hex)
