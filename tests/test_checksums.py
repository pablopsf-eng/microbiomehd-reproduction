"""Tests for mbhd.checksums. Fully synthetic -- no data or network dependency.

Known-answer values, independent of this code (confirmed with coreutils:
``printf 'hello world' | sha256sum`` and ``printf '' | md5sum``):
sha256(b"hello world") =
    b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
md5(b"") = d41d8cd98f00b204e9800998ecf8427e
"""

from __future__ import annotations

import hashlib

import pytest

from mbhd.checksums import ChecksumMismatchError, verify_checksum

SHA256_HELLO_WORLD = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
MD5_EMPTY = "d41d8cd98f00b204e9800998ecf8427e"


def test_matching_sha256_does_not_raise(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello world")
    verify_checksum(path, SHA256_HELLO_WORLD, "sha256")


def test_matching_md5_does_not_raise(tmp_path):
    # md5 is the algorithm actually recorded for the real Zenodo-sourced files
    # this function verifies (see the pinned docs/zenodo_840333_manifest.json).
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    verify_checksum(path, MD5_EMPTY, "md5")


def test_mismatch_names_path_algorithm_expected_and_actual(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello world")
    wrong_hex = "0" * 64

    with pytest.raises(ChecksumMismatchError) as exc_info:
        verify_checksum(path, wrong_hex, "sha256")

    message = str(exc_info.value)
    assert str(path) in message
    assert "sha256" in message
    assert wrong_hex in message
    assert SHA256_HELLO_WORLD in message


def test_missing_file_raises_file_not_found_error(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        verify_checksum(missing, SHA256_HELLO_WORLD, "sha256")


def test_unsupported_algorithm_raises_value_error(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello world")
    with pytest.raises(ValueError):
        verify_checksum(path, SHA256_HELLO_WORLD, "not-a-real-algorithm")


def test_empty_file_is_a_well_defined_checksum(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    verify_checksum(path, MD5_EMPTY, "md5")


def test_hash_then_verify_round_trip(tmp_path):
    """Verifying a file against its own just-computed digest must never raise."""
    for content in (b"", b"some content", bytes(range(256)) * 10):
        path = tmp_path / "content.bin"
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        verify_checksum(path, digest, "sha256")
