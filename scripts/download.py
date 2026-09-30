#!/usr/bin/env python3
"""Download everything this project's pipeline needs: the MicrobiomeHD archive
(Zenodo record 840333) and the paper's supplementary data files (GitHub,
``mbhd.download.GITHUB_SUPP_FILES``) -- both fetched by this one script and one
command, so a fresh clone can run the rest of the pipeline without any manual
step.

Every file is streamed from its manifest entry and verified against its
recorded checksum before being trusted; a mismatch raises rather than leaving a
corrupt file in place. Study archives are kept under ``<dest>/archives/`` and
only the members this project reads (metadata, the RDP-assigned OTU table,
summary_file.txt) are extracted alongside them -- see src/mbhd/download.py for
why.

Usage:
    uv run scripts/download.py
    uv run scripts/download.py --dest /path/to/data/raw
    uv run scripts/download.py --only dataset_info.yaml --only crc_zhao_results.tar.gz
    uv run scripts/download.py --force
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

from mbhd.checksums import ChecksumMismatchError
from mbhd.download import (
    DownloadError,
    all_entries,
    download_entry,
    extract_needed,
    is_archive,
)
from mbhd.manifest import UnknownManifestEntryError, entry_for

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "docs" / "zenodo_840333_manifest.json"
_DEFAULT_DEST = _REPO_ROOT / "data" / "raw"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"destination directory (default: {_DEFAULT_DEST})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help=(
            "pinned Zenodo manifest path (default: "
            f"{_DEFAULT_MANIFEST}); the 5 GitHub supplementary files "
            "(mbhd.download.GITHUB_SUPP_FILES) are always fetched too"
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="KEY",
        help="fetch only this manifest key (repeatable); default: fetch everything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a checksum-clean copy is already present",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="download and verify archives but skip extracting members from them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    index = all_entries(args.manifest)
    if args.only:
        try:
            keys = [entry_for(index, key).key for key in args.only]
        except UnknownManifestEntryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        keys = list(index)

    # A single transient failure (Zenodo's flakiness includes
    # intermittent 504 gateway time-outs) must not stop every file
    # after it from even being attempted. Each entry's outcome is independent,
    # so failures are collected and reported together; the run only exits
    # nonzero, never silently, if any remain.
    downloaded = skipped = extracted = 0
    failures: list[tuple[str, str]] = []
    for key in keys:
        entry = index[key]
        try:
            result = download_entry(entry, args.dest, force=args.force)
        except (DownloadError, ChecksumMismatchError) as exc:
            print(f"FAILED    {key}: {exc}", file=sys.stderr)
            failures.append((key, str(exc)))
            continue

        print(f"{result:9s} {key}")
        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1

        if is_archive(key) and not args.no_extract:
            archive_path = args.dest / "archives" / key
            try:
                members = extract_needed(archive_path, args.dest)
            except (ValueError, tarfile.TarError) as exc:
                print(f"FAILED    {key} (extraction): {exc}", file=sys.stderr)
                failures.append((key, f"extraction failed: {exc}"))
                continue
            extracted += len(members)

    print(
        f"\n{downloaded} downloaded, {skipped} already verified, "
        f"{extracted} members extracted, {len(failures)} failed"
    )
    if failures:
        print(
            "\nFailed entries (rerun the same command to retry -- already",
            file=sys.stderr,
        )
        print(
            "-verified files are skipped, so only these are re-fetched):",
            file=sys.stderr,
        )
        for key, message in failures:
            print(f"  {key}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
