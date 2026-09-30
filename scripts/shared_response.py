#!/usr/bin/env python3
"""Reproduce the paper's "shared response" (non-specific genera) result from
this project's own reproduced q-values, and diff it against both published
file-S3 variants.

Runs the already-verified labelling rule (`mbhd.shared_response.
shared_response_labels`) twice, to localize any disagreement (same design
principle M2 used to separate the abundance and statistics layers):

- **full-29**: every dataset that does not hard-error (all 30 analysed
  datasets are attempted; `ob_zupancic` fails and contributes nothing).
- **clean-27**: only the 27 datasets whose genus set already reproduces
  file-S1 exactly (docs/reproduction-m3.md; updated 2026-09-23
  from 23 once `mbhd.labels.CONDITION_FILTER` made `edd_singh`/
  `hiv_lozupone`/`nash_wong`/`ob_goodrich` reconcile exactly too) -- isolates M3's own
  new code (this module) from M1's inherited, already-documented sample-count gaps.

Each of those two runs is compared against both published file-S3 variants
(Zenodo's no-exclusion list and GitHub's three-exclusion published-figures
list), for four comparisons total. Writes one label CSV per comparison to
`results/shared_response/` (gitignored) and a provenance file; the
human-readable summary is written into docs/reproduction-m3.md by hand from
this script's own printed output, matching scripts/reproduce.py's
provenance-recording convention.

Usage:
    uv run scripts/shared_response.py
    uv run scripts/shared_response.py --out results/shared_response --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.shared_response import (
    PUBLISHED_EXCLUSIONS,
    build_qvalue_matrix,
    shared_response_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "shared_response"

#: The 27 datasets whose genus set reproduces file-S1 exactly (docs/
#: reproduction-m3.md -- derived from `scripts/reproduce.py --all`,
#: not re-derived here to avoid this script silently drifting from that
#: run's own definition of "exact"; updated 2026-09-23, 23 -> 27, see
#: module docstring). Includes 3 datasets whose Table-1 sample count still
#: does not reconcile (`crc_wang`, `liv_zhang`, `ob_turnbaugh`) but whose
#: genus *set* is unaffected by that gap.
CLEAN_27 = frozenset(
    {
        "art_scher",
        "asd_kang",
        "asd_son",
        "cdi_schubert",
        "cdi_vincent",
        "crc_baxter",
        "crc_chen",
        "crc_wang",
        "crc_zeller",
        "edd_singh",
        "hiv_dinh",
        "hiv_lozupone",
        "ibd_gevers",
        "ibd_morgan",
        "ibd_papa",
        "ibd_willing",
        "liv_zhang",
        "nash_wong",
        "nash_zhu",
        "noncdi_schubert",
        "ob_goodrich",
        "ob_ross",
        "ob_turnbaugh",
        "ob_zhu",
        "par_scheperjans",
        "t1d_alkanani",
        "t1d_mejialeon",
    }
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help=f"downloaded MicrobiomeHD archive root (default: {_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"output directory, created if needed (default: {_DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def _read_labels(path: Path) -> dict[str, str]:
    """Parse a file-S3. Returns {lineage: label}, "" meaning unlabelled."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        out[fields[0]] = fields[-1].strip()
    return out


def _diff(
    reproduced: dict[str, str], published: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One row per lineage in either set; a summary dict of agreement counts."""
    lineages = sorted(set(reproduced) | set(published))
    table = pd.DataFrame(index=pd.Index(lineages, name="lineage"))
    table["reproduced_label"] = pd.Series(reproduced).reindex(lineages)
    table["published_label"] = pd.Series(published).reindex(lineages)
    both = table["reproduced_label"].notna() & table["published_label"].notna()
    table["agrees"] = pd.NA
    table.loc[both, "agrees"] = (
        table.loc[both, "reproduced_label"] == table.loc[both, "published_label"]
    )

    summary = {
        "n_reproduced_pool": int(table["reproduced_label"].notna().sum()),
        "n_published_pool": int(table["published_label"].notna().sum()),
        "n_common": int(both.sum()),
        "n_agree": int((table.loc[both, "agrees"] == True).sum()),  # noqa: E712
        "n_only_reproduced": int(
            (table["reproduced_label"].notna() & table["published_label"].isna()).sum()
        ),
        "n_only_published": int(
            (table["published_label"].notna() & table["reproduced_label"].isna()).sum()
        ),
    }
    return table, summary


def _provenance() -> dict[str, str]:
    versions = {}
    for package in ("pandas", "numpy", "scipy", "statsmodels", "pyyaml"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        **{f"{pkg}_version": v for pkg, v in versions.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    index = load_dataset_info(args.data_root / "dataset_info.yaml")
    dataset_ids = analysed_dataset_ids(index)
    non_clean = frozenset(dataset_ids) - CLEAN_27

    rows, failures = build_qvalue_matrix(dataset_ids, index, args.data_root)
    if not rows:
        print(
            "no dataset reproduced -- cannot compute shared response", file=sys.stderr
        )
        return 1

    s3_zenodo = args.data_root / "file-S3.core_genera.txt"
    s3_github = args.data_root / "file-S3.nonspecific_genera.txt"

    comparisons = {
        "full29_vs_zenodo": (frozenset(), s3_zenodo),
        "full29_vs_github": (PUBLISHED_EXCLUSIONS, s3_github),
        "clean27_vs_zenodo": (non_clean, s3_zenodo),
        "clean27_vs_github": (non_clean | PUBLISHED_EXCLUSIONS, s3_github),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for name, (exclusions, published_path) in comparisons.items():
        if not published_path.exists():
            print(f"SKIPPED {name}: {published_path.name} not fetched", file=sys.stderr)
            continue
        reproduced = shared_response_labels(rows, exclusions)
        published = _read_labels(published_path)
        table, summary = _diff(reproduced, published)
        table.to_csv(args.out / f"{name}.csv")
        summaries[name] = summary
        print(
            f"{name}: reproduced pool {summary['n_reproduced_pool']}, "
            f"published pool {summary['n_published_pool']}, "
            f"{summary['n_agree']}/{summary['n_common']} common lineages agree"
        )

    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                **_provenance(),
                "dataset_ids": dataset_ids,
                "clean_27": sorted(CLEAN_27),
                "failed": failures,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\noutput written to {args.out}")
    if not summaries:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
