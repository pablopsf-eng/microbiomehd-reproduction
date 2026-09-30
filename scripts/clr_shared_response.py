#!/usr/bin/env python3
"""M4 CLR question: does a centered-log-ratio-transformed differential-
abundance pass change the "shared response" label set, relative to this
project's own already-committed Euclidean result (docs/reproduction-m3.md)?

**Decided (2026-09-18, Pablo):** standalone/manual, not wired into the
Snakemake DAG -- lower
verification bar than the rest of the DAG (no published ground truth exists
for this comparison, see the module docstrings in mbhd.compositional and
mbhd.classifier), and this avoids lengthening the full pipeline's real,
network-dependent run for an exploratory analysis.

Runs `mbhd.shared_response.build_qvalue_matrix` twice over the same 30
analysed datasets -- once with the default (Euclidean) `differential_
abundance`, once with `abundance_transform=functools.partial(mbhd.
compositional.clr, pseudocount_fn=mbhd.compositional.multiplicative_
replacement)` -- and diffs the two labellings over the same two dataset
tiers M3 used (full-29, clean-27), both restricted to
`mbhd.shared_response.PUBLISHED_EXCLUSIONS` (M3's own primary-target
exclusion set) so the comparison is apples-to-apples with M3's headline
result rather than crossed against a second, unrelated exclusion dimension.

Usage:
    uv run scripts/clr_shared_response.py
    uv run scripts/clr_shared_response.py --out results/clr_shared_response \
        --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from functools import partial
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.differential import differential_abundance
from mbhd.shared_response import (
    PUBLISHED_EXCLUSIONS,
    build_qvalue_matrix,
    shared_response_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "clr_shared_response"

#: Must match scripts/shared_response.py's own CLEAN_27 exactly -- not
#: re-derived here, for the same reason that script gives (see
#: docs/reproduction-m3.md for the definition of "genus-set-exact").
#: Updated 2026-09-23, 23 -> 27: mbhd.labels.CONDITION_FILTER made
#: edd_singh/hiv_lozupone/nash_wong/ob_goodrich reconcile exactly too.
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


def _clr_compute_fn(dataset_id: str, index, data_root: Path) -> pd.DataFrame:
    return differential_abundance(
        dataset_id,
        index,
        data_root,
        abundance_transform=partial(clr, pseudocount_fn=multiplicative_replacement),
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


def _diff(
    clr_labels: dict[str, str], euclidean_labels: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One row per lineage in either label set; a summary of agreement/flips."""
    lineages = sorted(set(clr_labels) | set(euclidean_labels))
    table = pd.DataFrame(index=pd.Index(lineages, name="lineage"))
    table["euclidean_label"] = pd.Series(euclidean_labels).reindex(lineages)
    table["clr_label"] = pd.Series(clr_labels).reindex(lineages)
    both = table["euclidean_label"].notna() & table["clr_label"].notna()
    table["flipped"] = pd.NA
    table.loc[both, "flipped"] = (
        table.loc[both, "euclidean_label"] != table.loc[both, "clr_label"]
    )

    summary = {
        "n_euclidean_pool": int(table["euclidean_label"].notna().sum()),
        "n_clr_pool": int(table["clr_label"].notna().sum()),
        "n_common": int(both.sum()),
        "n_flipped": int((table.loc[both, "flipped"] == True).sum()),  # noqa: E712
        "n_only_euclidean": int(
            (table["euclidean_label"].notna() & table["clr_label"].isna()).sum()
        ),
        "n_only_clr": int(
            (table["clr_label"].notna() & table["euclidean_label"].isna()).sum()
        ),
    }
    return table, summary


def _provenance() -> dict[str, str]:
    versions = {}
    for package in (
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "pyyaml",
    ):
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

    euclid_rows, euclid_failures = build_qvalue_matrix(
        dataset_ids, index, args.data_root
    )
    clr_rows, clr_failures = build_qvalue_matrix(
        dataset_ids, index, args.data_root, compute_fn=_clr_compute_fn
    )
    if not euclid_rows or not clr_rows:
        print("no dataset reproduced -- cannot compute the CLR diff", file=sys.stderr)
        return 1

    tiers = {
        "full29": PUBLISHED_EXCLUSIONS,
        "clean27": non_clean | PUBLISHED_EXCLUSIONS,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    for name, exclusions in tiers.items():
        euclid_labels = shared_response_labels(euclid_rows, exclusions)
        clr_labels = shared_response_labels(clr_rows, exclusions)
        table, summary = _diff(clr_labels, euclid_labels)
        table.to_csv(args.out / f"{name}.csv")
        summaries[name] = summary
        print(
            f"{name}: euclidean pool {summary['n_euclidean_pool']}, "
            f"clr pool {summary['n_clr_pool']}, "
            f"{summary['n_flipped']}/{summary['n_common']} common lineages "
            "have a different label under CLR"
        )

    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                **_provenance(),
                "dataset_ids": dataset_ids,
                "clean_27": sorted(CLEAN_27),
                "euclid_failed": euclid_failures,
                "clr_failed": clr_failures,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\noutput written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
