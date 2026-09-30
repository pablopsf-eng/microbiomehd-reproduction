#!/usr/bin/env python3
"""Compute per-genus differential abundance and diff it against the published
file-S1/file-S5 ground truth, for one or more MicrobiomeHD datasets.

Milestone-agnostic by design: the dataset subset is a command-line argument
rather than being hardcoded, so a later milestone's run over more datasets
reuses this script instead of a per-milestone sibling (e.g. no
``reproduce_m2.py`` / ``reproduce_m3.py`` pair).

Writes one CSV per dataset (every genus in either the reproduction or the
published file, with both q-values, both effect sizes, and the differences)
plus a cross-dataset summary CSV and a provenance file recording the exact
tool versions used -- versions belong in the output itself, not only in the
README or the lockfile.

Usage:
    uv run scripts/reproduce.py                       # M2's 4 subset datasets
    uv run scripts/reproduce.py --datasets cdi_schubert crc_baxter
    uv run scripts/reproduce.py --all                  # all 30 analysed datasets
    uv run scripts/reproduce.py --out results/reproduction --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.differential import EXPECTED_DATASET_ERRORS, differential_abundance

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "reproduction"

#: M2's chosen subset: all four are from M1's 18 exactly-reconciled datasets,
#: so no result here is confounded by M1's open sample-count mismatches.
M2_SUBSET = ["cdi_schubert", "crc_baxter", "par_scheperjans", "t1d_mejialeon"]

_SIGNIFICANCE_THRESHOLD = 0.05


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET_ID",
        help=f"dataset IDs to reproduce (default: M2's subset, {M2_SUBSET})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="reproduce all 30 analysed datasets instead of --datasets",
    )
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


def _is_significant(signed_q: pd.Series) -> pd.Series:
    return (
        signed_q.notna()
        & (signed_q.abs() < _SIGNIFICANCE_THRESHOLD)
        & (signed_q != 0.0)
    )


def _diff_table(
    reproduced: pd.DataFrame, published_q: pd.Series, published_effect: pd.Series
) -> pd.DataFrame:
    """One row per genus in either the reproduction or the published file.

    A genus present on only one side is a real finding (a genus-set
    mismatch), not an error -- it appears here as a row with NaN on the
    other side rather than being silently dropped by an inner join.
    """
    genera = sorted(set(reproduced.index) | set(published_q.dropna().index))
    table = pd.DataFrame(index=pd.Index(genera, name="genus"))

    table["published_signed_q"] = published_q.reindex(genera)
    table["reproduced_signed_q"] = reproduced["signed_q_value"].reindex(genera)
    table["q_value_abs_diff"] = (
        table["reproduced_signed_q"] - table["published_signed_q"]
    ).abs()

    table["published_significant"] = _is_significant(table["published_signed_q"])
    table["reproduced_significant"] = _is_significant(table["reproduced_signed_q"])
    table["significance_agrees"] = (
        table["published_significant"] == table["reproduced_significant"]
    )

    both_significant = table["published_significant"] & table["reproduced_significant"]
    table["sign_agrees"] = pd.array([pd.NA] * len(table), dtype="boolean")
    table.loc[both_significant, "sign_agrees"] = np.sign(
        table.loc[both_significant, "published_signed_q"]
    ) == np.sign(table.loc[both_significant, "reproduced_signed_q"])

    table["published_effect"] = published_effect.reindex(genera)
    table["reproduced_log2_fold_change"] = reproduced["log2_fold_change"].reindex(
        genera
    )
    finite = np.isfinite(table["reproduced_log2_fold_change"])
    table["effect_abs_diff"] = np.where(
        finite,
        (table["reproduced_log2_fold_change"] - table["published_effect"]).abs(),
        np.nan,
    )
    return table


def _summarize(dataset_id: str, diff: pd.DataFrame) -> dict[str, object]:
    common = diff["published_signed_q"].notna() & diff["reproduced_signed_q"].notna()
    both_significant = diff["published_significant"] & diff["reproduced_significant"]
    finite_effect = diff["effect_abs_diff"].notna()
    return {
        "dataset_id": dataset_id,
        "n_published_genera": int(diff["published_signed_q"].notna().sum()),
        "n_reproduced_genera": int(diff["reproduced_signed_q"].notna().sum()),
        "n_missing_from_reproduction": int(
            (
                diff["published_signed_q"].notna() & diff["reproduced_signed_q"].isna()
            ).sum()
        ),
        "n_extra_in_reproduction": int(
            (
                diff["reproduced_signed_q"].notna() & diff["published_signed_q"].isna()
            ).sum()
        ),
        "n_published_significant": int(diff["published_significant"].sum()),
        "n_significance_agree": int(diff.loc[common, "significance_agrees"].sum()),
        "n_significance_common": int(common.sum()),
        "n_sign_agree": int((diff.loc[both_significant, "sign_agrees"] == True).sum()),  # noqa: E712
        "n_both_significant": int(both_significant.sum()),
        "max_q_diff_where_both_significant": float(
            diff.loc[both_significant, "q_value_abs_diff"].max()
        )
        if both_significant.any()
        else None,
        "n_finite_effect_comparisons": int(finite_effect.sum()),
        "max_effect_diff": float(diff.loc[finite_effect, "effect_abs_diff"].max())
        if finite_effect.any()
        else None,
    }


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
    if args.all:
        dataset_ids = analysed_dataset_ids(index)
    elif args.datasets:
        dataset_ids = args.datasets
    else:
        dataset_ids = M2_SUBSET

    s1 = pd.read_csv(args.data_root / "file-S1.qvalues.txt", sep="\t", index_col=0)
    s5 = pd.read_csv(args.data_root / "file-S5.effects.txt", sep="\t", index_col=0)

    args.out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    for dataset_id in dataset_ids:
        try:
            reproduced = differential_abundance(dataset_id, index, args.data_root)
        except EXPECTED_DATASET_ERRORS as exc:
            print(f"FAILED  {dataset_id}: {exc}", file=sys.stderr)
            failures.append((dataset_id, str(exc)))
            continue

        published_q = (
            s1[dataset_id] if dataset_id in s1.columns else pd.Series(dtype=float)
        )
        published_effect = (
            s5[dataset_id] if dataset_id in s5.columns else pd.Series(dtype=float)
        )
        diff = _diff_table(reproduced, published_q, published_effect)
        diff.to_csv(args.out / f"{dataset_id}.csv")

        summary = _summarize(dataset_id, diff)
        summaries.append(summary)
        print(
            f"{dataset_id}: {summary['n_significance_agree']}/"
            f"{summary['n_significance_common']} significance calls agree, "
            f"{summary['n_sign_agree']}/{summary['n_both_significant']} signs agree"
        )

    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {**_provenance(), "dataset_ids": dataset_ids, "failed": failures}, indent=2
        )
        + "\n"
    )

    print(f"\n{len(summaries)} dataset(s) reproduced, {len(failures)} failed")
    print(f"output written to {args.out}")
    # In --all mode a handful of known-mismatched datasets are expected to
    # fail (docs/reproduction-m2.md) and must not fail this script, but every
    # requested dataset failing indicates a systemic bug, not a documented
    # per-dataset gap, and should still be reported with a nonzero exit.
    if not dataset_ids or len(failures) == len(dataset_ids):
        return 1
    return 1 if failures and not args.all else 0


if __name__ == "__main__":
    raise SystemExit(main())
