#!/usr/bin/env python3
"""Tests the unconfirmed CLR tie-correction hypothesis named in
docs/reproduction-m4.md's Part A: does CLR's zero-replacement, by turning
tied-at-zero cells into small but distinct values, make Kruskal-Wallis's
tie correction less conservative, and is *that* what drives the larger
CLR-transformed shared-response pool (248 vs. 156 genera, full-29)?

**A one-off diagnostic, not a pipeline component.** `mbhd.stats.
kruskal_wallis` is not modified -- it is heavily verified, M2-era code
every existing caller depends on; changing its return shape for a one-off
question isn't worth the risk. Instead this script recomputes the same
tie-correction factor `C` and uncorrected statistic `H_uncorrected` the
function's own docstring already states the formula for
(`C = 1 - sum(t_i**3-t_i) / (N**3-N)`), reusing `mbhd.stats._average_ranks`
for the ranking step, and self-checks every single recomputed `(H, p)`
against `mbhd.stats.kruskal_wallis`'s own real output on the identical
input arrays -- if that check ever fails, the script aborts loudly (a bug
in the duplicated formula, not an interesting finding).

**The direction was not assumed going in.** `H = H_uncorrected / C`, and
`C` is *smaller* when ties are heavier -- dividing by a smaller `C` makes
`H` *larger*, not smaller. So, holding `H_uncorrected` fixed, heavier ties
push `H` up and `p` down (more significant), the opposite of the
originally-guessed "fewer ties -> smaller p-values" direction. This script
measures both the tie-correction effect and the (very possibly dominant,
un-named) `H_uncorrected` effect separately, and reports whichever one the
real data actually supports.

Usage:
    uv run scripts/clr_tie_investigation.py
    uv run scripts/clr_tie_investigation.py --out results/clr_tie_investigation \
        --data-root data/raw
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from mbhd.abundance import analysis_samples, filter_table, to_genus_abundance
from mbhd.compositional import clr, multiplicative_replacement
from mbhd.datasets import (
    analysed_dataset_ids,
    load_dataset_info,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.differential import EXPECTED_DATASET_ERRORS, NoSurvivingSamplesError
from mbhd.io import DISEASE_STATE_COLUMN, encoding_for, load_metadata, load_otu_table
from mbhd.labels import (
    resolved_case_control,
    restrict_to_condition,
    restrict_to_sample_type,
)
from mbhd.stats import _average_ranks, kruskal_wallis

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "clr_tie_investigation"

#: Not a judgment call -- same exclusion M2/M3/M4 already document
#: (ZeroReadSampleError before any per-genus test can run at all).
_EXCLUDED = frozenset({"ob_zupancic"})

#: p-values are clamped to this floor before log10, in case a sufficiently
#: extreme comparison ever underflows erfc() to an exact 0.0 in float64
#: (not observed in the real full-29 run -- minimum p there is ~4e-30) --
#: a documented defensive floor, not a silent -inf/nan if it ever happens.
_P_FLOOR = 1e-300


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    return parser.parse_args(argv)


def _tie_correction_and_h_uncorrected(
    control: np.ndarray, case: np.ndarray
) -> tuple[float, float]:
    """Recomputes exactly what mbhd.stats.kruskal_wallis computes internally
    (same formula, its own docstring), but returns the two intermediate
    quantities (C, H_uncorrected) instead of only the final (H, p)."""
    all_values = np.concatenate([control, case])
    n_total = all_values.size

    ranks = _average_ranks(all_values)
    rank_sum_control = ranks[: control.size].sum()
    rank_sum_case = ranks[control.size :].sum()

    h_uncorrected = (12.0 / (n_total * (n_total + 1))) * (
        rank_sum_control**2 / control.size + rank_sum_case**2 / case.size
    ) - 3 * (n_total + 1)

    _, tie_sizes = np.unique(all_values, return_counts=True)
    tie_term = np.sum(tie_sizes**3 - tie_sizes)
    tie_denominator = n_total**3 - n_total
    correction = 1.0 - tie_term / tie_denominator if tie_denominator > 0 else 0.0
    return correction, h_uncorrected


def _self_check(
    control: np.ndarray, case: np.ndarray, correction: float, h_uncorrected: float
) -> tuple[float, float]:
    """Combines (C, H_uncorrected) into (H, p) the same way kruskal_wallis
    does, and asserts it matches kruskal_wallis's own real output on the
    identical arrays -- the correctness gate for this whole script."""
    h = 0.0 if correction <= 0 else max(h_uncorrected / correction, 0.0)
    p_value = math.erfc(math.sqrt(h / 2))

    real_h, real_p = kruskal_wallis(control, case)
    if not (math.isclose(h, real_h, rel_tol=1e-9, abs_tol=1e-12)):
        raise AssertionError(
            f"recomputed H={h!r} does not match kruskal_wallis's own "
            f"H={real_h!r} -- formula duplication bug, not a finding"
        )
    if not (math.isclose(p_value, real_p, rel_tol=1e-9, abs_tol=1e-12)):
        raise AssertionError(
            f"recomputed p={p_value!r} does not match kruskal_wallis's own "
            f"p={real_p!r} -- formula duplication bug, not a finding"
        )
    return h, p_value


def _rows_for_dataset(
    dataset_id: str, index, data_root: Path
) -> list[dict[str, object]]:
    folder = study_dir(index, dataset_id, data_root)
    metadata = load_metadata(
        metadata_path(dataset_id, folder), encoding=encoding_for(dataset_id)
    )
    metadata = restrict_to_sample_type(dataset_id, metadata)
    metadata = restrict_to_condition(dataset_id, metadata)
    otu_table = load_otu_table(otu_table_path(dataset_id, folder))

    sample_ids = analysis_samples(metadata, otu_table.columns)
    filtered = filter_table(otu_table, sample_ids)
    genus_euclidean = to_genus_abundance(filtered)
    genus_clr = clr(genus_euclidean, pseudocount_fn=multiplicative_replacement)

    spec = resolved_case_control(dataset_id)
    control_ids = [
        s
        for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.controls)]
        if s in genus_euclidean.columns
    ]
    case_ids = [
        s
        for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.cases)]
        if s in genus_euclidean.columns
    ]
    if not control_ids or not case_ids:
        raise NoSurvivingSamplesError(
            f"{dataset_id}: {len(control_ids)} control(s), {len(case_ids)} "
            "case(s) left after filtering -- need at least one of each"
        )

    rows: list[dict[str, object]] = []
    for genus in genus_euclidean.index:
        rec: dict[str, object] = {"dataset_id": dataset_id, "genus": genus}
        for label, table in (("euclidean", genus_euclidean), ("clr", genus_clr)):
            control = table.loc[genus, control_ids].to_numpy(dtype=float)
            case = table.loc[genus, case_ids].to_numpy(dtype=float)
            correction, h_uncorrected = _tie_correction_and_h_uncorrected(control, case)
            h, p_value = _self_check(control, case, correction, h_uncorrected)
            rec[f"C_{label}"] = correction
            rec[f"H_uncorrected_{label}"] = h_uncorrected
            rec[f"H_{label}"] = h
            rec[f"p_{label}"] = p_value
        rows.append(rec)
    return rows


def _provenance() -> dict[str, str]:
    versions = {}
    for package in ("pandas", "numpy", "pyyaml"):
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
    args.out.mkdir(parents=True, exist_ok=True)

    index = load_dataset_info(args.data_root / "dataset_info.yaml")
    dataset_ids = [d for d in analysed_dataset_ids(index) if d not in _EXCLUDED]

    all_rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for dataset_id in dataset_ids:
        try:
            all_rows.extend(_rows_for_dataset(dataset_id, index, args.data_root))
        except EXPECTED_DATASET_ERRORS as exc:
            failures[dataset_id] = str(exc)
            print(f"SKIPPED {dataset_id}: {exc}", file=sys.stderr)

    if not all_rows:
        print("no comparisons computed -- cannot test the hypothesis", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)
    df["delta_C"] = df["C_clr"] - df["C_euclidean"]
    df["delta_H_uncorrected"] = df["H_uncorrected_clr"] - df["H_uncorrected_euclidean"]
    df["neg_log10_p_euclidean"] = -np.log10(df["p_euclidean"].clip(lower=_P_FLOOR))
    df["neg_log10_p_clr"] = -np.log10(df["p_clr"].clip(lower=_P_FLOOR))
    df["delta_neg_log10_p"] = df["neg_log10_p_clr"] - df["neg_log10_p_euclidean"]

    df.to_csv(args.out / "comparisons.csv", index=False)

    n = len(df)
    mean_c_euclidean = float(df["C_euclidean"].mean())
    mean_c_clr = float(df["C_clr"].mean())
    corr_delta_c = float(df["delta_C"].corr(df["delta_neg_log10_p"]))
    corr_delta_h_unc = float(df["delta_H_uncorrected"].corr(df["delta_neg_log10_p"]))

    n_datasets_used = len(dataset_ids) - len(failures)
    print(f"{n} genus/dataset comparisons across {n_datasets_used} datasets")
    print(
        f"mean C (tie correction factor): euclidean={mean_c_euclidean:.4f}, "
        f"clr={mean_c_clr:.4f}"
    )
    print(
        "premise check (does CLR have less tie-correction, i.e. "
        f"C_clr > C_euclidean?): {'YES' if mean_c_clr > mean_c_euclidean else 'NO'} "
        f"(delta_C mean={float(df['delta_C'].mean()):.4f})"
    )
    print(
        f"correlation(delta_C, delta_neg_log10_p) = {corr_delta_c:.4f}  "
        "(the named tie-correction mechanism)"
    )
    print(
        "correlation(delta_H_uncorrected, delta_neg_log10_p) = "
        f"{corr_delta_h_unc:.4f}  (the competing, un-named rank-separation effect)"
    )

    summary = {
        "n_comparisons": n,
        "n_datasets": len(dataset_ids) - len(failures),
        "mean_C_euclidean": mean_c_euclidean,
        "mean_C_clr": mean_c_clr,
        "mean_delta_C": float(df["delta_C"].mean()),
        "corr_delta_C_vs_delta_neg_log10_p": corr_delta_c,
        "corr_delta_H_uncorrected_vs_delta_neg_log10_p": corr_delta_h_unc,
        "failed": failures,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "provenance.json").write_text(
        json.dumps(_provenance(), indent=2) + "\n"
    )
    print(f"\noutput written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
