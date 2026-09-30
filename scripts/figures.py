#!/usr/bin/env python3
"""Generate M3's in-scope figure equivalents from this project's own reproduced numbers.

**Scope (decided 2026-09-17):** Fig 1a (per-study classifier AUC) is descoped to M4 --
no classifier exists anywhere in this project, and building one properly
here would quietly become M4's actual content. Fig 3a (phylogenetic
presence/absence track) is out of scope permanently -- no phylogeny is ever
built in this project. The four remaining panels (1b, 2, 3b, 3c) are
generated here as **equivalent plots from this project's own reproduced
numbers, not pixel-identical to the published images** -- a rough-tolerance,
secondary check (the published figures can be read only to a rough tolerance,
not to the precision of the supplementary files), not the primary numeric
target (the diff CSVs from `scripts/reproduce.py`/`scripts/shared_
response.py` are that).

Each figure is saved with axis labels, units, and printed with a one-line
caption stating a concrete, checkable conclusion.
Captions are also written to `docs/reproduction-m3.md` by hand from this
script's own output, alongside the numbers behind each figure -- a figure
is never shipped as an image alone.

Usage:
    uv run scripts/figures.py
    uv run scripts/figures.py --out results/figures --data-root data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.differential import EXPECTED_DATASET_ERRORS, differential_abundance
from mbhd.shared_response import (
    PUBLISHED_EXCLUSIONS,
    QValueRows,
    is_significant,
    shared_response_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "raw"
_DEFAULT_OUT = _REPO_ROOT / "results" / "figures"


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


def _compute_per_dataset(dataset_ids, index, data_root):
    """{dataset_id: differential_abundance's DataFrame}, failures recorded."""
    per_dataset = {}
    failures = {}
    for dataset_id in dataset_ids:
        try:
            per_dataset[dataset_id] = differential_abundance(
                dataset_id, index, data_root
            )
        except EXPECTED_DATASET_ERRORS as exc:
            failures[dataset_id] = str(exc)
    return per_dataset, failures


def _rows_from_per_dataset(per_dataset: dict, dataset_ids: list[str]) -> QValueRows:
    """The same outer-join shape as mbhd.shared_response.build_qvalue_matrix,
    built directly from an already-computed per_dataset dict instead of
    calling differential_abundance a second time (fig1b/fig3c already need
    per_dataset for their own abundance/count numbers)."""
    all_genera: set[str] = set()
    for df in per_dataset.values():
        all_genera.update(df.index)
    return {
        genus: {
            dataset_id: (
                float(per_dataset[dataset_id].loc[genus, "signed_q_value"])
                if dataset_id in per_dataset and genus in per_dataset[dataset_id].index
                else None
            )
            for dataset_id in dataset_ids
        }
        for genus in sorted(all_genera)
    }


def fig1b(per_dataset: dict, out_dir: Path) -> str:
    """Per-dataset significant-genus counts, split by direction."""
    case_counts, control_counts = {}, {}
    for dataset_id, df in per_dataset.items():
        sig = df[df["signed_q_value"].apply(is_significant)]
        case_counts[dataset_id] = int((sig["signed_q_value"] > 0).sum())
        control_counts[dataset_id] = int((sig["signed_q_value"] < 0).sum())

    ids = sorted(case_counts, key=lambda d: case_counts[d] + control_counts[d])
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(ids) + 1))
    ax.barh(
        y,
        [control_counts[d] for d in ids],
        color="#4C72B0",
        label="control-enriched (q < 0.05)",
    )
    ax.barh(
        y,
        [case_counts[d] for d in ids],
        left=[control_counts[d] for d in ids],
        color="#DD8452",
        label="case-enriched (q < 0.05)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel("number of significant genera (Benjamini-Hochberg q < 0.05)")
    ax.set_title("Fig 1b-equivalent: significant genera per dataset, by direction")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1b_significant_genera_per_dataset.png", dpi=150)
    plt.close(fig)

    n_case_dominant = sum(1 for d in ids if case_counts[d] > control_counts[d])
    return (
        f"Fig 1b-equivalent: {n_case_dominant} of {len(ids)} reproduced datasets "
        "show more case-enriched than control-enriched significant genera "
        "(this project's own from-scratch Kruskal-Wallis + Benjamini-Hochberg "
        "pipeline, not the published file-S1 values)."
    )


def fig2(
    rows: QValueRows, dataset_ids: list[str], n_reproduced: int, out_dir: Path
) -> str:
    """Heatmap of the reproduced signed-significance matrix itself."""
    pool = [g for g, qs in rows.items() if any(is_significant(q) for q in qs.values())]
    pool.sort(key=lambda g: -sum(is_significant(rows[g][d]) for d in dataset_ids))

    matrix = np.full((len(pool), len(dataset_ids)), np.nan)
    for i, genus in enumerate(pool):
        for j, dataset_id in enumerate(dataset_ids):
            q = rows[genus][dataset_id]
            if q is None:
                continue
            matrix[i, j] = np.sign(q) if is_significant(q) else 0.0

    fig, ax = plt.subplots(figsize=(10, max(6, 0.03 * len(pool))))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(dataset_ids)))
    ax.set_xticklabels(dataset_ids, rotation=90, fontsize=5)
    ax.set_yticks([])
    ax.set_xlabel("dataset")
    ax.set_ylabel(f"genus (n={len(pool)}, significant in >=1 dataset)")
    ax.set_title("Fig 2-equivalent: reproduced signed-significance matrix")
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(
        ["control-enriched\n(q<0.05)", "not significant", "case-enriched\n(q<0.05)"]
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_qvalue_matrix.png", dpi=150)
    plt.close(fig)

    return (
        f"Fig 2-equivalent: {len(pool)} genera reach Benjamini-Hochberg q<0.05 in "
        f"at least one of {n_reproduced} reproduced datasets ({len(dataset_ids)} "
        "attempted); grey cells mark a genus absent from that dataset after "
        "study-level filtering, not a zero."
    )


def fig3b(
    rows: QValueRows,
    dataset_ids: list[str],
    shared_labels: dict[str, str],
    out_dir: Path,
) -> str:
    """Per dataset: fraction of its own significant genera in the shared pool."""
    fractions = {}
    for dataset_id in dataset_ids:
        sig_genera = [g for g in rows if is_significant(rows[g].get(dataset_id))]
        if not sig_genera:
            continue
        n_shared = sum(1 for g in sig_genera if shared_labels.get(g, "") != "")
        fractions[dataset_id] = n_shared / len(sig_genera)

    ids = sorted(fractions, key=lambda d: fractions[d])
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(ids) + 1))
    ax.barh(range(len(ids)), [fractions[d] for d in ids], color="#55A868")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel(
        "fraction of a dataset's own significant genera in the shared-response pool"
    )
    ax.set_xlim(0, 1)
    ax.set_title("Fig 3b-equivalent: shared-response overlap per dataset")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3b_overlap_per_dataset.png", dpi=150)
    plt.close(fig)

    mean_frac = float(np.mean(list(fractions.values())))
    return (
        f"Fig 3b-equivalent: across {len(ids)} datasets with >=1 significant genus, "
        f"a mean of {mean_frac:.0%} of a dataset's own significant genera also belong "
        "to the cross-disease shared-response pool (full-29 run, GitHub-style "
        f"exclusions {sorted(PUBLISHED_EXCLUSIONS)} applied when defining the pool)."
    )


def fig3c(
    rows: QValueRows,
    per_dataset: dict,
    shared_labels: dict[str, str],
    out_dir: Path,
) -> str:
    """Per-genus abundance and cross-study ubiquity, shared vs. disease-specific."""
    pool = [g for g in rows if any(is_significant(q) for q in rows[g].values())]
    n_attempted = len(per_dataset)

    abundance: dict[str, float] = {}
    prevalence: dict[str, float] = {}
    for genus in pool:
        present = [d for d, q in rows[genus].items() if q is not None]
        means = [
            (
                per_dataset[d].loc[genus, "control_mean"]
                + per_dataset[d].loc[genus, "case_mean"]
            )
            / 2
            for d in present
            if d in per_dataset and genus in per_dataset[d].index
        ]
        if means:
            abundance[genus] = float(np.mean(means))
        prevalence[genus] = len(present) / n_attempted

    shared_genera = [g for g in pool if shared_labels.get(g, "") != ""]
    specific_genera = [g for g in pool if shared_labels.get(g, "") == ""]
    shared_abund = [abundance[g] for g in shared_genera if abundance.get(g, 0) > 0]
    specific_abund = [abundance[g] for g in specific_genera if abundance.get(g, 0) > 0]
    shared_prev = [prevalence[g] for g in shared_genera]
    specific_prev = [prevalence[g] for g in specific_genera]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].boxplot(
        [np.log10(shared_abund), np.log10(specific_abund)],
        tick_labels=["shared", "specific"],
    )
    axes[0].set_ylabel("log10(mean relative abundance)")
    axes[0].set_title("abundance")
    axes[1].boxplot([shared_prev, specific_prev], tick_labels=["shared", "specific"])
    axes[1].set_ylabel("fraction of reproduced\ndatasets genus survives filtering in")
    axes[1].set_title("cross-study ubiquity")
    fig.suptitle("Fig 3c-equivalent: shared vs. disease-specific pool genera")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(out_dir / "fig3c_abundance_ubiquity.png", dpi=150)
    plt.close(fig)

    return (
        f"Fig 3c-equivalent: shared-response genera (n={len(shared_abund)}) have "
        f"median relative abundance {np.median(shared_abund):.2e} vs. "
        f"disease-specific genera (n={len(specific_abund)}) at "
        f"{np.median(specific_abund):.2e}; median cross-study ubiquity "
        f"{np.median(shared_prev):.0%} (shared) vs. {np.median(specific_prev):.0%} "
        "(specific). Ubiquity here is the fraction of this project's own reproduced "
        "datasets a genus survives study-level filtering in -- a cross-study proxy "
        "for the paper's own within-study sample prevalence, not identical to it."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)

    index = load_dataset_info(args.data_root / "dataset_info.yaml")
    dataset_ids = analysed_dataset_ids(index)

    per_dataset, failures = _compute_per_dataset(dataset_ids, index, args.data_root)
    if not per_dataset:
        print("no dataset reproduced -- cannot generate figures", file=sys.stderr)
        return 1
    if failures:
        print(f"skipped (documented failure): {failures}", file=sys.stderr)

    rows = _rows_from_per_dataset(per_dataset, dataset_ids)
    shared_labels = shared_response_labels(rows, PUBLISHED_EXCLUSIONS)

    captions = [
        fig1b(per_dataset, args.out),
        fig2(rows, dataset_ids, len(per_dataset), args.out),
        fig3b(rows, dataset_ids, shared_labels, args.out),
        fig3c(rows, per_dataset, shared_labels, args.out),
    ]
    for caption in captions:
        print(caption)

    (args.out / "captions.txt").write_text("\n".join(captions) + "\n")
    print(f"\nfigures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
