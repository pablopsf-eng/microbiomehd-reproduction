"""Independent re-derivation of the 3 remaining ``mbhd.labels.KNOWN_MISMATCHES``
entries not already covered by ``test_labels_reconciliation_verification.py``
(``crc_wang``, ``liv_zhang``): ``hiv_noguerajulian``, ``ob_turnbaugh``,
``ob_zupancic``.

Reads ``dataset_info.yaml`` and the raw metadata/OTU-table files directly with
plain pandas -- ``mbhd.labels`` is imported only to read the pinned
``KNOWN_MISMATCHES`` values and the ``CASE_CONTROL`` label mapping being
reconciled against, never to compute the reconciliation itself
(``reconcile_counts`` is never called here).

Ground truth: the currently-observed (H, case) counts for each of the 5
unresolved datasets (pinned in ``KNOWN_MISMATCHES``), and the real
downloaded archive under ``data/raw``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mbhd.labels import CASE_CONTROL, KNOWN_MISMATCHES

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _folder_for(dataset_id: str) -> str:
    info = yaml.safe_load((DATA_ROOT / "dataset_info.yaml").read_text())
    return info[dataset_id]["folder"]


def _summary_otu_path(folder: str) -> Path:
    summary = (DATA_ROOT / folder / "summary_file.txt").read_text()
    for line in summary.splitlines():
        if line.startswith("OTU_TABLE_RDP"):
            fname = line.split("\t", 1)[1].strip()
            return DATA_ROOT / folder / "RDP" / fname
    raise ValueError(f"{folder}: no OTU_TABLE_RDP key in summary_file.txt")


def _load_metadata(folder: str, fname: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA_ROOT / folder / fname,
        sep="\t",
        index_col=0,
        dtype={0: str},
        encoding="latin-1",
        keep_default_na=False,
    )


def _load_otu_sample_ids(path: Path) -> set[str]:
    header = pd.read_csv(path, sep="\t", nrows=0, encoding="latin-1")
    return set(header.columns[1:])


def _reconcile(
    folder: str, metadata_fname: str, controls: list[str], cases: list[str]
) -> tuple[int, int]:
    metadata = _load_metadata(folder, metadata_fname)
    otu_ids = _load_otu_sample_ids(_summary_otu_path(folder))
    observed = metadata.loc[metadata.index.isin(otu_ids), "DiseaseState"]
    return int(observed.isin(controls).sum()), int(observed.isin(cases).sum())


@pytest.mark.parametrize(
    ("dataset_id", "folder", "metadata_fname"),
    [
        (
            "hiv_noguerajulian",
            "hiv_noguerajulian_results",
            "hiv_noguerajulian.metadata.txt",
        ),
        ("ob_turnbaugh", "ob_gordon_2008_v2_results", "ob_gordon_2008_v2.metadata.txt"),
        ("ob_zupancic", "ob_zupancic_results", "ob_zupancic.metadata.txt"),
    ],
)
def test_known_mismatch_reproduces_pinned_observed_count(
    dataset_id: str, folder: str, metadata_fname: str
):
    _require(DATA_ROOT / "dataset_info.yaml")
    spec = CASE_CONTROL[dataset_id]
    n_controls, n_cases = _reconcile(folder, metadata_fname, spec.controls, spec.cases)
    assert (n_controls, n_cases) == KNOWN_MISMATCHES[dataset_id]
