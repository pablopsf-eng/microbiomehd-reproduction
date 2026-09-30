"""Independent Table 1 reconciliation for 4 real datasets, reading raw
``data/raw`` files directly with plain pandas -- ``mbhd.labels`` is never
imported, so this is not just "does the module agree with itself."

Covers 2 of the 5 currently-pinned ``mbhd.labels.KNOWN_MISMATCHES``
(``crc_wang``, ``liv_zhang``) and 2 of the datasets resolved via
``SAMPLE_TYPE_FILTER``/``CONDITION_FILTER`` (``ibd_morgan``, ``edd_singh``).

Ground truth: ``tests/data/table1_sample_counts.csv`` (hand-transcribed from
the published paper) and the real downloaded archive under ``data/raw``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


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
        encoding="latin-1",  # decodes any byte sequence; safe for a probe
        keep_default_na=False,
    )


def _load_otu_sample_ids(path: Path) -> set[str]:
    header = pd.read_csv(path, sep="\t", nrows=0, encoding="latin-1")
    return set(header.columns[1:])


def _reconcile(
    folder: str,
    metadata_fname: str,
    disease_col: str,
    controls: list[str],
    cases: list[str],
    extra_filter=None,
) -> tuple[int, int]:
    metadata = _load_metadata(folder, metadata_fname)
    if extra_filter is not None:
        metadata = extra_filter(metadata)
    otu_ids = _load_otu_sample_ids(_summary_otu_path(folder))
    observed = metadata.loc[metadata.index.isin(otu_ids), disease_col]
    return int(observed.isin(controls).sum()), int(observed.isin(cases).sum())


def test_crc_wang_known_mismatch_reproduces_pinned_observed_count():
    _require(DATA_ROOT / "dataset_info.yaml")
    # Published (54, 44), observed (56, 46).
    n_controls, n_cases = _reconcile(
        "crc_zhao_results",
        "crc_zhao.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["CRC"],
    )
    assert (n_controls, n_cases) == (56, 46)
    assert (n_controls, n_cases) != (54, 44)  # confirms the documented gap is real


def test_liv_zhang_known_mismatch_reproduces_pinned_observed_count():
    _require(DATA_ROOT / "dataset_info.yaml")
    # Published (25, 46), observed (26, 51).
    n_controls, n_cases = _reconcile(
        "mhe_zhang_results",
        "mhe_zhang.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["CIRR", "MHE"],
    )
    assert (n_controls, n_cases) == (26, 51)
    assert (n_controls, n_cases) != (25, 46)


def test_ibd_morgan_reconciles_exactly_once_stool_filter_applied():
    _require(DATA_ROOT / "dataset_info.yaml")

    # SAMPLE_TYPE_FILTER: without the
    # filter the count does not match Table 1; with it, it matches exactly.
    def stool_only(md: pd.DataFrame) -> pd.DataFrame:
        return md.loc[md["SampleType"] == "stool"]

    unfiltered = _reconcile(
        "ibd_huttenhower_results",
        "ibd_huttenhower.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["UC", "CD"],
    )
    filtered = _reconcile(
        "ibd_huttenhower_results",
        "ibd_huttenhower.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["UC", "CD"],
        extra_filter=stool_only,
    )
    assert filtered == (18, 108)  # Table 1's published (H, case) for ibd_morgan
    assert unfiltered != (18, 108)


def test_edd_singh_reconciles_exactly_once_condition_filter_applied():
    _require(DATA_ROOT / "dataset_info.yaml")

    # CONDITION_FILTER:
    # {"Time Point": [1]} -> (82, 201) exactly.
    def time_point_1(md: pd.DataFrame) -> pd.DataFrame:
        return md.loc[md["Time Point"].astype(str) == "1"]

    unfiltered = _reconcile(
        "edd_singh_results",
        "edd_singh.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["EDD"],
    )
    filtered = _reconcile(
        "edd_singh_results",
        "edd_singh.metadata.txt",
        "DiseaseState",
        controls=["H"],
        cases=["EDD"],
        extra_filter=time_point_1,
    )
    assert filtered == (82, 201)  # Table 1's published (H, case) for edd_singh
    assert unfiltered != (82, 201)
