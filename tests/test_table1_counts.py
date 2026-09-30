"""Reconcile loader-derived, real per-dataset sample counts against Table 1.

This is M1's literal done-when criterion:
for every one of the 30 analysed datasets, intersect the real metadata's
sample IDs with the real OTU table's sample IDs, group by the case/control
mapping in mbhd.labels, and compare to tests/data/table1_sample_counts.csv --
the hand-verified fixture, not a re-fetch. Skips (not fails) if data/raw is
absent, matching the project's established _require()/skip convention.
"""

from __future__ import annotations

import csv
import pathlib

import pytest

from mbhd.datasets import (
    analysed_dataset_ids,
    entry_for,
    load_dataset_info,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.io import encoding_for, load_metadata, load_otu_table
from mbhd.labels import CASE_CONTROL, KNOWN_MISMATCHES, reconcile_counts

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"
TABLE1_FIXTURE = _REPO_ROOT / "tests" / "data" / "table1_sample_counts.csv"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


#: Table 1's own "Dataset ID" text -> the real dataset_info.yaml ID. Table 1
#: names studies by first author + year + disease; dataset_info.yaml uses a
#: short prefix_author form. Transcribed by matching author/year/disease
#: across both sources -- every one of the 30 analysed IDs appears exactly
#: once (checked below).
TABLE1_NAME_TO_DATASET_ID = {
    "Singh 2015, EDD": "edd_singh",
    "Schubert 2014, CDI": "cdi_schubert",
    "Schubert 2014, non-CDI": "noncdi_schubert",
    "Vincent 2013, CDI": "cdi_vincent",
    "Youngster 2014, CDI": "cdi_youngster",
    "Goodrich 2014, OB": "ob_goodrich",
    "Turnbaugh 2009, OB": "ob_turnbaugh",
    "Zupancic 2012, OB": "ob_zupancic",
    "Ross 2015, OB": "ob_ross",
    "Zhu 2013, OB": "ob_zhu",
    "Baxter 2016, CRC": "crc_baxter",
    "Zeller 2014, CRC": "crc_zeller",
    "Wang 2012, CRC": "crc_wang",
    "Chen 2012, CRC": "crc_chen",
    "Gevers 2014, IBD": "ibd_gevers",
    "Morgan 2012, IBD": "ibd_morgan",
    "Papa 2012, IBD": "ibd_papa",
    "Willing 2010, IBD": "ibd_willing",
    "Noguera-Julian 2016, HIV": "hiv_noguerajulian",
    "Dinh 2015, HIV": "hiv_dinh",
    "Lozupone 2013, HIV": "hiv_lozupone",
    "Son 2015, ASD": "asd_son",
    "Kang 2013, ASD": "asd_kang",
    "Alkanani 2015, T1D": "t1d_alkanani",
    "Mejia-Leon 2014, T1D": "t1d_mejialeon",
    "Wong 2013, NASH": "nash_wong",
    "Zhu 2013, NASH": "nash_zhu",
    "Scher 2013, ART": "art_scher",
    "Zhang 2013, LIV": "liv_zhang",
    "Scheperjans 2015, PAR": "par_scheperjans",
}


def _load_table1_fixture() -> dict[str, tuple[int, int]]:
    """{dataset_id: (n_controls, n_cases)} from the hand-verified fixture."""
    with open(TABLE1_FIXTURE) as f:
        rows = [row for row in f if not row.startswith("#")]
    result = {}
    for row in csv.DictReader(rows):
        dataset_id = TABLE1_NAME_TO_DATASET_ID[row["dataset_id"]]
        result[dataset_id] = (int(row["n_controls"]), int(row["n_cases"]))
    return result


def test_table1_name_mapping_covers_every_analysed_dataset_exactly_once() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    assert sorted(TABLE1_NAME_TO_DATASET_ID.values()) == analysed_dataset_ids(index)


def test_case_control_spec_exists_for_every_analysed_dataset() -> None:
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    for dataset_id in analysed_dataset_ids(index):
        assert dataset_id in CASE_CONTROL


def test_reconciliation_against_table1() -> None:
    """The actual done-when check.

    Every analysed dataset is compared against Table 1. Verified-matching
    datasets must match exactly (a regression here means a real bug).
    Datasets in KNOWN_MISMATCHES are allowed to disagree with Table 1, but
    only by exactly the pinned amount -- if their observed count changes at
    all, that is new information and must be
    investigated and re-pinned, not silently absorbed.
    """
    _require(DATASET_INFO, TABLE1_FIXTURE)
    index = load_dataset_info(DATASET_INFO)
    published = _load_table1_fixture()

    report_lines = [
        f"{'dataset':20s} {'pub_ctrl':>8s} {'obs_ctrl':>8s} "
        f"{'pub_case':>8s} {'obs_case':>8s}  status"
    ]
    unexpected_mismatches: list[str] = []
    drifted_known_mismatches: list[str] = []

    for dataset_id in analysed_dataset_ids(index):
        folder = DATA_RAW / entry_for(index, dataset_id).folder
        if not folder.is_dir():
            pytest.skip(f"data not fetched: {folder.name}")

        sd = study_dir(index, dataset_id, DATA_RAW)
        metadata = load_metadata(
            metadata_path(dataset_id, sd), encoding=encoding_for(dataset_id)
        )
        otu = load_otu_table(otu_table_path(dataset_id, sd))
        counts = reconcile_counts(dataset_id, metadata, set(otu.columns))

        pub_ctrl, pub_case = published[dataset_id]
        matches = counts.controls == pub_ctrl and counts.cases == pub_case

        if dataset_id in KNOWN_MISMATCHES:
            expected_ctrl, expected_case = KNOWN_MISMATCHES[dataset_id]
            if (counts.controls, counts.cases) != (expected_ctrl, expected_case):
                drifted_known_mismatches.append(
                    f"{dataset_id}: pinned ({expected_ctrl}, {expected_case}), "
                    f"now observed ({counts.controls}, {counts.cases})"
                )
            status = "known mismatch (pinned)" if matches is False else "NOW MATCHES?"
        elif not matches:
            unexpected_mismatches.append(
                f"{dataset_id}: published ({pub_ctrl}, {pub_case}), "
                f"observed ({counts.controls}, {counts.cases})"
            )
            status = "*** UNEXPECTED MISMATCH ***"
        else:
            status = "ok"

        report_lines.append(
            f"{dataset_id:20s} {pub_ctrl:8d} {counts.controls:8d} "
            f"{pub_case:8d} {counts.cases:8d}  {status}"
        )

    report = "\n".join(report_lines)
    assert not unexpected_mismatches, (
        f"{len(unexpected_mismatches)} dataset(s) newly disagree with Table 1 "
        f"(not in KNOWN_MISMATCHES):\n"
        + "\n".join(unexpected_mismatches)
        + "\n\nFull report:\n"
        + report
    )
    assert not drifted_known_mismatches, (
        "Observed counts changed for dataset(s) already in KNOWN_MISMATCHES -- "
        "re-investigate and re-pin mbhd.labels.KNOWN_MISMATCHES:\n"
        + "\n".join(drifted_known_mismatches)
    )
