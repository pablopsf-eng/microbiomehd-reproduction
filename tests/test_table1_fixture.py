"""Tests for the Table 1 sample-count fixture.

tests/data/table1_sample_counts.csv is committed, hand-verified ground truth
(see its own header comment for provenance) -- these tests run offline, with
no dependency on data/raw/ or the network.
"""

from __future__ import annotations

import csv
import pathlib

FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / "table1_sample_counts.csv"
RAW_SOURCE = FIXTURE.resolve().parents[2] / "docs" / "datasets_table.csv"


def _read_rows() -> list[dict[str, str]]:
    lines = FIXTURE.read_text().splitlines()
    data_lines = [line for line in lines if not line.startswith("#")]
    return list(csv.DictReader(data_lines))


def test_fixture_matches_the_raw_source_byte_for_byte():
    """The fixture only renames docs/datasets_table.csv's header row -- every
    data value must be identical, so the two can never silently drift apart.
    """
    raw_rows = list(csv.reader(RAW_SOURCE.open(newline="")))[1:]
    with FIXTURE.open(newline="") as f:
        lines = f.readlines()
    data_lines = "".join(line for line in lines if not line.startswith("#"))
    fixture_rows = list(csv.reader(data_lines.splitlines()))[1:]
    assert fixture_rows == raw_rows


def test_row_count_matches_the_independently_derived_count():
    """30 rows, cross-checked against an independent count: 30 analysed
    datasets = 30 columns in file-S1.qvalues.txt = 30 Table 1 rows -- a count
    derived from the supplementary files, not from Table 1 itself.
    """
    rows = _read_rows()
    assert len(rows) == 30


def test_dataset_ids_are_unique():
    rows = _read_rows()
    dataset_ids = [row["dataset_id"] for row in rows]
    assert len(dataset_ids) == len(set(dataset_ids))


def test_sample_counts_are_positive_integers():
    rows = _read_rows()
    for row in rows:
        n_controls = int(row["n_controls"])
        n_cases = int(row["n_cases"])
        assert n_controls > 0, row
        assert n_cases > 0, row


def test_schubert_rows_share_the_same_control_count():
    """The CDI and non-CDI Schubert rows draw on the same control set per the
    paper's Methods (each was used as an independent case group against the
    same controls) -- their n_controls must be identical. A mismatch would
    mean a transcription error, not a data quirk.
    """
    rows = _read_rows()
    schubert = [row for row in rows if row["dataset_id"].startswith("Schubert 2014")]
    assert len(schubert) == 2
    n_controls = {row["n_controls"] for row in schubert}
    assert n_controls == {"154"}
