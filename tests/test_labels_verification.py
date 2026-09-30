"""Independent verification of mbhd.labels.CONDITION_FILTER against real data.

Unlike tests/test_labels.py (synthetic fixtures) and tests/test_table1_counts.py
(exercises reconcile_counts end-to-end for all 30 datasets), this file
re-derives the CONDITION_FILTER claim from scratch for the specific four
datasets it touches, without importing CONDITION_FILTER's *values* -- it reads
dataset_info.yaml's raw condition: field directly and applies it by hand with
plain pandas, then compares against the same reconciliation primitive
(resolved_case_control + DiseaseState intersection) reconcile_counts uses
internally. This catches a wrong entry in CONDITION_FILTER that a test which
merely calls restrict_to_condition() and checks it doesn't crash would miss.

Also independently exercises the isin()-based multi-value branch of
restrict_to_condition (no adopted production CONDITION_FILTER entry needs
more than one kept value, so this path is otherwise only covered by
tests/test_labels.py's own synthetic test of the same function):
test_restrict_to_condition_multi_value_path_actually_uses_isin calls the
real function with a monkeypatched synthetic two-value entry, a second,
independently-written check of the same contract.

Skips (not fails) if data/raw is absent, matching the project's established
_require() convention (see tests/test_table1_counts.py).
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from mbhd.datasets import (
    load_dataset_info,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.io import DISEASE_STATE_COLUMN, encoding_for, load_metadata, load_otu_table
from mbhd.labels import resolved_case_control, restrict_to_condition

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"


def _require() -> None:
    if not DATASET_INFO.exists():
        pytest.skip("data not fetched: dataset_info.yaml")


def _observed_counts(
    dataset_id: str, metadata: pd.DataFrame, otu_ids: set
) -> tuple[int, int]:
    spec = resolved_case_control(dataset_id)
    observed = metadata.loc[metadata.index.isin(otu_ids), DISEASE_STATE_COLUMN]
    return int(observed.isin(spec.controls).sum()), int(observed.isin(spec.cases).sum())


def _load(dataset_id: str, index):
    folder = study_dir(index, dataset_id, DATA_RAW)
    metadata = load_metadata(
        metadata_path(dataset_id, folder), encoding=encoding_for(dataset_id)
    )
    otu = load_otu_table(otu_table_path(dataset_id, folder))
    return metadata, set(otu.columns)


#: dataset_id -> Table 1's published (controls, cases), transcribed from
#: tests/data/table1_sample_counts.csv (edd_singh, hiv_lozupone, nash_wong,
#: ob_goodrich rows).
PUBLISHED = {
    "edd_singh": (82, 201),
    "hiv_lozupone": (13, 23),
    "nash_wong": (22, 16),
    "ob_goodrich": (428, 185),
}


@pytest.mark.parametrize("dataset_id", sorted(PUBLISHED))
def test_condition_field_from_yaml_reproduces_table1_exactly(dataset_id: str) -> None:
    """Reads dataset_info.yaml's raw condition: field directly (not
    CONDITION_FILTER) and applies it by hand -- an independent re-derivation,
    not a re-check of the same lookup table under test."""
    _require()
    index = load_dataset_info(DATASET_INFO)
    condition = index[dataset_id].raw.get("condition")
    assert condition, f"{dataset_id}: dataset_info.yaml has no condition: field"
    assert len(condition) == 1, f"{dataset_id}: expected exactly one condition column"
    column, keep_values = next(iter(condition.items()))

    metadata, otu_ids = _load(dataset_id, index)
    assert column in metadata.columns
    filtered = metadata.loc[metadata[column].isin(keep_values)]

    assert _observed_counts(dataset_id, filtered, otu_ids) == PUBLISHED[dataset_id]


@pytest.mark.parametrize(
    "dataset_id,published",
    [("hiv_noguerajulian", (34, 205)), ("ob_zupancic", (96, 101))],
)
def test_excluded_datasets_condition_field_does_not_reproduce_table1(
    dataset_id: str, published: tuple[int, int]
) -> None:
    """hiv_noguerajulian and ob_zupancic both have a condition: field but are
    deliberately absent from CONDITION_FILTER. Confirms applying their field
    gets *close* but not exact -- the documented reason they were left out,
    not silently assumed."""
    _require()
    index = load_dataset_info(DATASET_INFO)
    condition = index[dataset_id].raw.get("condition")
    assert condition, f"{dataset_id}: dataset_info.yaml has no condition: field"
    column, keep_values = next(iter(condition.items()))

    metadata, otu_ids = _load(dataset_id, index)
    filtered = metadata.loc[metadata[column].isin(keep_values)]
    observed = _observed_counts(dataset_id, filtered, otu_ids)

    assert observed != published, (
        f"{dataset_id}: applying its own condition: field now reproduces "
        f"Table 1 exactly ({observed}) -- CONDITION_FILTER should be updated "
        "to include it, this test result is new information"
    )


def test_restrict_to_condition_multi_value_path_actually_uses_isin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No production CONDITION_FILTER entry has more than one kept value
    today, so this path is otherwise only covered by tests/test_labels.py's
    own test_restrict_to_condition_supports_multiple_kept_values. This test
    monkeypatches CONDITION_FILTER with a synthetic two-value entry and
    calls the real function, a second, independently-written check of the
    same contract, so a regression (e.g. swapping .isin() for `==`, which
    silently keeps zero rows for a tuple) is actually caught."""
    import mbhd.labels as labels_module

    monkeypatch.setitem(
        labels_module.CONDITION_FILTER, "cdi_schubert", ("cohort", ("BCN0", "STK"))
    )
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "CDI"],
            "cohort": ["BCN0", "BCN1", "STK"],
        },
        index=["S1", "S2", "S3"],
    )
    result = restrict_to_condition("cdi_schubert", metadata)
    assert list(result.index) == ["S1", "S3"]
