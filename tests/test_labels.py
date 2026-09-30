"""Tests for mbhd.labels.reconcile_counts, with synthetic fixtures.

The real, hand-verified end-to-end check (all 30 analysed datasets against
Table 1) lives in tests/test_table1_counts.py. This file covers
reconcile_counts's own contract in isolation -- the intersection rule,
label-alias resolution, and exclusion reporting -- with hand-computed
expected values, independent of any real downloaded data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbhd.labels import (
    CASE_CONTROL,
    CONDITION_FILTER,
    KNOWN_MISMATCHES,
    SAMPLE_TYPE_FILTER,
    SUBJECT_ID_COLUMN,
    LabelCounts,
    reconcile_counts,
    restrict_to_condition,
    restrict_to_sample_type,
)


def _metadata(rows: dict[str, str]) -> pd.DataFrame:
    """{sample_id: DiseaseState} -> a one-column metadata DataFrame."""
    return pd.DataFrame({"DiseaseState": rows})


def test_reconcile_counts_hand_computed() -> None:
    """4 samples: 2 control (H), 2 case (CDI), all present in both tables."""
    metadata = _metadata({"S1": "H", "S2": "H", "S3": "CDI", "S4": "CDI"})
    counts = reconcile_counts(
        "cdi_schubert", metadata, otu_sample_ids={"S1", "S2", "S3", "S4"}
    )
    assert counts == LabelCounts(controls=2, cases=2, excluded={})


def test_reconcile_counts_only_counts_the_intersection() -> None:
    """S3 is in the metadata but not the OTU table -- must not be counted."""
    metadata = _metadata({"S1": "H", "S2": "H", "S3": "CDI"})
    counts = reconcile_counts("cdi_schubert", metadata, otu_sample_ids={"S1", "S2"})
    assert counts.controls == 2
    assert counts.cases == 0


def test_reconcile_counts_reports_unmapped_values_as_excluded() -> None:
    """A DiseaseState value that is neither control nor case must be
    reported, never silently added to either side."""
    metadata = _metadata({"S1": "H", "S2": "CDI", "S3": "SOMETHING_ELSE"})
    counts = reconcile_counts(
        "cdi_schubert", metadata, otu_sample_ids={"S1", "S2", "S3"}
    )
    assert counts.controls == 1
    assert counts.cases == 1
    assert counts.excluded == {"SOMETHING_ELSE": 1}


def test_reconcile_counts_handles_compound_case_labels() -> None:
    """ibd_morgan's cases are the union of UC and CD (Table 1's "UC, CD").
    ibd_morgan is also in SAMPLE_TYPE_FILTER, so the synthetic fixture
    needs a uniform "stool" SampleType column -- a no-op filter here, but
    without it restrict_to_sample_type would raise (the column is
    genuinely required, not just missing by test-fixture oversight)."""
    metadata = _metadata({"S1": "H", "S2": "UC", "S3": "CD", "S4": "CD"})
    metadata["SampleType"] = "stool"
    counts = reconcile_counts(
        "ibd_morgan", metadata, otu_sample_ids={"S1", "S2", "S3", "S4"}
    )
    assert counts.controls == 1
    assert counts.cases == 3


def test_reconcile_counts_applies_label_aliases() -> None:
    """ob_zhu's Table 1 label "OB" resolves to the actual value "nonNASH-OB"."""
    metadata = _metadata({"S1": "H", "S2": "nonNASH-OB", "S3": "NASH"})
    counts = reconcile_counts("ob_zhu", metadata, otu_sample_ids={"S1", "S2", "S3"})
    assert counts.controls == 1
    assert counts.cases == 1
    # NASH is neither ob_zhu's control nor its case label -- excluded.
    assert counts.excluded == {"NASH": 1}


def test_reconcile_counts_raises_on_unknown_dataset() -> None:
    metadata = _metadata({"S1": "H"})
    with pytest.raises(KeyError):
        reconcile_counts("not_a_real_dataset", metadata, otu_sample_ids={"S1"})


def test_reconcile_counts_raises_on_missing_disease_state_column() -> None:
    metadata = pd.DataFrame({"other_column": ["H", "CDI"]}, index=["S1", "S2"])
    with pytest.raises(ValueError, match="cdi_schubert.*DiseaseState"):
        reconcile_counts("cdi_schubert", metadata, otu_sample_ids={"S1", "S2"})


def test_reconcile_counts_empty_otu_ids_gives_zero_counts_not_a_crash() -> None:
    metadata = _metadata({"S1": "H", "S2": "CDI"})
    counts = reconcile_counts("cdi_schubert", metadata, otu_sample_ids=set())
    assert counts == LabelCounts(controls=0, cases=0, excluded={})


def test_case_control_and_known_mismatches_are_disjoint_key_sets_within_30() -> None:
    """CASE_CONTROL covers all 30 analysed datasets; KNOWN_MISMATCHES is a
    5-entry subset of it, never a dataset outside it. (9 -> 5: edd_singh,
    hiv_lozupone, nash_wong, and ob_goodrich removed once CONDITION_FILTER
    made them reconcile exactly, the same way ibd_morgan/ibd_willing were
    removed via SAMPLE_TYPE_FILTER.)"""
    assert len(CASE_CONTROL) == 30
    assert len(KNOWN_MISMATCHES) == 5
    assert set(KNOWN_MISMATCHES).issubset(set(CASE_CONTROL))


def test_reconcile_counts_collapses_to_distinct_subjects_when_overridden() -> None:
    """cdi_youngster: Table 1 counts
    distinct patients, not samples, for this one longitudinal FMT study --
    two samples from the same subject must count once, not twice."""
    assert SUBJECT_ID_COLUMN == {"cdi_youngster": "subject"}
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "H", "CDI", "CDI"],
            "subject": ["P1", "P1", "P2", "P3", "P3"],
        },
        index=["S1", "S2", "S3", "S4", "S5"],
    )
    counts = reconcile_counts(
        "cdi_youngster", metadata, otu_sample_ids=set(metadata.index)
    )
    assert counts.controls == 2  # P1, P2 -- not 3 samples
    assert counts.cases == 1  # P3 -- not 2 samples


def test_reconcile_counts_subject_override_does_not_apply_to_other_datasets() -> None:
    """A dataset absent from SUBJECT_ID_COLUMN still counts samples, even if
    it happens to carry a column also named "subject"."""
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "CDI"],
            "subject": ["P1", "P1", "P2"],
        },
        index=["S1", "S2", "S3"],
    )
    counts = reconcile_counts(
        "cdi_schubert", metadata, otu_sample_ids=set(metadata.index)
    )
    assert counts.controls == 2  # sample count, not distinct-subject count
    assert counts.cases == 1


def test_restrict_to_sample_type_is_a_no_op_without_an_entry() -> None:
    """The vast majority of datasets have no SAMPLE_TYPE_FILTER entry --
    metadata must come back completely unchanged, not just "similar"."""
    metadata = _metadata({"S1": "H", "S2": "CDI"})
    result = restrict_to_sample_type("cdi_schubert", metadata)
    pd.testing.assert_frame_equal(result, metadata)


def test_restrict_to_sample_type_filters_to_the_kept_value() -> None:
    assert SAMPLE_TYPE_FILTER["ibd_morgan"] == ("SampleType", "stool")
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "CD"],
            "SampleType": ["stool", "biopsy", "stool"],
        },
        index=["S1", "S2", "S3"],
    )
    result = restrict_to_sample_type("ibd_morgan", metadata)
    assert list(result.index) == ["S1", "S3"]


def test_restrict_to_sample_type_raises_when_the_column_is_missing() -> None:
    """A silent no-op here would defeat the point of the filter -- a
    dataset in SAMPLE_TYPE_FILTER whose metadata lacks the named column
    must fail loudly, not quietly return everything."""
    metadata = _metadata({"S1": "H", "S2": "CD"})
    with pytest.raises(ValueError, match="ibd_morgan.*SampleType"):
        restrict_to_sample_type("ibd_morgan", metadata)


def test_reconcile_counts_applies_sample_type_filter_for_ibd_morgan() -> None:
    """End-to-end through reconcile_counts, not just the helper in
    isolation: the one biopsy sample must not be counted at all."""
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "CD"],
            "SampleType": ["stool", "biopsy", "stool"],
        },
        index=["S1", "S2", "S3"],
    )
    counts = reconcile_counts("ibd_morgan", metadata, otu_sample_ids={"S1", "S2", "S3"})
    assert counts.controls == 1  # S2 (biopsy) excluded
    assert counts.cases == 1


def test_restrict_to_condition_is_a_no_op_without_an_entry() -> None:
    """The vast majority of datasets have no CONDITION_FILTER entry --
    metadata must come back completely unchanged, not just "similar"."""
    metadata = _metadata({"S1": "H", "S2": "CDI"})
    result = restrict_to_condition("cdi_schubert", metadata)
    pd.testing.assert_frame_equal(result, metadata)


def test_restrict_to_condition_filters_to_the_kept_values() -> None:
    assert CONDITION_FILTER["edd_singh"] == ("Time Point", (1,))
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "EDD"],
            "Time Point": [1, 2, 1],
        },
        index=["S1", "S2", "S3"],
    )
    result = restrict_to_condition("edd_singh", metadata)
    assert list(result.index) == ["S1", "S3"]


def test_restrict_to_condition_supports_multiple_kept_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No adopted production entry needs more than one value, but the
    isin-based multi-value path must still work -- exercised here through
    the real function with a monkeypatched synthetic two-value entry,
    not by computing the expected result with raw pandas and comparing it
    to itself (a regression here, e.g. `==` instead of `.isin()`, must
    actually be caught)."""
    monkeypatch.setitem(CONDITION_FILTER, "cdi_schubert", ("cohort", ("BCN0", "STK")))
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "HIV"],
            "cohort": ["BCN0", "BCN1", "STK"],
        },
        index=["S1", "S2", "S3"],
    )
    result = restrict_to_condition("cdi_schubert", metadata)
    assert list(result.index) == ["S1", "S3"]


def test_restrict_to_condition_raises_when_the_column_is_missing() -> None:
    """A silent no-op here would defeat the point of the filter -- a
    dataset in CONDITION_FILTER whose metadata lacks the named column must
    fail loudly, not quietly return everything."""
    metadata = _metadata({"S1": "H", "S2": "EDD"})
    with pytest.raises(ValueError, match="edd_singh.*Time Point"):
        restrict_to_condition("edd_singh", metadata)


def test_reconcile_counts_applies_condition_filter_for_edd_singh() -> None:
    """End-to-end through reconcile_counts, not just the helper in
    isolation: the Time Point 2 (follow-up) sample must not be counted."""
    metadata = pd.DataFrame(
        {
            "DiseaseState": ["H", "H", "EDD", "EDD"],
            "Time Point": [1, 1, 1, 2],
        },
        index=["S1", "S2", "S3", "S4"],
    )
    counts = reconcile_counts(
        "edd_singh", metadata, otu_sample_ids={"S1", "S2", "S3", "S4"}
    )
    assert counts.controls == 2
    assert counts.cases == 1  # S4 (Time Point 2) excluded
