"""Tests for mbhd.shared_response.build_qvalue_matrix.

Two layers, matching the project's established pattern: fully synthetic
tests of build_qvalue_matrix's own contract (genus alignment, failure
recording, no double-counting of a shared folder's two dataset IDs) with no
data dependency, and a real-data check (skip-guarded) that no reproduced
q-value is ever exactly 0.0 -- a flagged edge case.

shared_response_labels/disease_of/is_significant's own regression tests
against the *published* files live in tests/test_supplementary_files.py,
unchanged by this move.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.differential import NoSurvivingSamplesError
from mbhd.shared_response import build_qvalue_matrix, disease_of, shared_response_labels

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW = _REPO_ROOT / "data" / "raw"
DATASET_INFO = DATA_RAW / "dataset_info.yaml"


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _fake_result(genus_to_q: dict[str, float]) -> pd.DataFrame:
    """A minimal stand-in for differential_abundance's return shape --
    only signed_q_value is read by build_qvalue_matrix."""
    return pd.DataFrame(
        {"signed_q_value": list(genus_to_q.values())},
        index=pd.Index(list(genus_to_q), name="genus"),
    )


def test_build_qvalue_matrix_outer_joins_disjoint_genus_sets(monkeypatch):
    """Dataset A has genus X only, dataset B has genus Y only -- both genera
    must appear in the matrix, each None for the dataset that lacks it."""
    results = {
        "a": _fake_result({"g__X": 0.01}),
        "b": _fake_result({"g__Y": -0.02}),
    }

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, failures = build_qvalue_matrix(["a", "b"], index={}, data_root=pathlib.Path())

    assert failures == {}
    assert rows == {
        "g__X": {"a": 0.01, "b": None},
        "g__Y": {"a": None, "b": -0.02},
    }


def test_build_qvalue_matrix_records_a_hard_failure_without_dropping_it(monkeypatch):
    """A dataset that raises an EXPECTED_DATASET_ERRORS member is recorded in
    failures, its genera contribute nothing, and it still gets a None-filled
    row entry for every genus another dataset does contribute."""
    results = {"a": _fake_result({"g__X": 0.01})}

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        if dataset_id == "broken":
            raise NoSurvivingSamplesError("broken: 0 controls, 0 cases")
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, failures = build_qvalue_matrix(
        ["a", "broken"], index={}, data_root=pathlib.Path()
    )

    assert failures == {"broken": "broken: 0 controls, 0 cases"}
    assert rows == {"g__X": {"a": 0.01, "broken": None}}


def test_build_qvalue_matrix_an_unexpected_exception_propagates(monkeypatch):
    """A bug (not an EXPECTED_DATASET_ERRORS member) must not be swallowed."""

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        raise RuntimeError("not a documented per-dataset failure mode")

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    with pytest.raises(RuntimeError, match="not a documented"):
        build_qvalue_matrix(["a"], index={}, data_root=pathlib.Path())


# --------------------------------------------------------------------------
# compute_fn's default must not move build_qvalue_matrix's existing behaviour.
# --------------------------------------------------------------------------


def test_build_qvalue_matrix_compute_fn_default_matches_omitting_it(monkeypatch):
    """Calling with compute_fn explicitly None must be identical to omitting
    it -- both resolve, at call time, to this module's own
    differential_abundance name (so a caller's monkeypatch of that name still
    works either way, which this test also exercises)."""
    results = {"a": _fake_result({"g__X": 0.01})}

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )

    without_kwarg = build_qvalue_matrix(["a"], index={}, data_root=pathlib.Path())
    with_none = build_qvalue_matrix(
        ["a"], index={}, data_root=pathlib.Path(), compute_fn=None
    )
    assert without_kwarg == with_none == ({"g__X": {"a": 0.01}}, {})


def test_build_qvalue_matrix_respects_an_explicit_compute_fn():
    """An explicit compute_fn (e.g. the M4 CLR path's
    functools.partial(differential_abundance, abundance_transform=...)) is
    used instead of the default, and never falls back to the real
    differential_abundance if the explicit one raises."""
    calls: list[str] = []

    def clr_like_compute_fn(dataset_id, index, data_root):
        calls.append(dataset_id)
        return _fake_result({"g__CLR": -0.02})

    rows, failures = build_qvalue_matrix(
        ["a"], index={}, data_root=pathlib.Path(), compute_fn=clr_like_compute_fn
    )
    assert calls == ["a"]
    assert failures == {}
    assert rows == {"g__CLR": {"a": -0.02}}


def test_shared_diarrhea_folder_datasets_do_not_double_count_as_two_diseases():
    """cdi_schubert and noncdi_schubert share one folder and correlated
    controls (same Schubert study) -- both map to
    "diarrhea" via disease_of, so they must count as *one* disease's worth
    of evidence, not two, toward the >=2-diseases labelling threshold."""
    assert disease_of("cdi_schubert") == disease_of("noncdi_schubert") == "diarrhea"

    rows = {
        "g__Shared": {
            "cdi_schubert": 0.01,  # significant, case-associated (disease)
            "noncdi_schubert": 0.02,  # same direction, same disease group
            "crc_baxter": 0.9,  # not significant
        }
    }
    labels = shared_response_labels(rows)
    # Only one disease's worth of evidence (diarrhea) -- must not be labelled
    # "disease" (which needs >= 2 distinct diseases).
    assert labels["g__Shared"] == ""


def test_shared_response_labels_needs_two_genuinely_distinct_diseases():
    """A second, truly distinct disease does cross the threshold."""
    rows = {
        "g__Shared": {
            "cdi_schubert": 0.01,
            "crc_baxter": 0.02,
        }
    }
    labels = shared_response_labels(rows)
    assert labels["g__Shared"] == "disease"


def test_real_reproduced_qvalues_are_never_exactly_zero():
    """Flagged edge case: the published data's
    "exact 0.0 is not significant" clause was derived from *published*
    q-values, all sitting in the three least-powered comparisons. Checked
    directly against every dataset this project can reproduce: as of
    2026-09-17, no reproduced q-value is ever exactly 0.0 (see
    docs/reproduction-m3.md) -- documented here as a real, re-checked
    finding, not assumed to still hold as mbhd.stats evolves."""
    _require(DATASET_INFO)
    index = load_dataset_info(DATASET_INFO)
    rows, _failures = build_qvalue_matrix(analysed_dataset_ids(index), index, DATA_RAW)
    if not rows:
        pytest.skip("no dataset reproduced (data not fully fetched)")
    exact_zeros = [
        (genus, dataset_id)
        for genus, qs in rows.items()
        for dataset_id, q in qs.items()
        if q == 0.0
    ]
    assert exact_zeros == []
