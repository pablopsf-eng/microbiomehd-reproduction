"""Independent verification of mbhd.shared_response.build_qvalue_matrix's
outer-join contract, written fresh (not copied from tests/test_shared_
response.py) as part of an external review of M3.

Focus: a genus absent from one dataset's own reproduction must become
``None`` in that dataset's slot -- never silently ``0.0`` (which would be a
real, present, degenerate value) and never
simply missing from the inner dict (which would break downstream code that
iterates ``qs.items()`` expecting every requested dataset_id to be a key,
e.g. mbhd.shared_response.shared_response_labels's own loop).
"""

from __future__ import annotations

import pathlib

import pandas as pd

from mbhd.shared_response import build_qvalue_matrix


def _fake_result(genus_to_q: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"signed_q_value": list(genus_to_q.values())},
        index=pd.Index(list(genus_to_q), name="genus"),
    )


def test_partial_overlap_missing_genus_becomes_none_not_zero_not_omitted(monkeypatch):
    """Dataset A has genera X and Y; dataset B has only X. Y must appear in
    B's row as an explicit None key, not 0.0 and not absent from the dict."""
    results = {
        "a": _fake_result({"g__X": 0.01, "g__Y": -0.03}),
        "b": _fake_result({"g__X": 0.02}),
    }

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, failures = build_qvalue_matrix(["a", "b"], index={}, data_root=pathlib.Path())

    assert failures == {}
    # Y is present in the returned dict for "b" ...
    assert "b" in rows["g__Y"]
    # ... and its value is exactly None, distinguishable from a real 0.0.
    assert rows["g__Y"]["b"] is None
    assert rows["g__X"] == {"a": 0.01, "b": 0.02}


def test_a_genuine_zero_qvalue_survives_as_zero_not_none(monkeypatch):
    """A dataset that genuinely reports signed_q_value == 0.0 for a genus
    must keep that 0.0 through the outer join -- a naive `value or None`-
    style implementation would wrongly collapse 0.0 to None since 0.0 is
    falsy in Python."""
    results = {
        "a": _fake_result({"g__Zero": 0.0}),
        "b": _fake_result({"g__Zero": -0.01}),
    }

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, _failures = build_qvalue_matrix(
        ["a", "b"], index={}, data_root=pathlib.Path()
    )

    assert rows["g__Zero"]["a"] == 0.0
    assert rows["g__Zero"]["a"] is not None


def test_dataset_with_no_genera_at_all_still_gets_none_for_every_genus(monkeypatch):
    """An edge case not covered by the existing disjoint-set test: a dataset
    that succeeds but produces an empty genus index (e.g. everything filtered
    out) must still receive a None entry for every genus another dataset
    contributes, not be silently absent from the requested dataset_ids."""
    results = {
        "a": _fake_result({"g__X": 0.01}),
        "empty": _fake_result({}),
    }

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, failures = build_qvalue_matrix(
        ["a", "empty"], index={}, data_root=pathlib.Path()
    )

    assert failures == {}
    assert rows == {"g__X": {"a": 0.01, "empty": None}}


def test_requested_dataset_order_is_preserved_in_every_row(monkeypatch):
    """Every genus's inner dict must have a key for every requested
    dataset_id, in the outer-join sense (order/coverage), regardless of
    which dataset happened to run first."""
    results = {
        "a": _fake_result({"g__X": 0.01}),
        "b": _fake_result({"g__X": 0.02}),
        "c": _fake_result({"g__Y": 0.03}),
    }

    def fake_differential_abundance(dataset_id, index, data_root, **kwargs):
        return results[dataset_id]

    monkeypatch.setattr(
        "mbhd.shared_response.differential_abundance", fake_differential_abundance
    )
    rows, _failures = build_qvalue_matrix(
        ["a", "b", "c"], index={}, data_root=pathlib.Path()
    )

    for genus_row in rows.values():
        assert set(genus_row.keys()) == {"a", "b", "c"}
