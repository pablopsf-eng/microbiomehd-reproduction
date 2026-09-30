"""Regression tests for the published MicrobiomeHD supplementary files.

These pin down the labelling rule recovered from the data: given
``file-S1.qvalues.txt``, the rule reproduces *both* published versions of
file-S3 exactly, differing only in which datasets are excluded.

The rule itself now lives in ``mbhd.shared_response`` (moved there in M3). These
tests are therefore now a check on our own implementation, not only on the
reference data.

The data files are not committed (Zenodo is
CC BY-NC 4.0, the GitHub supplementary files carry no licence at all). Every
test skips when they are absent, so a clean checkout stays green.
"""

from __future__ import annotations

import pathlib

import pytest

from mbhd.shared_response import (
    PUBLISHED_EXCLUSIONS,
    QValueRows,
    disease_of,
    is_significant,
    shared_response_labels,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"

S1 = DATA / "file-S1.qvalues.txt"
S3_ZENODO = DATA / "file-S3.core_genera.txt"
S3_GITHUB = DATA / "file-S3.nonspecific_genera.txt"

ALPHA = 0.05


def _require(*paths: pathlib.Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def read_qvalues(path: pathlib.Path) -> tuple[list[str], QValueRows]:
    """Parse file-S1. Returns (dataset_ids, {lineage: {dataset_id: q}}).

    Values are *signed*: negative means the mean abundance was higher in
    controls, positive means higher in cases. A blank cell means the genus was
    absent from that dataset and is returned as None -- distinct from 0.0,
    which is present but degenerate.
    """
    lines = path.read_text().splitlines()
    datasets = lines[0].split("\t")[1:]
    rows: QValueRows = {}
    for line in lines[1:]:
        parts = line.split("\t")
        rows[parts[0]] = {
            d: (float(v) if v.strip() else None)
            for d, v in zip(datasets, parts[1:], strict=True)
        }
    return datasets, rows


def read_labels(path: pathlib.Path) -> dict[str, str]:
    """Parse a file-S3. Returns {lineage: label}, where "" means unlabelled."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        out[fields[0]] = fields[-1].strip()
    return out


# --------------------------------------------------------------------------
# The rule reproduces both published files exactly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, exclusions, expected_counts",
    [
        pytest.param(
            S3_ZENODO,
            frozenset(),
            {"": 79, "health": 34, "disease": 26, "mixed": 7},
            id="zenodo-updated-list",
        ),
        pytest.param(
            S3_GITHUB,
            PUBLISHED_EXCLUSIONS,
            {"": 93, "health": 24, "disease": 20, "mixed": 7},
            id="github-published-figures",
        ),
    ],
)
def test_rule_reproduces_published_file(path, exclusions, expected_counts):
    _require(S1, path)
    _, rows = read_qvalues(S1)
    computed = shared_response_labels(rows, exclusions)
    expected = read_labels(path)

    assert set(computed) == set(expected), "row sets differ"
    assert computed == expected, "label disagreement"

    counts: dict[str, int] = {}
    for label in computed.values():
        counts[label] = counts.get(label, 0) + 1
    assert counts == expected_counts


def test_published_counts_match_the_paper():
    """The GitHub file carries the paper's Fig. 3a figures: 24 / 20 / 7."""
    _require(S3_GITHUB)
    labels = read_labels(S3_GITHUB)
    assert sum(v == "health" for v in labels.values()) == 24
    assert sum(v == "disease" for v in labels.values()) == 20
    assert sum(v == "mixed" for v in labels.values()) == 7


def test_versions_differ_only_additively():
    """Zenodo is a strict superset of GitHub: nothing dropped, nothing flipped."""
    _require(S3_ZENODO, S3_GITHUB)
    zenodo, github = read_labels(S3_ZENODO), read_labels(S3_GITHUB)

    assert set(github) < set(zenodo)
    assert len(set(zenodo) - set(github)) == 2  # Papillibacter, Allisonella

    relabelled = {g: (github[g], zenodo[g]) for g in github if github[g] != zenodo[g]}
    assert len(relabelled) == 16
    # Every change is blank -> labelled. No direction is ever reversed.
    assert all(old == "" and new != "" for old, new in relabelled.values())


# --------------------------------------------------------------------------
# Structural invariants of the input files.
# --------------------------------------------------------------------------


def test_signed_pool_matches_the_paper():
    """152 genera significant in >= 1 dataset -- and the two traps around it."""
    _require(S1)
    _, rows = read_qvalues(S1)

    # The paper's 152 counts exact zeros as significant.
    paper_pool = {
        g
        for g, qs in rows.items()
        if any(q is not None and abs(q) < ALPHA for q in qs.values())
    }
    assert len(paper_pool) == 152

    # Excluding them drops six genera whose only significant cell was a zero,
    # leaving exactly the 146 rows of the Zenodo file-S3.
    working_pool = {
        g for g, qs in rows.items() if any(is_significant(q) for q in qs.values())
    }
    assert len(working_pool) == 146
    assert len(paper_pool - working_pool) == 6

    # Unsigned comparison treats every negative value as significant.
    naive = {
        g
        for g, qs in rows.items()
        if any(q is not None and q < ALPHA for q in qs.values())
    }
    assert len(naive) == 240


def test_exact_zeros_are_confined_to_the_least_powered_datasets():
    _require(S1)
    _, rows = read_qvalues(S1)
    zeros = [
        (lineage, dataset_id)
        for lineage, qs in rows.items()
        for dataset_id, q in qs.items()
        if q == 0.0
    ]
    assert len(zeros) == 11
    assert {d for _, d in zeros} == {"cdi_youngster", "ob_zhu", "nash_zhu"}


def test_youngster_is_included_in_the_shared_response():
    """Excluding it is wrong: the paper drops it only from the classifiers."""
    _require(S1, S3_ZENODO)
    _, rows = read_qvalues(S1)
    expected = read_labels(S3_ZENODO)

    without = shared_response_labels(rows, frozenset({"cdi_youngster"}))
    disagreements = {
        g.split(";")[-1]
        for g in set(without) & set(expected)
        if without[g] != expected[g]
    }
    assert disagreements == {
        "g__Anaerosporobacter",
        "g__Hespellia",
        "g__Subdoligranulum",
    }


def test_s1_shape():
    _require(S1)
    datasets, rows = read_qvalues(S1)
    assert len(datasets) == 30
    assert len(rows) == 298


def test_s3_lineages_are_unique_and_genus_names_do_not_collide():
    """Keying on the trailing g__ name is safe."""
    _require(S3_ZENODO)
    lineages = list(read_labels(S3_ZENODO))
    assert len(lineages) == len(set(lineages)) == 146
    genera = [t.split(";")[-1] for t in lineages]
    assert len(genera) == len(set(genera))


def test_disease_of_yields_eleven_groups():
    """disease_of collapses cdi/noncdi/edd into one diarrhea group."""
    assert disease_of("cdi_schubert") == "diarrhea"
    assert disease_of("noncdi_schubert") == "diarrhea"
    assert disease_of("edd_singh") == "diarrhea"
    assert disease_of("crc_baxter") == "crc"
