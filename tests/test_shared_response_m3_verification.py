"""Independent re-derivation of the M3 headline: 138/142 (97.2%) lineage
agreement between this project's own reproduced shared-response pool
(full-29 datasets) and the published GitHub ``file-S3.nonspecific_genera.txt``.

Ground truth: the real, downloaded ``data/raw/file-S3.nonspecific_genera.txt``.

Independence from ``mbhd.shared_response``: this file reimplements the
labelling rule and the file-S3 parser from
scratch, rather than calling ``mbhd.shared_response.shared_response_labels``/
``build_qvalue_matrix``. It does reuse ``mbhd.differential.
differential_abundance`` as the q-value source -- that function is
independently verified against ``file-S1.qvalues.txt`` in
``tests/test_differential_m2_verification.py``, so this test isolates the
labelling-rule/diff logic, not the underlying statistics, as the thing being
checked here.

Slow (recomputes differential abundance for all 30 analysed datasets,
~20s): not parametrized, one big end-to-end test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mbhd.datasets import analysed_dataset_ids, load_dataset_info
from mbhd.differential import EXPECTED_DATASET_ERRORS, differential_abundance

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
ALPHA = 0.05
DIARRHEA_PREFIXES = {"cdi", "noncdi", "edd"}
PUBLISHED_EXCLUSIONS = {"ibd_papa", "ibd_gevers", "hiv_lozupone"}


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _disease_of(dataset_id: str) -> str:
    prefix = dataset_id.split("_")[0]
    return "diarrhea" if prefix in DIARRHEA_PREFIXES else prefix


def _is_significant(q: float | None) -> bool:
    return q is not None and q != 0.0 and abs(q) < ALPHA


def _label_pool(
    qmatrix: dict[str, dict[str, float | None]], exclusions: set[str]
) -> dict[str, str]:
    """From-scratch reimplementation of the recovered labelling rule
    -- independent of
    mbhd.shared_response.shared_response_labels."""
    labels: dict[str, str] = {}
    for lineage, per_dataset in qmatrix.items():
        health: set[str] = set()
        disease: set[str] = set()
        in_pool = False
        for dataset_id, q in per_dataset.items():
            if dataset_id in exclusions or not _is_significant(q):
                continue
            in_pool = True
            (health if q < 0 else disease).add(_disease_of(dataset_id))
        if not in_pool:
            continue
        if len(health) >= 2 and len(disease) >= 2:
            labels[lineage] = "mixed"
        elif len(health) >= 2:
            labels[lineage] = "health"
        elif len(disease) >= 2:
            labels[lineage] = "disease"
        else:
            labels[lineage] = ""
    return labels


def _parse_file_s3(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        out[fields[0]] = fields[-1].strip()
    return out


def test_full29_vs_github_agreement_is_138_of_142():
    _require(DATA_ROOT / "file-S3.nonspecific_genera.txt")
    index = load_dataset_info(DATA_ROOT / "dataset_info.yaml")
    dataset_ids = analysed_dataset_ids(index)
    assert len(dataset_ids) == 30

    per_dataset_q = {}
    failures = {}
    for dataset_id in dataset_ids:
        try:
            per_dataset_q[dataset_id] = differential_abundance(
                dataset_id, index, DATA_ROOT
            )["signed_q_value"]
        except EXPECTED_DATASET_ERRORS as exc:
            failures[dataset_id] = str(exc)

    # ob_zupancic is the one documented hard failure (M2/M3): zero-read
    # samples after study-level filtering.
    assert set(failures) == {"ob_zupancic"}
    assert len(per_dataset_q) == 29

    all_genera: set[str] = set()
    for series in per_dataset_q.values():
        all_genera.update(series.index)

    qmatrix = {
        genus: {
            dataset_id: (
                float(per_dataset_q[dataset_id][genus])
                if dataset_id in per_dataset_q
                and genus in per_dataset_q[dataset_id].index
                else None
            )
            for dataset_id in dataset_ids
        }
        for genus in all_genera
    }

    reproduced_labels = _label_pool(qmatrix, PUBLISHED_EXCLUSIONS)
    published_labels = _parse_file_s3(DATA_ROOT / "file-S3.nonspecific_genera.txt")

    assert len(reproduced_labels) == 152
    assert len(published_labels) == 144

    common = set(reproduced_labels) & set(published_labels)
    only_reproduced = set(reproduced_labels) - set(published_labels)
    only_published = set(published_labels) - set(reproduced_labels)
    agree = sum(1 for g in common if reproduced_labels[g] == published_labels[g])

    assert len(common) == 142
    assert agree == 138
    assert len(only_reproduced) == 10
    assert len(only_published) == 2
    assert agree / len(common) == pytest.approx(138 / 142)
