"""The "shared response" (non-specific genera) result, from any q-value source.

Moved from ``tests/test_supplementary_files.py`` now that ``mbhd`` has
a home for logic describing the *paper's own* labelling rule, recovered by
comparison against the published files: a genus enters the "shared response"
pool if it is significant (``abs(signed_q) < ALPHA``, excluding exact ``0.0``) in
at least one non-excluded dataset, and is labelled by the *direction* of
that significance across at least two distinct diseases.

``disease_of``, ``is_significant``, and ``shared_response_labels`` are
unchanged from the test file -- this move does not touch their behaviour,
only their location, and ``tests/test_supplementary_files.py`` still pins
the exact same published-file-reproduction regression tests, now importing
production code instead of a test-local copy.

``build_qvalue_matrix`` is new (M3): it assembles *this project's own*
reproduced signed q-values (``mbhd.differential.differential_abundance``,
called once per dataset) into the same ``{lineage: {dataset_id: q_or_None}}``
shape ``shared_response_labels`` already expects for the published
``file-S1.qvalues.txt`` -- so the identical, already-verified labelling rule
applies unchanged to reproduced data, closing the loop from raw OTU tables
to the paper's headline finding independently for the first time in this
project.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mbhd.datasets import DatasetInfo
from mbhd.differential import EXPECTED_DATASET_ERRORS, differential_abundance

ALPHA = 0.05

#: The three diarrhoeal prefixes are one disease.
DIARRHEA_PREFIXES = frozenset({"cdi", "noncdi", "edd"})

#: Published figures excluded these three: two used non-healthy controls
#: (Papa, Gevers) and one showed a signal reflecting behaviour rather than
#: disease state (Lozupone). This is the GitHub file-S3 variant's exclusion
#: set -- the paper's published result and the M3 reproduction target.
#: The updated Zenodo file-S3 excludes nothing.
PUBLISHED_EXCLUSIONS = frozenset({"ibd_papa", "ibd_gevers", "hiv_lozupone"})

#: {lineage: {dataset_id: signed_q_or_None}}. A missing/``None`` value means
#: the genus was absent from that dataset -- distinct from an explicit
#: ``0.0``, which is present but degenerate.
QValueRows = dict[str, dict[str, float | None]]


def disease_of(dataset_id: str) -> str:
    """Map a dataset ID to its disease group.

    Yields 11 groups across the 30 analysed datasets. The paper describes ten
    diseases; the discrepancy is unresolved (unverified guess: two liver
    conditions counted as one) but does not affect reproduction -- this mapping
    reproduces both published file-S3 versions exactly.
    """
    prefix = dataset_id.split("_")[0]
    return "diarrhea" if prefix in DIARRHEA_PREFIXES else prefix


def is_significant(q: float | None) -> bool:
    """Significance test, with both traps handled.

    ``q < ALPHA`` would count every negative value as significant and inflate
    the pool from 152 to 240 on the published data. Exact ``0.0`` is excluded:
    on the published data it appears 11 times, only in the three
    least-powered comparisons, and treating it as significant produces two
    wrong labels. No reproduced q-value has ever been observed to be exactly
    ``0.0`` (docs/reproduction-m3.md) -- this clause currently has
    nothing to apply to on our own data, but is kept so the rule stays
    identical to the one verified against the published files.
    """
    return q is not None and q != 0.0 and abs(q) < ALPHA


def shared_response_labels(
    rows: QValueRows,
    exclusions: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Apply the recovered rule. Returns {lineage: label} for pool members only.

    A genus enters the pool if it is significant in at least one non-excluded
    dataset. It is labelled if significant in the same direction across at
    least two distinct diseases; "mixed" if that holds in both directions.
    Works identically whether ``rows`` came from the published
    ``file-S1.qvalues.txt`` or from ``build_qvalue_matrix``'s reproduced
    q-values -- the rule itself does not know or care which.
    """
    labels: dict[str, str] = {}
    for lineage, qs in rows.items():
        health: set[str] = set()
        disease: set[str] = set()
        in_pool = False
        for dataset_id, q in qs.items():
            if dataset_id in exclusions or not is_significant(q):
                continue
            in_pool = True
            (health if q < 0 else disease).add(disease_of(dataset_id))
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


def build_qvalue_matrix(
    dataset_ids: list[str],
    index: dict[str, DatasetInfo],
    data_root: Path,
    *,
    compute_fn: Callable[..., pd.DataFrame] | None = None,
) -> tuple[QValueRows, dict[str, str]]:
    """Assemble this project's own reproduced signed q-values into file-S1's shape.

    Calls ``compute_fn`` (default: ``mbhd.differential.differential_abundance``,
    resolved at call time so a test that monkeypatches this module's
    ``differential_abundance`` name keeps working unchanged) once per dataset
    in ``dataset_ids`` -- e.g. ``functools.partial(differential_abundance,
    abundance_transform=...)`` for the M4 CLR path, which reuses this exact outer-join/
    failure-handling logic instead of a second, duplicated assembly function.
    Default ``None`` preserves exactly today's Euclidean behaviour --
    regression-tested against M3's committed numbers.

    A dataset that raises one of
    ``mbhd.differential.EXPECTED_DATASET_ERRORS`` (a documented, non-crashing
    per-dataset failure, e.g. ``ob_zupancic``'s ``ZeroReadSampleError``) is
    recorded in the returned ``failures`` dict rather than silently dropped
    or allowed to abort the whole run; any other exception propagates
    (never fall back silently).

    Returns ``(rows, failures)``:

    - ``rows``: ``{genus: {dataset_id: signed_q_or_None}}`` -- an outer join
      over every genus seen in *any* successfully-reproduced dataset, and
      every dataset_id originally requested (including failed ones, whose
      entries are all ``None``). A genus absent from a given dataset's own
      reproduction, or belonging to a dataset that failed outright, is
      ``None`` -- matching ``file-S1.qvalues.txt``'s own blank-cell
      convention, never silently omitted from that dataset's inner dict.
    - ``failures``: ``{dataset_id: str(exception)}`` for every dataset that
      did not produce a comparison.

    The result is intended to be passed directly to ``shared_response_
    labels``, exactly as ``tests/test_supplementary_files.py``'s parsed
    ``file-S1.qvalues.txt`` rows already are.
    """
    fn = compute_fn if compute_fn is not None else differential_abundance

    per_dataset_q = {}
    failures: dict[str, str] = {}
    for dataset_id in dataset_ids:
        try:
            reproduced = fn(dataset_id, index, data_root)
        except EXPECTED_DATASET_ERRORS as exc:
            failures[dataset_id] = str(exc)
            continue
        per_dataset_q[dataset_id] = reproduced["signed_q_value"]

    all_genera: set[str] = set()
    for series in per_dataset_q.values():
        all_genera.update(series.index)

    rows: QValueRows = {
        genus: {
            dataset_id: (
                float(per_dataset_q[dataset_id][genus])
                if dataset_id in per_dataset_q
                and genus in per_dataset_q[dataset_id].index
                else None
            )
            for dataset_id in dataset_ids
        }
        for genus in sorted(all_genera)
    }
    return rows, failures
