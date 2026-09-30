"""Per-genus case/control differential abundance for one MicrobiomeHD dataset.

Orchestrates ``mbhd.datasets`` (path resolution), ``mbhd.io`` (loading),
``mbhd.labels`` (case/control label resolution), ``mbhd.abundance``
(study-level filtering + genus collapse), and ``mbhd.stats`` (from-scratch
Kruskal-Wallis + Benjamini-Hochberg) into the one comparison table this
project computes from scratch. Never imports ``scipy`` or ``statsmodels``
directly, and neither does anything it calls --
``tests/test_no_reference_library_imports.py`` enforces this.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from mbhd.abundance import (
    ZeroReadSampleError,
    analysis_samples,
    filter_table,
    to_genus_abundance,
)
from mbhd.datasets import (
    DatasetFileError,
    DatasetInfo,
    metadata_path,
    otu_table_path,
    study_dir,
)
from mbhd.io import (
    DISEASE_STATE_COLUMN,
    DuplicateSampleIdError,
    EmptyFileError,
    MissingColumnError,
    encoding_for,
    load_metadata,
    load_otu_table,
)
from mbhd.labels import (
    resolved_case_control,
    restrict_to_condition,
    restrict_to_sample_type,
)
from mbhd.stats import benjamini_hochberg, kruskal_wallis, signed_qvalues


class NoSurvivingSamplesError(ValueError):
    """Raised when one or both groups have no samples left after filtering."""


#: Errors that mean "this dataset's real data does not support a
#: reproduction attempt" (not extracted, malformed, or genuinely too small
#: after filtering) -- expected for some real datasets (e.g. ob_zupancic's
#: ZeroReadSampleError) and reported as a documented per-dataset failure, not
#: a crash. Any other exception is a real bug and must propagate (never
#: fall back silently). Shared by scripts/reproduce.py and
#: scripts/shared_response.py so the two scripts cannot silently diverge on
#: what counts as an expected, non-crashing failure.
EXPECTED_DATASET_ERRORS: tuple[type[Exception], ...] = (
    DatasetFileError,
    NoSurvivingSamplesError,
    DuplicateSampleIdError,
    EmptyFileError,
    MissingColumnError,
    ZeroReadSampleError,
    FileNotFoundError,
)


def differential_abundance(
    dataset_id: str,
    index: dict[str, DatasetInfo],
    data_root: Path,
    *,
    min_sample_reads: int = 100,
    min_otu_reads: int = 10,
    min_prevalence: float = 0.01,
    abundance_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Per-genus case/control differential abundance for ``dataset_id``.

    Pipeline: restrict to
    the study's own kept ``SampleType`` (``mbhd.labels.
    restrict_to_sample_type``) then its own kept ``condition:`` value
    (``mbhd.labels.restrict_to_condition``) -- each a no-op for the vast
    majority of datasets -- then to the study's labelled samples -> drop
    samples/OTUs failing the Methods' read-count and prevalence filters ->
    relative abundance over all
    surviving OTUs -> drop genus-unannotated OTUs -> collapse to genus ->
    two-group Kruskal-Wallis per genus -> Benjamini-Hochberg across all
    genera in this dataset -> sign from the group-mean comparison.

    ``abundance_transform``, if given, is applied to the genus x sample
    relative-abundance table immediately before the per-genus Kruskal-Wallis
    loop (e.g. ``functools.partial(mbhd.compositional.clr, pseudocount_fn=
    mbhd.compositional.multiplicative_replacement)`` for the M4 CLR path).
    Default ``None`` preserves exactly today's Euclidean (untransformed)
    behaviour -- regression-tested byte-identical to the existing M2/M3
    numbers. Kruskal-Wallis
    itself (``mbhd.stats``) is untouched either way; only the matrix feeding
    it changes.

    Returns a DataFrame indexed by genus lineage (``mbhd.abundance.genus_of``
    format) with columns:

    - ``p_value`` -- raw Kruskal-Wallis p-value.
    - ``q_value`` -- Benjamini-Hochberg-adjusted p-value (unsigned).
      Both raw and adjusted values are reported because every genus
      in a dataset is a separate hypothesis test.
    - ``signed_q_value`` -- ``q_value`` with the published sign convention
      (``mbhd.stats.signed_qvalues``); comparable directly to
      ``file-S1.qvalues.txt``.
    - ``control_mean`` / ``case_mean`` -- mean relative abundance per group.
    - ``log2_fold_change`` -- ``log2(case_mean / control_mean)``. 0.0 when
      both means are 0; ``+inf``/``-inf`` when exactly one mean is 0. The
      published ``file-S5.effects.txt`` instead fills these cases with the
      whole table's max/min value, which depends on every dataset's effect
      sizes jointly --
      deliberately not reproduced here; a caller comparing against
      ``file-S5`` must treat a non-finite value here as a known,
      unreproduced fill-convention difference, not a bug.
    - ``n_controls`` / ``n_cases`` -- surviving sample counts per group, so
      class imbalance is visible in the output rather than buried.

    Raises ``NoSurvivingSamplesError`` if either group has zero samples left
    after filtering (rather than silently running Kruskal-Wallis on one
    group, or crashing inside it with a confusing message).
    """
    folder = study_dir(index, dataset_id, data_root)
    metadata = load_metadata(
        metadata_path(dataset_id, folder), encoding=encoding_for(dataset_id)
    )
    metadata = restrict_to_sample_type(dataset_id, metadata)
    metadata = restrict_to_condition(dataset_id, metadata)
    otu_table = load_otu_table(otu_table_path(dataset_id, folder))

    sample_ids = analysis_samples(metadata, otu_table.columns)
    filtered = filter_table(
        otu_table,
        sample_ids,
        min_sample_reads=min_sample_reads,
        min_otu_reads=min_otu_reads,
        min_prevalence=min_prevalence,
    )
    genus_abundance = to_genus_abundance(filtered)
    if abundance_transform is not None:
        genus_abundance = abundance_transform(genus_abundance)

    spec = resolved_case_control(dataset_id)
    control_ids = [
        s
        for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.controls)]
        if s in genus_abundance.columns
    ]
    case_ids = [
        s
        for s in metadata.index[metadata[DISEASE_STATE_COLUMN].isin(spec.cases)]
        if s in genus_abundance.columns
    ]
    if not control_ids or not case_ids:
        raise NoSurvivingSamplesError(
            f"{dataset_id}: {len(control_ids)} control(s), {len(case_ids)} "
            "case(s) left after filtering -- need at least one of each"
        )

    genera = genus_abundance.index.tolist()
    p_values = np.empty(len(genera))
    control_means = np.empty(len(genera))
    case_means = np.empty(len(genera))
    for i, genus in enumerate(genera):
        control_values = genus_abundance.loc[genus, control_ids].to_numpy()
        case_values = genus_abundance.loc[genus, case_ids].to_numpy()
        _, p_values[i] = kruskal_wallis(control_values, case_values)
        control_means[i] = control_values.mean()
        case_means[i] = case_values.mean()

    q_values = benjamini_hochberg(p_values)
    signed_q = signed_qvalues(q_values, control_means, case_means)
    log2_fold_change = _log2_fold_change(control_means, case_means)

    return pd.DataFrame(
        {
            "p_value": p_values,
            "q_value": q_values,
            "signed_q_value": signed_q,
            "control_mean": control_means,
            "case_mean": case_means,
            "log2_fold_change": log2_fold_change,
            "n_controls": len(control_ids),
            "n_cases": len(case_ids),
        },
        index=pd.Index(genera, name="genus"),
    )


def _log2_fold_change(control_means: np.ndarray, case_means: np.ndarray) -> np.ndarray:
    """``log2(case / control)`` per genus, with the zero-mean cases defined.

    0.0 when both means are 0 (no signal either way); ``+inf`` when only the
    control mean is 0 (infinitely enriched in cases); ``-inf`` when only the
    case mean is 0 (infinitely depleted). See ``differential_abundance``'s
    docstring for why this deliberately does not match ``file-S5``'s
    whole-table max/min fill convention.
    """
    result = np.empty(control_means.shape, dtype=float)
    both_zero = (control_means == 0) & (case_means == 0)
    control_zero = (control_means == 0) & ~both_zero
    case_zero = (case_means == 0) & ~both_zero
    finite = ~both_zero & ~control_zero & ~case_zero

    result[both_zero] = 0.0
    result[control_zero] = np.inf
    result[case_zero] = -np.inf
    result[finite] = np.log2(case_means[finite] / control_means[finite])
    return result
