"""Case/control labelling for the 30 analysed MicrobiomeHD datasets.

Each dataset's controls/cases mapping is **transcribed from Table 1** of
Duvallet et al. (2017) (``tests/data/table1_sample_counts.csv``), which names
every study's controls label and cases label directly -- this is not a
guessed heuristic. A handful of Table 1's label strings do not match the
metadata's actual ``DiseaseState`` value verbatim (e.g. Table 1 writes
"non-IBD", the data says ``nonIBD``); ``LABEL_ALIASES`` records every one of
those, each accepted only after it produced an exact match against Table 1's
published count, never guessed ahead of that check. A compound Table 1 label
("UC, CD", "PSA, RA", "CIRR, MHE") means the disease group is the union of
more than one ``DiseaseState`` value, so ``CASE_CONTROL`` maps to *lists*.

A sample counts toward a group only if its ID is in **both** the metadata
and the OTU table (this, not ``dataset_info.yaml``'s own
``sample_size`` field, is the rule that reproduces Table 1).
Any ``DiseaseState`` value present in a dataset but not listed in either
group is excluded and reported, never silently folded into one side.

**5 of 30 datasets do not reconcile to Table 1's published count even after
label-alias correction.** Real evidence was gathered for several (ruling out,
not confirming, specific mechanisms) but
none was resolved, so none is guessed here. ``KNOWN_MISMATCHES`` pins each
one's currently-observed (controls, cases) pair so that ``tests/
test_table1_counts.py`` catches any *change* to the observed number as a
regression, while the pre-existing, documented gap itself does not fail the
suite -- "could not verify" is an acceptable answer; inventing a filter to
force a match is not.

Seven datasets started in this state but were resolved by finding a real,
previously-unapplied filtering rule, not by guessing: ``cdi_youngster``
(found in M3) counts distinct **patients**,
not samples, for this one longitudinal FMT study -- ``SUBJECT_ID_COLUMN``
records the metadata column to collapse on for that one dataset, every
other dataset still counts samples. ``ibd_morgan``/``ibd_willing`` and
``edd_singh``/``hiv_lozupone``/``nash_wong``/``ob_goodrich`` each had real
samples that should have been excluded before counting -- non-stool
``SampleType`` values for the first two (``SAMPLE_TYPE_FILTER``), a value of
MicrobiomeHD's own upstream ``dataset_info.yaml`` ``condition:`` field for
the other four (``CONDITION_FILTER``).
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from mbhd.io import DISEASE_STATE_COLUMN


class CaseControlSpec(NamedTuple):
    """The DiseaseState values counted as controls / cases for one dataset."""

    controls: list[str]
    cases: list[str]


class LabelCounts(NamedTuple):
    """Observed sample counts after the metadata/OTU-table intersection."""

    controls: int
    cases: int
    #: DiseaseState value -> count, for values present in the intersected
    #: samples but listed as neither control nor case.
    excluded: dict[str, int]


#: Table 1's controls_label/cases_label, transcribed per dataset -- see the
#: module docstring. Table 1's own label string, not yet alias-corrected.
CASE_CONTROL: dict[str, CaseControlSpec] = {
    "art_scher": CaseControlSpec(controls=["H"], cases=["PSA", "RA"]),
    "asd_kang": CaseControlSpec(controls=["H"], cases=["ASD"]),
    "asd_son": CaseControlSpec(controls=["H"], cases=["ASD"]),
    "cdi_schubert": CaseControlSpec(controls=["H"], cases=["CDI"]),
    "cdi_vincent": CaseControlSpec(controls=["H"], cases=["CDI"]),
    "cdi_youngster": CaseControlSpec(controls=["H"], cases=["CDI"]),
    "crc_baxter": CaseControlSpec(controls=["H"], cases=["CRC"]),
    "crc_chen": CaseControlSpec(controls=["H"], cases=["CRC"]),
    "crc_wang": CaseControlSpec(controls=["H"], cases=["CRC"]),
    "crc_zeller": CaseControlSpec(controls=["H"], cases=["CRC"]),
    "edd_singh": CaseControlSpec(controls=["H"], cases=["EDD"]),
    "hiv_dinh": CaseControlSpec(controls=["H"], cases=["HIV"]),
    "hiv_lozupone": CaseControlSpec(controls=["H"], cases=["HIV"]),
    "hiv_noguerajulian": CaseControlSpec(controls=["H"], cases=["HIV"]),
    # Table 1: "non-IBD" -> actual value "nonIBD" (LABEL_ALIASES).
    "ibd_gevers": CaseControlSpec(controls=["non-IBD"], cases=["CD"]),
    "ibd_morgan": CaseControlSpec(controls=["H"], cases=["UC", "CD"]),
    "ibd_papa": CaseControlSpec(controls=["non-IBD"], cases=["UC", "CD"]),
    "ibd_willing": CaseControlSpec(controls=["H"], cases=["UC", "CD"]),
    "liv_zhang": CaseControlSpec(controls=["H"], cases=["CIRR", "MHE"]),
    "nash_wong": CaseControlSpec(controls=["H"], cases=["NASH"]),
    "nash_zhu": CaseControlSpec(controls=["H"], cases=["NASH"]),
    # Table 1: "non-CDI" -> actual value "nonCDI" (LABEL_ALIASES).
    "noncdi_schubert": CaseControlSpec(controls=["H"], cases=["non-CDI"]),
    "ob_goodrich": CaseControlSpec(controls=["H"], cases=["OB"]),
    "ob_ross": CaseControlSpec(controls=["H"], cases=["OB"]),
    "ob_turnbaugh": CaseControlSpec(controls=["H"], cases=["OB"]),
    # Table 1: "OB" -> actual value "nonNASH-OB" (LABEL_ALIASES). ob_zhu's
    # metadata has no "OB" DiseaseState value at all; "nonNASH-OB" is the
    # obesity arm of this shared nash_zhu/ob_zhu cohort.
    "ob_zhu": CaseControlSpec(controls=["H"], cases=["OB"]),
    "ob_zupancic": CaseControlSpec(controls=["H"], cases=["OB"]),
    "par_scheperjans": CaseControlSpec(controls=["H"], cases=["PAR"]),
    "t1d_alkanani": CaseControlSpec(controls=["H"], cases=["T1D"]),
    "t1d_mejialeon": CaseControlSpec(controls=["H"], cases=["T1D"]),
}

#: {dataset_id: {table1_label: actual_disease_state_value}}. Applied to both
#: controls and cases before matching. Each entry was accepted only after
#: verifying it produces an exact match against Table 1's published count for
#: that dataset -- see the module docstring.
LABEL_ALIASES: dict[str, dict[str, str]] = {
    "ibd_gevers": {"non-IBD": "nonIBD"},
    "ibd_papa": {"non-IBD": "nonIBD"},
    "noncdi_schubert": {"non-CDI": "nonCDI"},
    "ob_zhu": {"OB": "nonNASH-OB"},
}

#: dataset_id -> metadata column of distinct-patient identifiers to collapse
#: samples onto before comparing against Table 1, overriding the default
#: per-sample count. Discovered directly from real data, not guessed:
#: cdi_youngster's 18 intersected H-labelled *samples* collapse to exactly 4
#: distinct ``subject`` values, and its 27 CDI samples collapse to exactly
#: 19 -- both match Table 1's published (4, 19) exactly.
#: This is a longitudinal FMT study with repeat sampling per patient; every
#: other dataset in CASE_CONTROL counts samples, matching Table 1's own
#: convention for the rest of the paper.
SUBJECT_ID_COLUMN: dict[str, str] = {"cdi_youngster": "subject"}

#: dataset_id -> (SampleType-like column, value to keep). Two datasets
#: include non-stool samples (biopsy/tissue) the paper's own "stool
#: samples only" Methods criterion excludes -- discovered by surveying
#: every one of the 30 analysed datasets' SampleType-like columns and
#: cross-checking against Table 1: restricting each of these two to its "stool"
#: rows reproduces Table 1 exactly. Only one other dataset, hiv_lozupone, has
#: any non-stool value at all (6 "control blank" rows) -- those already carry a blank
#: DiseaseState and are already excluded by the existing intersection
#: rule, so no entry is needed for it; no other dataset, including any of
#: the already-exactly-reconciled ones, has a non-stool sample.
SAMPLE_TYPE_FILTER: dict[str, tuple[str, str]] = {
    "ibd_morgan": ("SampleType", "stool"),
    "ibd_willing": ("SampleType", "stool"),
}

#: dataset_id -> (metadata column, values to keep). Sourced from the
#: MicrobiomeHD upstream db/dataset_info.yaml's own (previously-unused)
#: `condition:` field, already parsed into DatasetInfo.raw but read by no
#: code before this. Every entry here has been independently verified to
#: reproduce Table 1 *exactly* via reconcile_counts against the real
#: downloaded data -- this dict is not a blind transcription of
#: dataset_info.yaml: datasets
#: whose condition field only improves, but does not exactly reconcile, the
#: count (hiv_noguerajulian, ob_zupancic) are deliberately left out and stay
#: in KNOWN_MISMATCHES with an investigation note instead. cdi_youngster
#: also has a condition: entry but is already correctly handled by
#: SUBJECT_ID_COLUMN -- not duplicated here.
CONDITION_FILTER: dict[str, tuple[str, tuple[object, ...]]] = {
    "edd_singh": ("Time Point", (1,)),
    "hiv_lozupone": ("time_point", ("1",)),
    "nash_wong": ("Status_s", ("Baseline",)),
    "ob_goodrich": ("n_sample", (0,)),
}

#: dataset_id -> (observed_controls, observed_cases) as computed by
#: reconcile_counts() against the real, downloaded data. Pinned so a change
#: in the observed count -- for better or worse -- is caught by
#: tests/test_table1_counts.py rather than silently drifting further from
#: (or coincidentally back toward) Table 1. None is resolved, none is guessed.
#: ibd_morgan/ibd_willing removed (not re-pinned): SAMPLE_TYPE_FILTER now
#: makes both reconcile exactly, the same way cdi_youngster was removed in
#: M3 once its own mismatch was resolved. edd_singh/hiv_lozupone/nash_wong/
#: ob_goodrich removed the same way once CONDITION_FILTER made all four
#: reconcile exactly.
KNOWN_MISMATCHES: dict[str, tuple[int, int]] = {
    "crc_wang": (56, 46),
    "hiv_noguerajulian": (57, 293),
    "liv_zhang": (26, 51),
    "ob_turnbaugh": (61, 196),
    "ob_zupancic": (127, 128),
}


def restrict_to_sample_type(dataset_id: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Restricts ``metadata`` to the single kept ``SampleType``-like value
    if ``dataset_id`` has a ``SAMPLE_TYPE_FILTER`` entry; returns it
    unchanged otherwise (the vast majority of datasets, which either have
    no such column or are already 100% stool).

    Must be called once, immediately after ``load_metadata``, before any
    other sample-selection logic sees the frame -- this is the single
    place the paper's "stool samples only" Methods rule is enforced
    (enforced nowhere before this was found).

    Raises ``ValueError`` naming the dataset if the named column is not
    present in ``metadata`` -- a silent no-op here would defeat the whole
    point of a filter meant to exclude real samples.
    """
    entry = SAMPLE_TYPE_FILTER.get(dataset_id)
    if entry is None:
        return metadata
    column, keep_value = entry
    if column not in metadata.columns:
        raise ValueError(
            f"{dataset_id}: SAMPLE_TYPE_FILTER names column {column!r}, "
            f"not present in metadata (columns: {list(metadata.columns)})"
        )
    return metadata.loc[metadata[column] == keep_value]


def restrict_to_condition(dataset_id: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Restricts ``metadata`` to the kept values of a ``CONDITION_FILTER``
    entry's column if ``dataset_id`` has one; returns it unchanged otherwise
    (the vast majority of datasets, which have no such entry).

    Same contract as ``restrict_to_sample_type``: called once, immediately
    after it, before any other sample-selection logic sees the frame. No
    dataset has entries in both dicts today, so the two calls' relative
    order does not matter functionally.

    Raises ``ValueError`` naming the dataset and column if the named column
    is not present in ``metadata`` -- a silent no-op here would defeat the
    whole point of a filter meant to exclude real samples.
    """
    entry = CONDITION_FILTER.get(dataset_id)
    if entry is None:
        return metadata
    column, keep_values = entry
    if column not in metadata.columns:
        raise ValueError(
            f"{dataset_id}: CONDITION_FILTER names column {column!r}, "
            f"not present in metadata (columns: {list(metadata.columns)})"
        )
    return metadata.loc[metadata[column].isin(keep_values)]


def _resolve(dataset_id: str, labels: list[str]) -> list[str]:
    """Apply LABEL_ALIASES to one dataset's control or case label list."""
    aliases = LABEL_ALIASES.get(dataset_id, {})
    return [aliases.get(label, label) for label in labels]


def resolved_case_control(dataset_id: str) -> CaseControlSpec:
    """``CASE_CONTROL[dataset_id]`` with ``LABEL_ALIASES`` already applied.

    Returns the actual ``DiseaseState`` values, not Table 1's own label text
    -- e.g. ``ibd_gevers``'s controls become ``["nonIBD"]``, not
    ``["non-IBD"]``. The single place this resolution happens, used by both
    ``reconcile_counts`` (M1) and ``mbhd.differential.differential_abundance``
    (M2), so a caller building the actual control/case sample sets never
    reimplements alias lookup.

    Raises ``KeyError`` if ``dataset_id`` has no entry in ``CASE_CONTROL``.
    """
    spec = CASE_CONTROL[dataset_id]
    return CaseControlSpec(
        controls=_resolve(dataset_id, spec.controls),
        cases=_resolve(dataset_id, spec.cases),
    )


def intersected_disease_states(
    metadata: pd.DataFrame, otu_sample_ids: set[str]
) -> pd.Series:
    """``DiseaseState`` values for samples present in both tables.

    The shared primitive behind ``reconcile_counts`` (M1) and
    ``mbhd.abundance.analysis_samples`` (M2): both need "which samples are
    genuinely in this study's processed data" -- a sample counts only if its
    ID is in both ``metadata.index`` and ``otu_sample_ids``
    -- and then use that set differently:
    ``reconcile_counts`` keeps only the samples whose value is one of the
    recognised control/case labels for one dataset ID's comparison;
    ``analysis_samples`` keeps every sample regardless of its value,
    including a blank one -- a different, larger set (see
    ``mbhd.abundance``'s module docstring for why blank labels must stay in
    the M2 filtering scope, verified against ``crc_zeller``'s real data).
    """
    intersected = metadata.index.isin(otu_sample_ids)
    return metadata.loc[intersected, DISEASE_STATE_COLUMN]


def reconcile_counts(
    dataset_id: str, metadata: pd.DataFrame, otu_sample_ids: set[str]
) -> LabelCounts:
    """Count controls/cases for ``dataset_id`` using the verified intersection rule.

    ``metadata`` is first restricted by ``restrict_to_sample_type`` and then
    ``restrict_to_condition`` (each a no-op for the vast majority of
    datasets, which have no entry in the respective dict). A sample counts
    only if its ID is in both ``metadata.index`` and ``otu_sample_ids``.
    Any ``DiseaseState`` value among the intersected samples that is listed as
    neither a control nor a case is excluded and reported in
    ``LabelCounts.excluded`` -- never silently added to
    either group.

    For a dataset listed in ``SUBJECT_ID_COLUMN``, controls/cases are counted
    as *distinct values* of that metadata column among the intersected
    samples, not sample counts -- see ``SUBJECT_ID_COLUMN``'s docstring.
    ``excluded`` still reports raw *sample* counts for unmapped
    ``DiseaseState`` values either way, since Table 1 has no patient-level
    concept of an excluded group to compare against.

    Raises ``KeyError`` if ``dataset_id`` has no entry in ``CASE_CONTROL``, and
    ``ValueError`` naming the dataset if ``metadata`` has no ``DiseaseState``
    column -- checked here too, not only by ``mbhd.io.load_metadata``, so a
    caller that builds ``metadata`` some other way still gets a named error
    instead of a bare pandas ``KeyError``.
    """
    metadata = restrict_to_sample_type(dataset_id, metadata)
    metadata = restrict_to_condition(dataset_id, metadata)
    spec = resolved_case_control(dataset_id)

    if DISEASE_STATE_COLUMN not in metadata.columns:
        raise ValueError(
            f"{dataset_id}: metadata has no {DISEASE_STATE_COLUMN!r} column "
            f"(columns present: {list(metadata.columns)})"
        )

    observed = intersected_disease_states(metadata, otu_sample_ids)

    is_control = observed.isin(spec.controls)
    is_case = observed.isin(spec.cases)
    excluded = observed[~is_control & ~is_case].value_counts().to_dict()

    subject_column = SUBJECT_ID_COLUMN.get(dataset_id)
    if subject_column is not None:
        controls = metadata.loc[observed.index[is_control], subject_column].nunique()
        cases = metadata.loc[observed.index[is_case], subject_column].nunique()
    else:
        controls = int(is_control.sum())
        cases = int(is_case.sum())

    return LabelCounts(controls=controls, cases=cases, excluded=excluded)
