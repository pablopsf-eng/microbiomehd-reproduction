"""Filtering, genus collapse, and relative-abundance computation for one study.

**Orientation, never mixed:** every table here is OTUs/genera x samples
(rows = lineages, columns = sample IDs), matching ``mbhd.io.load_otu_table``.

**Sample scope for filtering is the whole study, not one comparison.** The
Methods' "OTUs present in under 1% of a study's samples" means every sample
sharing a study's metadata and OTU table, including disease arms -- and even
blank-label samples -- that are not part of the current case/control
comparison. Verified against real data three separate ways, each ruling out
a narrower candidate rule:

- ``cdi_schubert``'s comparison-only sample set (H vs CDI, 247 samples)
  reproduces only 113 of file-S1's 120 published genera; including its
  shared ``noncdi_schubert`` arm (336 samples total) reproduces all 120.
  Rules out "only the current comparison's two groups".
- ``crc_baxter``'s unused third arm, ``nonCRC`` (198 of 490 samples), is not
  any dataset ID's control or case label anywhere in
  ``mbhd.labels.CASE_CONTROL``, yet must be included to reproduce its
  published 163-genus set exactly -- H+CRC alone (292 samples) gives only
  153. Rules out "only labels some dataset ID actually recognises".
- ``crc_zeller`` -- one of M1's 18 *exactly sample-count-reconciled*
  datasets, so not a case anyone would suspect -- has 13 samples with a
  **blank** ``DiseaseState`` inside the metadata/OTU-table intersection.
  Excluding them (the natural-seeming rule: only labelled samples count)
  reproduces 186 of file-S1's 193 published genera, missing exactly 7:
  removing 13 samples from the denominator pushes some OTUs (e.g.
  ``g__Collinsella`` at 8 total reads) under the 10-read floor, and others
  (e.g. ``g__Polaromonas``, otherwise abundant but present in only 1 of the
  remaining 116 samples) under the 1% prevalence floor -- both filters lose
  genera here, not only the read-count one. Including all 13 reproduces all
  193 exactly. Rules out "only samples with a real disease label" -- a blank
  ``DiseaseState`` still contributes reads to the prevalence/read-count
  denominator even though it contributes no group membership.

The rule that survives all three: **every sample in the metadata/OTU-table
intersection**, full stop, regardless of its ``DiseaseState`` value --
exactly ``mbhd.labels.intersected_disease_states``'s own index, with no
further restriction. See ``analysis_samples``.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from mbhd.labels import intersected_disease_states


class ZeroReadSampleError(ValueError):
    """Raised when a sample has zero total reads over the OTUs being normalised."""


def analysis_samples(
    metadata: pd.DataFrame, otu_sample_ids: Iterable[str]
) -> list[str]:
    """All sample IDs in this study's scope, for use as ``filter_table``'s scope.

    Every sample present in both ``metadata`` and the OTU table
    (``mbhd.labels.intersected_disease_states`` -- the same metadata/OTU-
    table intersection rule ``reconcile_counts`` uses in M1), regardless of
    its ``DiseaseState`` value: not restricted to a recognised control/case
    label (see ``crc_baxter``'s ``nonCRC`` evidence in the module docstring),
    and not restricted to a non-blank label either (see ``crc_zeller``'s
    evidence) -- a blank-labelled sample still has real read counts that
    belong in the prevalence/read-count denominator, even though it
    contributes no group membership at the comparison stage.
    """
    observed = intersected_disease_states(metadata, set(otu_sample_ids))
    return observed.index.tolist()


def filter_table(
    otu_table: pd.DataFrame,
    sample_ids: Iterable[str],
    *,
    min_sample_reads: int = 100,
    min_otu_reads: int = 10,
    min_prevalence: float = 0.01,
) -> pd.DataFrame:
    """Apply the Methods' three study-level filters, in the verified order.

    Methods (Duvallet et al. 2017): "samples with fewer than 100 reads and
    OTUs with fewer than 10 reads were removed, along with OTUs present in
    under 1% of a study's samples." All three thresholds are inclusive at
    the boundary (``>=``, matching "fewer than X removed" = "X or more
    kept") and are named arguments rather than magic numbers so a caller can
    state exactly which Methods sentence justifies each one.

    ``sample_ids`` must be the study's *labelled* samples
    (``analysis_samples``), not just the current comparison's two groups --
    see the module docstring for why restricting to the comparison changes
    the 1% prevalence denominator and reproduces the wrong genus set.

    Order, empirically verified to matter (a different order does not
    reproduce file-S1's genus sets): (1) drop samples under
    ``min_sample_reads`` total reads, (2) drop OTUs under ``min_otu_reads``
    total reads over the *surviving* samples, (3) drop OTUs present in under
    ``min_prevalence`` fraction of the surviving samples. Returns raw counts,
    not yet relative abundance -- see ``to_genus_abundance`` for why
    normalising must happen after this, not before.
    """
    present = [s for s in sample_ids if s in otu_table.columns]
    table = otu_table[present]

    sample_reads = table.sum(axis=0)
    table = table.loc[:, sample_reads >= min_sample_reads]

    otu_reads = table.sum(axis=1)
    table = table.loc[otu_reads >= min_otu_reads]

    n_samples = table.shape[1]
    prevalence = (table > 0).sum(axis=1) / n_samples
    table = table.loc[prevalence >= min_prevalence]

    return table


def genus_of(lineage: str) -> str | None:
    """The genus-level prefix of a full OTU lineage, or ``None`` if unannotated.

    An OTU-table lineage carries two fields past genus:
    ``...;g__X;s__;d__denovoN`` -- verified across every lineage in the real
    downloaded data (19,314 OTUs checked for ``cdi_schubert`` alone): always
    exactly 8 ``;``-separated fields, with ``g__...`` always third from the
    end. ``file-S1.qvalues.txt``'s index is the lineage truncated exactly at
    that field (dropping ``s__`` and ``d__``), reproduced character-for-
    character by dropping the last two fields here.

    An OTU with an empty genus field (literally ``g__``, nothing after it)
    is unannotated at genus level -- the Methods call for these to be
    discarded before genus-level collapse -- and this returns ``None``.

    Raises ``ValueError`` if ``lineage`` does not have the expected 8-field
    shape with ``g__`` third from the end, rather than silently returning a
    wrong or partial genus string for a lineage from a source this project
    has not yet seen.
    """
    fields = lineage.split(";")
    if len(fields) < 3 or not fields[-3].startswith("g__"):
        raise ValueError(
            f"lineage does not have the expected '...;g__...;s__...;d__...' "
            f"shape (g__ three fields from the end): {lineage!r}"
        )
    if fields[-3] == "g__":
        return None
    return ";".join(fields[:-2])


def to_genus_abundance(otu_table: pd.DataFrame) -> pd.DataFrame:
    """Relative abundance over all surviving OTUs, then drop unannotated, then collapse.

    **This exact order is the single most important, least obvious finding
    behind this module**:
    genus-unannotated OTUs (~29% of ``cdi_schubert``'s reads after
    filtering) stay in the relative-abundance *denominator*. Verified
    against ``file-S5.effects.txt``'s published log2 fold-changes for
    ``cdi_schubert``: this order reproduces 92/92 finite comparisons to
    within 1e-12, while dropping unannotated OTUs *before* normalising is
    off by orders of magnitude (median absolute difference 1.89e-01, 0/92
    within 1e-6). Genus-level relative abundances therefore do **not** sum
    to 1 within a sample -- that is expected, not a bug.

    Input and output are both OTUs/genera x samples. Raises
    ``ZeroReadSampleError`` naming any sample with zero total reads over
    ``otu_table`` (which would otherwise silently produce a ``0/0`` -> NaN
    relative abundance for that whole sample column) -- callers should not
    pass a table where ``filter_table``'s own thresholds have left a sample
    with nothing in it. Observed for real on ``ob_zupancic`` (a dataset whose
    sample counts do not reconcile with Table 1) during the all-30 run -- a
    real data condition on a dataset already flagged as unreconciled, not a
    bug in this function.
    """
    totals = otu_table.sum(axis=0)
    empty_samples = totals.index[totals == 0].tolist()
    if empty_samples:
        raise ZeroReadSampleError(
            f"sample(s) with zero total reads after OTU-level filtering, "
            f"relative abundance is undefined for them: {empty_samples}"
        )

    relative = otu_table.div(totals, axis=1)

    genus = pd.Series(otu_table.index, index=otu_table.index).map(genus_of)
    annotated = relative.loc[genus.notna()].copy()
    annotated.index = genus.loc[genus.notna()].to_numpy()

    return annotated.groupby(level=0).sum()
