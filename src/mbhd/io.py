"""Load OTU tables and metadata for one MicrobiomeHD study into pandas DataFrames.

**Orientation, stated once, never mixed:** an OTU table loaded by
``load_otu_table`` is OTUs x samples (rows = OTU/genus lineages, columns =
sample IDs). Metadata loaded by ``load_metadata`` is samples x fields (rows =
sample IDs, columns = the study's metadata fields). This matches how the
files are stored on disk -- no transpose happens in either loader.

**Encoding is per-study, not universal.** Confirmed for cdi_schubert:
ISO-8859-1 (a `®` byte at file offset 799, in "MoBio PowerSoil®-htp"), which
raises ``UnicodeDecodeError`` under the default UTF-8 decode. Every other
study's metadata surveyed so far decodes as UTF-8. ``METADATA_ENCODINGS`` is
the single, explicit per-dataset-ID override list; there is deliberately no
try-UTF-8-then-fall-back-to-latin-1 path, because latin-1 decodes any byte
sequence at all and would silently mis-render genuinely corrupt bytes in a
study not yet known to need the override, rather than raising on them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Per-dataset-ID metadata encoding overrides. Default is "utf-8" (see
#: encoding_for). Derived by attempting a UTF-8 decode of all 30 analysed
#: datasets' metadata files and recording every one that failed -- not a
#: guess. Extend this if a future dataset is added and also fails to decode.
METADATA_ENCODINGS: dict[str, str] = {
    "cdi_schubert": "latin-1",
    "noncdi_schubert": "latin-1",  # same metadata file as cdi_schubert
    "crc_zeller": "latin-1",
}

#: The column every study's metadata is expected to carry -- the pipeline's
#: own standardized case/control label, not the submitter's original column
#: (which varies by study: disease_stat, Diagnostic, etc.).
DISEASE_STATE_COLUMN = "DiseaseState"


class EmptyFileError(ValueError):
    """Raised when a table file exists but has no data rows."""


class DuplicateSampleIdError(ValueError):
    """Raised when a table's sample-ID axis (columns for OTU tables, index
    for metadata) contains the same ID more than once."""


class MissingColumnError(ValueError):
    """Raised when metadata is missing a column this project requires."""


def encoding_for(dataset_id: str) -> str:
    """The text encoding to use for ``dataset_id``'s metadata file.

    Returns the override in ``METADATA_ENCODINGS`` if one is recorded,
    otherwise "utf-8". Callers should not guess encodings themselves -- add
    a confirmed override here instead, so every caller sees it.
    """
    return METADATA_ENCODINGS.get(dataset_id, "utf-8")


def load_otu_table(path: Path) -> pd.DataFrame:
    """Load a study's RDP-assigned OTU table.

    Orientation: OTUs x samples. Index is the full ``k__...;g__...;d__denovoN``
    lineage string (blank ``g__`` means the OTU was not annotated to genus
    level -- the Methods' own filter for that has already been applied by the
    RDP-assigned variant, verified on the real data). Columns are sample
    IDs. Values are non-negative integer read counts.

    Raises ``FileNotFoundError`` if ``path`` does not exist, ``EmptyFileError``
    if it has a header but no data rows, and ``DuplicateSampleIdError`` if the
    same sample ID appears as more than one column.
    """
    if not path.exists():
        raise FileNotFoundError(f"OTU table not found: {path}")

    # pandas silently disambiguates a duplicate header cell ("S1" and a second
    # "S1" become "S1" and "S1.1") rather than keeping it a real duplicate, so
    # checking table.columns *after* read_csv can never see the collision.
    # The raw header line is checked first, before that renaming happens.
    with open(path, newline="") as f:
        header_cells = f.readline().rstrip("\n").split("\t")[1:]
    seen: set[str] = set()
    header_duplicates = [c for c in header_cells if c in seen or seen.add(c)]
    if header_duplicates:
        raise DuplicateSampleIdError(
            f"{path}: duplicate sample ID(s) in OTU table columns: "
            f"{sorted(set(header_duplicates))}"
        )

    table = pd.read_csv(path, sep="\t", index_col=0)

    if table.shape[0] == 0:
        raise EmptyFileError(f"{path}: header present but no data rows")

    return table


def load_metadata(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a study's metadata file.

    Orientation: samples x fields. Index is the first column regardless of
    its header name (observed to vary across studies: ``sample_id``,
    ``#SampleID``, ``Sample_Name_s``, or blank). Requires a ``DiseaseState``
    column, since that is the pipeline's own standardized case/control label;
    the submitter's original label column differs per study and must not be used as a
    substitute.

    The index column is forced to ``str`` (``dtype={0: str}``): crc_baxter's
    sample IDs are all-digit (e.g. ``2005650``), and left to pandas' own type
    inference they parse as ``int64`` while the OTU table's columns -- taken
    from a header row, always strings -- do not. Without this, every sample
    ID in that study silently fails to match between the two tables (0 of 490
    intersect) despite being character-for-character identical; this was
    caught only by cross-checking against Table 1, not by any exception.

    ``keep_default_na=False``: pandas' default NA-string list includes the
    literal text ``"NA"``, which is a real, present ``DiseaseState`` value in
    at least one dataset (ibd_willing -- 10 rows). Left on, those cells parse
    as float ``nan``, and ``mbhd.labels.reconcile_counts``'s exclusion report
    (built with ``value_counts()``, which drops NaN by default) then loses
    them silently -- the counts stay correct, since ``isin()`` on NaN is
    always False, but the samples vanish from view exactly where a reader
    would look to understand why a dataset doesn't reconcile with Table 1.
    With this off, a blank cell becomes an empty string, a real and visible
    value, not an invisible one.

    Raises ``FileNotFoundError`` if ``path`` does not exist,
    ``UnicodeDecodeError`` (uncaught) if ``encoding`` is wrong for this file --
    see ``encoding_for`` for the per-study override table --
    ``EmptyFileError`` if it has a header but no data rows,
    ``DuplicateSampleIdError`` if the same sample ID appears as more than one
    row, and ``MissingColumnError`` if there is no ``DiseaseState`` column.
    """
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")

    metadata = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        encoding=encoding,
        dtype={0: str},
        keep_default_na=False,
    )

    if metadata.shape[0] == 0:
        raise EmptyFileError(f"{path}: header present but no data rows")

    duplicates = metadata.index[metadata.index.duplicated()].unique().tolist()
    if duplicates:
        raise DuplicateSampleIdError(
            f"{path}: duplicate sample ID(s) in metadata index: {duplicates}"
        )

    if DISEASE_STATE_COLUMN not in metadata.columns:
        raise MissingColumnError(
            f"{path}: no {DISEASE_STATE_COLUMN!r} column "
            f"(columns present: {list(metadata.columns)})"
        )

    return metadata
