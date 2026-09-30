"""Parse dataset_info.yaml and resolve per-study file paths within data/raw.

Folder names are not dataset IDs. dataset_info.yaml's ``folder:`` field is
named for the study's senior author (e.g. ``art_scher`` -> ``ra_littman_
results``), while the dataset ID uses the first author -- eleven of the 33
active IDs differ from their folder. Never build a path by string-formatting
a dataset ID; always resolve it through ``folder_for`` (or, for the actual
OTU table filename, through ``summary_file.txt`` -- see ``otu_table_path``).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

#: Active dataset IDs with no column in file-S1.qvalues.txt and no Table 1
#: row. Present in MicrobiomeHD but not carried into the paper's analysis --
#: not an anomaly, the database is deliberately broader than the publication.
UNANALYSED_DATASET_IDS = frozenset({"crc_wu", "crc_zackular", "ob_escobar"})


class DatasetInfo(NamedTuple):
    """One active entry from dataset_info.yaml."""

    dataset_id: str
    folder: str
    #: The full parsed yaml block for this dataset, e.g. region, sequencer,
    #: sample_size. NOTE: ``sample_size`` here is the source publication's own
    #: reported counts, not a measurement of the processed data -- it does not
    #: match Table 1 for every dataset (e.g. cdi_schubert: the yaml gives H 155,
    #: Table 1 gives H 154).
    #: The metadata/OTU-table intersection rule reproduces Table 1; this field
    #: does not. Never use it as a sample count; compute counts from the
    #: metadata/OTU-table intersection instead.
    raw: dict[str, object]


class UnknownDatasetError(LookupError):
    """Raised when a requested dataset ID is not an active entry."""


class DatasetFileError(Exception):
    """Raised when a study's folder or an expected file within it is missing.

    Always names the dataset ID and the path that was expected, never a bare
    FileNotFoundError with no context about which of the 30+ studies failed.
    """


def load_dataset_info(path: Path) -> dict[str, DatasetInfo]:
    """Parse dataset_info.yaml into ``{dataset_id: DatasetInfo}``.

    Uses ``yaml.safe_load`` -- required, not optional: the file contains free
    -text fields and a commented-out dataset block that break
    naive whitespace splitting. A commented-out block is a YAML comment and
    never appears in the parsed result.

    Raises ``ValueError`` if the file parses to something other than a
    mapping, or if any entry is missing its ``folder`` field.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{path}: expected a non-empty mapping of dataset IDs, "
            f"got {type(raw).__name__}"
        )

    result: dict[str, DatasetInfo] = {}
    for dataset_id, block in raw.items():
        if not isinstance(block, dict) or "folder" not in block:
            raise ValueError(f"{path}: dataset {dataset_id!r} has no 'folder' field")
        result[dataset_id] = DatasetInfo(
            dataset_id=dataset_id, folder=block["folder"], raw=block
        )
    return result


def entry_for(index: dict[str, DatasetInfo], dataset_id: str) -> DatasetInfo:
    """Look up ``dataset_id`` in a loaded dataset_info index.

    Raises ``UnknownDatasetError`` naming the id if it is not an active entry.
    """
    try:
        return index[dataset_id]
    except KeyError:
        raise UnknownDatasetError(
            f"{dataset_id!r} is not an active dataset in dataset_info.yaml"
        ) from None


def analysed_dataset_ids(index: dict[str, DatasetInfo]) -> list[str]:
    """The 30 dataset IDs carried into the paper's analysis.

    Active IDs (33) minus the 3 documented present-but-unanalysed IDs
    (``UNANALYSED_DATASET_IDS``). Matches the 30 columns of
    file-S1.qvalues.txt and the 30 rows of Table 1.
    """
    return sorted(set(index) - UNANALYSED_DATASET_IDS)


def study_dir(index: dict[str, DatasetInfo], dataset_id: str, data_root: Path) -> Path:
    """The extracted study directory for ``dataset_id`` under ``data_root``.

    Raises ``DatasetFileError`` naming the dataset and the missing path if the
    folder was not extracted (e.g. ``scripts/download.py`` has not been run).
    """
    entry = entry_for(index, dataset_id)
    path = data_root / entry.folder
    if not path.is_dir():
        raise DatasetFileError(
            f"{dataset_id}: expected extracted folder at {path}, "
            "not found -- run scripts/download.py first"
        )
    return path


def load_summary_file(path: Path) -> dict[str, str]:
    """Parse a study's summary_file.txt into ``{KEY: value}``.

    Tab-delimited ``KEY\\tvalue`` lines up to (not including) ``#ITS_start``.
    Verified identical across all 31 real archives: every file has exactly
    the section order ``#16S_start, #16S_end, #ITS_start, #ITS_end``, the ITS
    section is always empty (``PROCESSED\\tN/A`` and nothing else -- this
    project has no ITS data), and no key genuinely repeats within
    the retained (pre-ITS) portion of any file. ``PROCESSED`` itself repeats
    across the two sections with different values (``True`` then ``N/A``),
    which is why parsing stops at ``#ITS_start`` rather than reading the
    whole file into one flat dict.

    Raises ``ValueError`` naming the duplicate key and path if the same key
    nonetheless repeats with a different value before ``#ITS_start`` -- that
    would mean this file's structure disagrees with what every other archive
    showed, and must not be silently resolved by "last write wins".
    """
    if not path.exists():
        raise DatasetFileError(f"summary_file.txt not found: {path}")

    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#ITS_start"):
            break
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, value = line.partition("\t")
        if not sep:
            raise ValueError(f"{path}: line has no tab separator: {line!r}")
        if key in result and result[key] != value:
            raise ValueError(
                f"{path}: key {key!r} appears twice with different values "
                f"({result[key]!r} and {value!r})"
            )
        result[key] = value
    return result


def otu_table_path(dataset_id: str, study_folder: Path) -> Path:
    """Resolve the RDP-assigned OTU table path for one study via its summary file.

    Reads ``OTU_TABLE_RDP`` from ``study_folder/summary_file.txt`` rather than
    building the filename from ``dataset_id`` -- study archives are not
    uniform (e.g. cdi_schubert also ships a ``.dbOTU`` variant that must not
    be picked up by accident). The RDP-assigned table already has the
    Methods' confidence >= 0.5 threshold applied (empty ``g__`` exactly where
    confidence < 0.5, verified on cdi_schubert's full 19,314 OTUs), so no
    re-derivation from RDP_classifications is needed.

    Raises ``DatasetFileError`` naming the dataset if summary_file.txt has no
    ``OTU_TABLE_RDP`` key, or if the file it names does not exist.
    """
    summary = load_summary_file(study_folder / "summary_file.txt")
    if "OTU_TABLE_RDP" not in summary:
        raise DatasetFileError(
            f"{dataset_id}: summary_file.txt has no OTU_TABLE_RDP key "
            f"({study_folder / 'summary_file.txt'})"
        )
    path = study_folder / "RDP" / summary["OTU_TABLE_RDP"]
    if not path.exists():
        raise DatasetFileError(
            f"{dataset_id}: OTU_TABLE_RDP names {path}, which does not exist"
        )
    return path


def metadata_path(dataset_id: str, study_folder: Path) -> Path:
    """Resolve a study's metadata file: the single ``*.metadata.txt`` in its folder.

    Not built from ``dataset_id`` by string formatting, in case a study's
    metadata filename prefix ever diverges from its dataset ID (not yet
    observed, but the OTU table filename convention already has one
    documented exception, so this is resolved the same defensive way).

    Excludes macOS AppleDouble sidecar files (basename starting with ``._``):
    three archives (cdi_youngster, hiv_lozupone, hiv_noguerajulian) were
    packaged on a Mac and ship a ``._<name>.metadata.txt`` resource-fork stub
    alongside the real file, matching the same glob. ``scripts/download.py``
    already filters these at extraction time; this filter is defense in depth
    for any other extraction path.

    Raises ``DatasetFileError`` naming the dataset if zero or more than one
    match is found.
    """
    matches = sorted(
        p for p in study_folder.glob("*.metadata.txt") if not p.name.startswith("._")
    )
    if len(matches) != 1:
        raise DatasetFileError(
            f"{dataset_id}: expected exactly one *.metadata.txt in "
            f"{study_folder}, found {len(matches)}: {[m.name for m in matches]}"
        )
    return matches[0]
