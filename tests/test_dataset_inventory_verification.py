"""Independent re-derivation of the dataset inventory counts, straight from
``data/raw/dataset_info.yaml`` and
``data/raw/file-S1.qvalues.txt`` -- no ``mbhd`` code imported, so this does
not just re-check that ``mbhd.datasets`` agrees with itself.

Ground truth: the raw Zenodo ``dataset_info.yaml`` and the raw GitHub
``file-S1.qvalues.txt``, both already downloaded into ``data/raw`` (never
committed), plus the actual 31
``.tar.gz`` archives on disk as a real-filesystem cross-check on the
"31 archives" claim (not just trusting the yaml's own ``folder:`` field).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def _require(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"data not fetched: {', '.join(missing)}")


def _load_raw_yaml() -> dict:
    with open(DATA_ROOT / "dataset_info.yaml") as f:
        return yaml.safe_load(f)


def test_34_keys_33_active_1_commented():
    _require(DATA_ROOT / "dataset_info.yaml")
    text = (DATA_ROOT / "dataset_info.yaml").read_text()
    parsed = _load_raw_yaml()
    assert len(parsed) == 33  # active, real yaml keys

    # top-level key-shaped lines, active or commented (matches docs section
    # 11's grep, generalised to allow a leading '#')
    import re

    top_level_lines = [
        line for line in text.splitlines() if re.match(r"^#?[a-z0-9_]*:$", line)
    ]
    assert len(top_level_lines) == 34
    commented = [line for line in top_level_lines if line.startswith("#")]
    assert commented == ["#hiv_dubourg:"]


def test_33_active_to_31_archives_real_bijection():
    _require(DATA_ROOT / "dataset_info.yaml")
    parsed = _load_raw_yaml()
    folders_from_yaml = {parsed[k]["folder"] for k in parsed}
    assert len(folders_from_yaml) == 31

    archive_files = os.listdir(DATA_ROOT / "archives")
    archive_keys = {
        f[: -len(".tar.gz")] for f in archive_files if f.endswith(".tar.gz")
    }
    assert len(archive_keys) == 31
    assert folders_from_yaml == archive_keys, (
        f"yaml folders vs real archives on disk differ: "
        f"only-in-yaml={folders_from_yaml - archive_keys}, "
        f"only-on-disk={archive_keys - folders_from_yaml}"
    )

    # exactly two folders serve two dataset IDs each (33 -> 31)
    folder_counts: dict[str, int] = {}
    for k in parsed:
        folder_counts[parsed[k]["folder"]] = (
            folder_counts.get(parsed[k]["folder"], 0) + 1
        )
    shared = {f: n for f, n in folder_counts.items() if n > 1}
    assert shared == {"cdi_schubert_results": 2, "nash_ob_baker_results": 2}


def test_33_active_to_30_analysed_and_30_s1_columns():
    _require(DATA_ROOT / "dataset_info.yaml", DATA_ROOT / "file-S1.qvalues.txt")
    parsed = _load_raw_yaml()
    active_ids = set(parsed.keys())

    with open(DATA_ROOT / "file-S1.qvalues.txt") as f:
        header = f.readline().rstrip("\n").split("\t")
    s1_columns = [c for c in header if c]

    assert len(s1_columns) == 30
    assert set(s1_columns) <= active_ids, "every S1 column must be an active dataset ID"
    unanalysed = active_ids - set(s1_columns)
    assert unanalysed == {"crc_wu", "crc_zackular", "ob_escobar"}


def test_30_analysed_datasets_collapse_to_28_studies():
    """30 analysed IDs share only 28 distinct folders (Schubert and Zhu 2013
    each contribute 2 rows via one shared folder each) -- recomputed
    directly from folder-grouping, not asserted from the doc's prose."""
    _require(DATA_ROOT / "dataset_info.yaml", DATA_ROOT / "file-S1.qvalues.txt")
    parsed = _load_raw_yaml()
    with open(DATA_ROOT / "file-S1.qvalues.txt") as f:
        header = f.readline().rstrip("\n").split("\t")
    analysed_ids = [c for c in header if c]

    folder_to_ids: dict[str, list[str]] = {}
    for dataset_id in analysed_ids:
        folder_to_ids.setdefault(parsed[dataset_id]["folder"], []).append(dataset_id)

    assert len(folder_to_ids) == 28
    multi = {f: ids for f, ids in folder_to_ids.items() if len(ids) > 1}
    assert multi == {
        "cdi_schubert_results": ["cdi_schubert", "noncdi_schubert"],
        "nash_ob_baker_results": ["ob_zhu", "nash_zhu"],
    }
