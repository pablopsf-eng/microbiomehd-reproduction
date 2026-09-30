"""Snakemake workflow: one command reproduces the full MicrobiomeHD pipeline.

Rules: download_all -> differential_abundance[dataset_id] (x30, a
static list) -> shared_response (all 30 dataset attempts, failures recorded
not dropped) -> figures. Every rule is a thin shell/run wrapper around an
existing script in scripts/ -- logic stays in src/mbhd, nothing is
reimplemented here (the project's established "logic lives in src/, scripts
are entry points" convention).

The 30 dataset IDs are read from mbhd.labels.CASE_CONTROL -- a static,
committed Python mapping, not derived from the downloaded dataset_info.yaml
-- so the full DAG (job count, dependency structure) can be built and
dry-run-validated (`snakemake -n`) even before any data is downloaded, with
no Snakemake `checkpoint` needed for a dataset list that never changes
between runs.

**Network access is required when download_all actually runs, not when
this file is parsed or the image is built** (see Dockerfile). Zenodo has a
documented, real, intermittent 504 gateway time-out; if it
is down, download_all fails with that error directly, not a silent retry
loop that masks it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mbhd.labels import CASE_CONTROL

#: Static list, deliberately not read from dataset_info.yaml (see module
#: docstring) -- the 30 dataset IDs the paper's analysis covers.
DATASET_IDS = sorted(CASE_CONTROL)

DATA_ROOT = "data/raw"
REPRO_DIR = "results/reproduction_all"
SHARED_DIR = "results/shared_response"
FIGURES_DIR = "results/figures"


rule all:
    input:
        f"{SHARED_DIR}/full29_vs_zenodo.csv",
        f"{SHARED_DIR}/full29_vs_github.csv",
        f"{SHARED_DIR}/clean27_vs_zenodo.csv",
        f"{SHARED_DIR}/clean27_vs_github.csv",
        f"{FIGURES_DIR}/captions.txt",


rule download_all:
    """Fetch and checksum-verify the real MicrobiomeHD archive from Zenodo."""
    output:
        f"{DATA_ROOT}/dataset_info.yaml",
    shell:
        "uv run scripts/download.py"


rule differential_abundance:
    """One dataset's per-genus differential abundance, diffed against file-S1/S5.

    A documented, expected per-dataset failure (mbhd.differential.
    EXPECTED_DATASET_ERRORS, e.g. ob_zupancic's ZeroReadSampleError) still
    produces this rule's declared output -- a status file recording the
    failure -- so the DAG's output set is never silently missing a dataset
    without a trace. Any other, undocumented exception fails the rule loudly
    (never fall back silently), surfacing reproduce.py's own
    stderr in the Snakemake log.
    """
    input:
        f"{DATA_ROOT}/dataset_info.yaml",
    output:
        status=f"{REPRO_DIR}/{{dataset_id}}/status.json",
    params:
        dataset_id=lambda wc: wc.dataset_id,
    run:
        out_dir = Path(output.status).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # A stale provenance.json from a previous run at this same path (e.g.
        # a local --forcerun while iterating) must never be read below: if a
        # dataset that used to be a documented failure starts raising a new,
        # undocumented exception, subprocess.run's own crash leaves the old
        # file's "failed" list untouched, which would silently reclassify a
        # real new bug as the old, known failure (found in M3's code review).
        # Removing it first means the check below only ever trusts a
        # provenance.json written by *this* invocation.
        provenance_path = out_dir / "provenance.json"
        provenance_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/reproduce.py",
                "--datasets",
                params.dataset_id,
                "--data-root",
                DATA_ROOT,
                "--out",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        failed_ids = []
        if provenance_path.exists():
            failed_ids = [
                dataset_id
                for dataset_id, _ in json.loads(provenance_path.read_text())["failed"]
            ]
        if result.returncode != 0 and params.dataset_id not in failed_ids:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise RuntimeError(
                f"{params.dataset_id}: scripts/reproduce.py failed with an "
                "undocumented error (not one of mbhd.differential."
                "EXPECTED_DATASET_ERRORS) -- see stderr above, this is a real bug"
            )
        Path(output.status).write_text(
            json.dumps(
                {
                    "dataset_id": params.dataset_id,
                    "succeeded": params.dataset_id not in failed_ids,
                }
            )
            + "\n"
        )


rule shared_response:
    """The shared-response result from this project's own reproduced q-values.

    Depends on every per-dataset status file, not their CSVs directly:
    scripts/shared_response.py recomputes differential_abundance itself
    (mbhd.shared_response.build_qvalue_matrix) rather than re-reading
    differential_abundance's per-dataset rule outputs, so this rule's real
    dependency is "every dataset has been attempted at least once with the
    current data/code", which the status files already certify.
    """
    input:
        expand(f"{REPRO_DIR}/{{dataset_id}}/status.json", dataset_id=DATASET_IDS),
    output:
        f"{SHARED_DIR}/full29_vs_zenodo.csv",
        f"{SHARED_DIR}/full29_vs_github.csv",
        f"{SHARED_DIR}/clean27_vs_zenodo.csv",
        f"{SHARED_DIR}/clean27_vs_github.csv",
    shell:
        f"uv run scripts/shared_response.py --data-root {DATA_ROOT} --out {SHARED_DIR}"


rule figures:
    """The four in-scope figure equivalents (Fig 1b/2/3b/3c)."""
    input:
        f"{SHARED_DIR}/full29_vs_github.csv",
    output:
        f"{FIGURES_DIR}/captions.txt",
    shell:
        f"uv run scripts/figures.py --data-root {DATA_ROOT} --out {FIGURES_DIR}"
