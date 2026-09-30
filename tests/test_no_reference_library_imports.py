"""mbhd must never import scipy, statsmodels, or pycombat.

They are dev-only dependencies (pyproject.toml's [dependency-groups] dev),
used solely in test files to prove a from-scratch mbhd implementation agrees
with an established reference: scipy/statsmodels for mbhd.stats
(tests/test_stats.py), pycombat for mbhd.batch (tests/test_batch.py) --
never imported by mbhd itself. This is checked with a subprocess rather than
by inspecting sys.modules in-process, because pytest's own collection may
have already imported one of these (e.g. while collecting test_stats.py or
test_batch.py) before this test runs, which would make an in-process check
meaningless regardless of what mbhd actually imports.

mbhd.classifier/mbhd.compositional are deliberately not covered here: both
have a real, legitimate runtime dependency on scikit-learn, which this
check has nothing to say about -- only the from-scratch-verified modules
that must never depend on their own dev-only oracle at runtime.
"""

from __future__ import annotations

import subprocess
import sys


def test_mbhd_does_not_import_scipy_or_statsmodels() -> None:
    script = (
        "import sys\n"
        "import mbhd.abundance, mbhd.checksums, mbhd.datasets, mbhd.differential\n"
        "import mbhd.download, mbhd.io, mbhd.labels, mbhd.manifest, mbhd.stats\n"
        "assert 'scipy' not in sys.modules, 'mbhd imported scipy'\n"
        "assert 'statsmodels' not in sys.modules, 'mbhd imported statsmodels'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_mbhd_batch_does_not_import_pycombat() -> None:
    script = (
        "import sys\n"
        "import mbhd.batch\n"
        "assert 'pycombat' not in sys.modules, 'mbhd.batch imported pycombat'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
