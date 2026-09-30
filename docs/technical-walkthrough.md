# Technical walkthrough

What this project does, how the code is organized, and why each design
decision was made — for a reader with no prior context on this repo. Every
number below is drawn from `docs/reproduction-m2.md` through `-m5.md` or
from this project's own runs;
nothing here is recomputed
or invented for this document. If you want the two-minute version instead,
read [`overview.md`](overview.md).

## 1. The question and the source paper

Duvallet, Gibbons, Gurry, Irizarry & Alm (2017), *Nature Communications*
(`10.1038/s41467-017-01973-8`), re-processed 28 published case-control gut
16S studies (ten diseases) through one standardized pipeline and asked: are
the microbiome shifts reported for each disease actually disease-specific,
or is much of what gets reported really a shared, non-specific response to
being sick? They found the latter — many "disease-associated" genera are
shared across unrelated diseases.

This project reproduces that analysis **independently, from the raw OTU
count tables**, using none of the authors' own analysis code as a starting
point (only their published *data* files, used as ground truth to check
against). It then asks one question of its own: does the genus signature
the paper identifies as "shared response" carry any signal that transfers
to a study it has never seen?

## 2. Data

**Source:** MicrobiomeHD, Zenodo record `10.5281/zenodo.840333` (31 study
archives, ~143 MB compressed / ~850 MB extracted, CC BY-NC 4.0), plus five
supplementary data files the authors published separately on GitHub
(`cduvallet/microbiomeHD`, no license — treated as all-rights-reserved).
Neither source is committed to this repo; `data/` is gitignored.
`scripts/download.py` fetches both, verifies every file's checksum before
trusting it, and extracts only the ~3 files per study this project actually
reads (metadata, the RDP-assigned OTU table, `summary_file.txt` — not the
raw FASTA, which is ~20x the size of what's needed).

**Counts that must reconcile** (the arithmetic connecting "what MicrobiomeHD
contains" to "what the paper analysed"):

```
34 yaml keys = 33 active + hiv_dubourg (commented out)
33 active -> 31 archives   (two dataset-ID pairs share one physical folder)
33 active -> 30 analysed   (3 datasets present but never carried into the paper)
30 analysed = 30 Table 1 rows = 28 published studies (2 studies contribute 2 rows each)
```

This project's own 30-row Table 1 fixture (`tests/data/table1_sample_counts.csv`)
was hand-verified by Pablo against the published paper's HTML
(PMC5716994), not re-fetched by a tool — that fixture is the ground truth
the whole reproduction is checked against.

## 3. Pipeline overview

```
download_all  (Zenodo + GitHub, checksum-verified)
      |
      v
differential_abundance[dataset_id]   x30, one Kruskal-Wallis+BH run per dataset
      |
      v
shared_response   (this project's own q-values -> the paper's "non-specific
      |             genera" labelling rule -> diffed against both published
      |             versions of that result)
      v
figures   (four in-scope figure equivalents, generated from this project's
           own numbers, not pixel copies of the paper's)
```

One command, `uv run snakemake --cores all`, runs the whole thing; `Snakefile`
is a thin dependency graph over the same scripts described below — no logic
lives in the Snakefile itself. `data/` and `results/` are both gitignored
and fully regenerable from a fresh clone plus network access.

**Outside the DAG, deliberately:** `scripts/classifier.py`,
`scripts/clr_shared_response.py`, `scripts/clr_tie_investigation.py`. These
are the extension analyses (below) — there is no published number to check
them against, so gating the reproducible, network-dependent full pipeline
run on them would add runtime without adding verification value. They are
run manually and their results are written up by hand into
`docs/reproduction-m4.md` / `-m5.md`.

## 4. `src/mbhd/` — the logic, file by file

Every module here is imported, never copy-pasted between scripts; `scripts/`
only orchestrates. Every module has a docstring stating its array
orientation (which axis is samples, which is genes/genera) explicitly and
never mixes conventions — this matters constantly in this codebase because
almost every bug found during the project (see §6) was an orientation or
sample-selection mistake, not an arithmetic one.

### `io.py` — loading OTU tables and metadata

Two loaders: `load_otu_table` (OTUs x samples, tab-separated, values are
integer read counts) and `load_metadata` (samples x fields, requires a
`DiseaseState` column). Both raise a specific, named exception
(`EmptyFileError`, `DuplicateSampleIdError`, `MissingColumnError`) rather
than letting pandas fail with a generic error deep inside a stack trace —
this is a recurring pattern across the whole codebase: catch the failure
mode at the boundary where it's cheap to name, not three functions later.

Three non-obvious decisions, each justified in the docstring with real
evidence, not by convention: (1) `dtype={0: str}` on the metadata index —
without it, one study's all-numeric sample IDs (`crc_baxter`) silently
fail to match the OTU table's string-typed columns and 0 of 490 samples
intersect, with no exception raised; (2) `keep_default_na=False` — pandas'
default NA list includes the literal string `"NA"`, which is a real
`DiseaseState` value in one dataset, and letting it parse as `float('nan')`
makes those samples silently disappear from a later exclusion report; (3)
a per-dataset encoding override table (`METADATA_ENCODINGS`) rather than a
try-UTF-8-then-fall-back-to-latin-1 pattern — latin-1 never raises on any
byte sequence, so a blind fallback would quietly mis-decode a genuinely
corrupt file instead of raising.

*Tested by:* `tests/test_io.py`.

### `manifest.py` / `checksums.py` / `download.py` — fetching data

`manifest.py` parses the committed Zenodo API response
(`docs/zenodo_840333_manifest.json`) into `{filename: ManifestEntry}`.
`checksums.py` is a small streaming-hash verifier (fixed-size chunks, so
a multi-hundred-MB archive is never held fully in memory). `download.py`
combines the Zenodo manifest with a small hardcoded table of the five
GitHub-hosted supplementary files (which carry no published checksum, so
their expected hashes are self-generated at first fetch and re-verified on
every later run — documented as a different provenance guarantee than the
Zenodo files, not conflated with it).

`fetch()` writes to a `.part` sibling file and renames only on success, so
an interrupted download can never be mistaken for a complete one — combined
with `is_verified()` (checksum-clean file already present -> skip), this
makes the whole download idempotent and resumable: re-running the script
after a network failure only re-fetches what's missing or corrupt.
`extract_needed()` uses tarfile's `"data"` extraction filter, which rejects
absolute paths and path traversal — relevant because these are archives
from an external source this project doesn't control.

*Tested by:* `tests/test_manifest.py`, `tests/test_checksums.py`,
`tests/test_download.py`.

### `datasets.py` — resolving paths inside one study's folder

Parses `dataset_info.yaml` and resolves, for one dataset ID, its extracted
folder and the OTU-table/metadata file paths inside it — via the file's
own declared name (`summary_file.txt`'s `OTU_TABLE_RDP` key), never by
string-formatting a filename from the dataset ID, because study archives
are not uniform (one study ships two OTU-table variants; the wrong guess
would silently pick the wrong one).

*Tested by:* `tests/test_datasets.py`.

### `labels.py` — case/control group definitions (M1)

The largest and most heavily-annotated module in the project, because this
is where "does our sample count match Table 1" lives, and getting it wrong
is invisible without a published number to check against. `CASE_CONTROL`
is a **transcription** of Table 1's own published labels, per dataset —
not a heuristic. `LABEL_ALIASES` handles the handful of cases where Table
1's label text doesn't match the data's literal string (e.g. Table 1 says
"non-IBD", the data says `nonIBD`) — each alias was accepted only after
confirming it produces an exact count match, never guessed ahead of that
check.

The sample-counting rule itself (`reconcile_counts`) is the project's most
load-bearing discovery: a sample counts only if its ID is in **both** the
metadata and the OTU table — not
`dataset_info.yaml`'s own `sample_size` field, which is the source
publication's self-reported number and doesn't match Table 1 for every
dataset. Three additional per-dataset filters were found this way, each
because a specific dataset's count didn't reconcile and the mismatch was
chased down against real data rather than shrugged off:

- `SUBJECT_ID_COLUMN` — one longitudinal FMT study counts distinct
  **patients**, not samples (repeat sampling per patient).
- `SAMPLE_TYPE_FILTER` — two studies include non-stool (biopsy/tissue)
  samples the paper's "stool only" criterion excludes.
- `CONDITION_FILTER` — four studies needed MicrobiomeHD's own upstream
  `condition:` field applied, a field nothing in this project read before.

**5 of 30 datasets still don't reconcile exactly** even after all of the
above. `KNOWN_MISMATCHES` pins each one's currently-observed count so a
*regression* (the number changing) fails the test suite, while the
pre-existing gap itself does not — per this project's working rule, "I
could not verify this" is an acceptable answer; inventing a filter to force
a match is not.

*Tested by:* `tests/test_labels.py`, `tests/test_labels_verification.py`,
`tests/test_labels_reconciliation_verification.py`,
`tests/test_known_mismatches_verification.py`, `tests/test_table1_counts.py`.

### `abundance.py` — study-level filtering and genus collapse (M2)

Applies the paper's Methods filters (drop samples under 100 reads, OTUs
under 10 reads or present in under 1% of samples) and collapses OTU-level
counts to genus-level relative abundance. Two decisions here were each
**verified against real data three separate ways** before being adopted,
because the "obvious" alternative reproduces the wrong genus count and the
gap is otherwise invisible:

1. The filtering *scope* is the whole study's sample set, not just the two
   groups in the current comparison — proven using three different
   datasets, each ruling out a narrower candidate rule (documented in full
   in the module's own docstring).
2. The *order of operations* in `to_genus_abundance` — relative abundance
   is computed over **all** surviving OTUs first, genus-unannotated OTUs
   are dropped **second**, and only then are the remainder collapsed to
   genus. Reversing steps 1 and 2 (the more "obvious" order) reproduces the
   published effect sizes off by orders of magnitude (median absolute
   difference 0.19, 0/92 comparisons within 1e-6); the order actually used
   reproduces all 92 finite published comparisons to within 1e-12.

*Tested by:* `tests/test_abundance.py`, `tests/test_abundance_verification.py`.

### `stats.py` — from-scratch Kruskal-Wallis and Benjamini-Hochberg

Deliberately does not import scipy or statsmodels — `tests/test_stats.py`
uses them only to *prove* this module agrees with them (to 1e-12), and
`tests/test_no_reference_library_imports.py` enforces at import-scan level
that `mbhd` itself never reaches for a reference stats library. Two
correctness fixes beyond a textbook implementation: (1) `kruskal_wallis`
clamps its statistic to be non-negative — on heavily-tied real data
(zero-inflated relative abundances), floating-point cancellation produces
a tiny negative `H`, which `scipy.stats.kruskal` does not guard against and
returns `nan` for, verified directly against real `par_scheperjans` data
during planning; a single `nan` p-value silently corrupts every other
dataset's Benjamini-Hochberg q-value once it enters the shared sort order.
(2) `benjamini_hochberg` raises loudly on any `nan` input rather than
letting it propagate, for the same reason.

*Tested by:* `tests/test_stats.py`, `tests/test_stats_verification.py`.

### `differential.py` — orchestrates one dataset's comparison (M2)

`differential_abundance()` wires `datasets` -> `io` -> `labels` ->
`abundance` -> `stats` into the one function this project's M2/M3 results
come from: per-genus Kruskal-Wallis p-value, Benjamini-Hochberg q-value,
the paper's signed-q convention (negative = higher in controls), group
means, log2 fold-change, and both groups' surviving sample counts (so
class imbalance is visible in every output row, not buried).
`EXPECTED_DATASET_ERRORS` is a shared tuple of exception types that mean
"this dataset's real data doesn't support a comparison" (e.g. a dataset
where every sample has zero reads after filtering) — used by every caller
(`scripts/reproduce.py`, `scripts/shared_response.py`, `Snakefile`) so a
documented, expected per-dataset failure is recorded rather than crashing
the whole run, while any *other* exception still propagates and stops
everything (never fail silently).

*Tested by:* `tests/test_differential.py`, `tests/test_differential_m2_verification.py`.

### `shared_response.py` — the paper's headline finding, from any q-value source (M3)

Implements the "non-specific genera" labelling rule recovered from the
authors' own supplementary-file README plus reverse-engineering four
further undocumented clauses by testing candidate rules against the
published files until one reproduced both exactly, 0 disagreements — the
full rule (signed significance, exact-zero exclusion, cross-disease
direction agreement, disease-prefix mapping, exclusion sets) is
described here and in the `mbhd.shared_response` docstrings.
`shared_response_labels()` is source-agnostic: it
runs identically over the published `file-S1.qvalues.txt` (used to
*recover and verify* the rule) or over `build_qvalue_matrix()`'s own
reproduced q-values (used to *apply* the same rule independently) — the
same function, not two implementations that could silently drift apart.

*Tested by:* `tests/test_shared_response.py`,
`tests/test_shared_response_verification.py`,
`tests/test_shared_response_m3_verification.py`,
`tests/test_supplementary_files.py`, `tests/test_supplementary_s5_s2_verification.py`.

### `compositional.py` — CLR transform (M4 extension)

Microbiome relative abundances are compositional (they sum to a fixed
total per sample), not Euclidean — a method that assumes Euclidean
geometry needs to say so and name its transform, per this project's own
working rules. `clr()` implements the centered log-ratio transform;
`multiplicative_replacement()` is the required zero-handling step
(log(0) is undefined) — a documented, deliberate choice (multiplicative
shrinkage of each sample's nonzero values so the column's own total is
preserved, one `delta` computed once from the whole table's smallest
nonzero value) over the alternative of a flat additive pseudocount, which
would be scale-blind across datasets that already vary by orders of
magnitude. `mbhd.stats` itself is completely unchanged by this module —
CLR only changes which matrix feeds the same Kruskal-Wallis test.

*Tested by:* `tests/test_compositional.py`.

### `classifier.py` — leave-one-folder-out shared-response classifier (M4)

The extension's central question: does the shared-response genus signature
generalize to an unseen study? `pooled_feature_matrix()` builds one matrix
across every dataset, **grouped by physical folder, not dataset ID** — two
dataset-ID pairs share a folder and can share literal control samples;
grouping by dataset ID would leak those samples across a fold's train/test
boundary. `leave_one_group_out_auc_dynamic_features()` is the leak-free
variant of leave-one-group-out cross-validation: each fold's feature
*column list* (which genera count as "shared response") is recomputed from
only that fold's training data, closing a feature-selection leakage path
found during this milestone's own review (§6). `permutation_null_auc*`
builds an empirical null by shuffling labels *within* each folder (never
across folders, which would leak folder identity into the label) — the
real result is judged against this null distribution, since there's no
published ground truth for this question.

`SEED = 42` lives here, in exactly one place, and is threaded explicitly
into every stochastic call (`LogisticRegression`'s `random_state`, the
permutation generator) — never a bare global `np.random.seed()` anywhere
in the project.

*Tested by:* `tests/test_classifier.py`, `tests/test_classifier_verification.py`,
`tests/test_classifier_m4m5_verification.py`.

### `batch.py` — parametric ComBat, from scratch (M5 extension)

M4's own named biggest weakness — pooling 27 independently-sequenced
studies with no correction for study-to-study technical batch effects — is
addressed by implementing ComBat (Johnson, Li & Rankin 2007) from scratch:
per-gene OLS regression on a batch design (plus an optional covariate)
gives a grand mean and pooled residual scale; each batch's own empirical
Bayes-shrunk location/scale adjustment is fit from **that batch's own
genes only** (never pooled across batches) — this is precisely why the
empirical-Bayes step doesn't leak information between batches, while the
upstream OLS fit *would*, which is why `combat()` must only ever see
training-fold rows when used inside a cross-validation loop.
`combat_for_fold()` is the `FoldTransform` hook `classifier.py` calls: it
fits on the training fold, then corrects the held-out fold using
`combat_new_batch()`, which standardizes with the *training* alpha/sigma
only and never sees the held-out fold's own labels — this is what makes it
safe to call inside a LOFO loop without leaking the test fold's identity
into its own correction.

One real-data condition handled explicitly, not glossed over: a genus
missing entirely from one study is filled `0.0` for every sample in that
study (upstream convention), which affects 27.9% of all (folder, genus)
cells — for such a cell, that batch's scale is left unadjusted rather than
computing a degenerate inverse-gamma prior fit from a column of literal
zeros.

*Tested by:* `tests/test_batch.py`, `tests/test_batch_m5_row5_leak_decomposition_verification.py`.

## 5. `scripts/` — entry points

Each script is a thin CLI wrapper: argument parsing, calling into `mbhd.*`,
writing CSV output. Every script that produces a numeric result also
writes a `provenance.json` recording the exact package versions and
timestamp of that run (per this project's provenance rule — versions
belong in the output, not only the lockfile) — `reproduce.py`,
`shared_response.py`, `classifier.py`, `clr_shared_response.py`, and
`clr_tie_investigation.py` all do; `download.py` and `figures.py` don't
(there's no numeric result to attach provenance to — `figures.py` instead
writes `captions.txt`).

- **`download.py`** — fetches everything (§4). Retryable/idempotent; a
  single failed file doesn't abort the whole run, but the script still
  exits nonzero if anything remains unfetched.
- **`reproduce.py`** — runs `differential_abundance` for one or more
  datasets and diffs the result against the published `file-S1`/`file-S5`.
  Dataset subset is a CLI argument, not hardcoded, so M2's 4-dataset subset
  and M3's all-30 run are the same script (`--all`).
- **`shared_response.py`** — runs `build_qvalue_matrix` +
  `shared_response_labels` and diffs against both published versions of
  the "non-specific genera" file, across two dataset tiers (all 30
  attempted vs. only the 27 that reproduce their genus set exactly) — this
  two-run design localizes any disagreement to either "genuinely new
  labelling disagreement" or "inherited from an already-known sample-count
  gap".
- **`figures.py`** — the four in-scope figure equivalents (per-dataset
  significant-genus counts, the signed-significance heatmap, per-dataset
  shared-response overlap, and shared-vs-specific abundance/ubiquity) —
  each saved with axis labels and units and printed with a one-line
  caption stating a concrete, checkable conclusion. Two of the paper's
  original panels (a classifier AUC panel, a phylogenetic track) were
  descoped: the first because building a classifier honestly is M4's own
  content, not a figure-script afterthought; the second because no
  phylogeny is ever built anywhere in this project.
- **`classifier.py`** (M4/M5, standalone) — the LOFO classifier run
  described in §4, with `--features clr` and `--batch-correct` flags
  layering the M4 and M5 extensions on top of the same core LOFO/null
  machinery.
- **`clr_shared_response.py`** (M4, standalone) — does the CLR transform
  change which genera land in the shared-response pool, relative to the
  Euclidean result already committed as this project's M3 headline?
- **`clr_tie_investigation.py`** (M4, standalone) — a one-off diagnostic
  testing a specific mechanistic hypothesis for *why* CLR changes the pool
  (does it make Kruskal-Wallis's tie correction less conservative?) —
  self-checks every recomputed statistic against `mbhd.stats.kruskal_wallis`'s
  own real output before trusting its own duplicated formula.

## 6. Infrastructure

- **`Snakefile`** — the DAG described in §3. The 30 dataset IDs are read
  from `mbhd.labels.CASE_CONTROL`, a static committed mapping, not from
  the downloaded `dataset_info.yaml` — this lets the whole DAG's structure
  be dry-run-validated (`snakemake -n`) with no data downloaded and no
  network access, which is exactly what CI does on every push.
- **`Dockerfile`** — multi-stage build, Python and `uv` both pinned to
  exact versions (including a manifest-list digest, not a floating tag).
  Two stacked bugs were found and fixed by an actual `docker run` against
  empty volumes during a later review, not by `docker build` succeeding
  alone: the runtime image originally had neither the `uv` binary itself
  (only the venv it built) nor the pinned Zenodo manifest file the
  download step needs at runtime. Both are now documented inline at the
  exact `COPY` line that fixes them.
- **`.github/workflows/ci.yml`** — `uv sync --locked` (fails loudly on any
  lockfile drift), `ruff check` + `ruff format --check`, `pytest`, and a
  structural-only `snakemake -n --forceall` dry run. The real, network-
  dependent end-to-end pipeline run is deliberately *not* run on every
  push — Zenodo's documented intermittent 504s would make every PR flaky
  through no fault of the code — so it's a separate, manually-triggered
  step instead.
- **`pyproject.toml` / `uv.lock`** — runtime dependencies (`pandas`,
  `numpy` via pandas, `pyyaml`, `scikit-learn`, `matplotlib`, `snakemake`)
  vs. dev-only dependencies (`scipy`, `statsmodels`, `pycombat`, `pytest`,
  `ruff`) are kept strictly separate — the dev group exists *only* to
  verify `mbhd`'s own from-scratch implementations against reference
  libraries, and none of them is importable by `mbhd` itself
  (`tests/test_no_reference_library_imports.py`).

## 7. Testing strategy

The test suite (`uv run pytest -q`) mixes four kinds of test, each doing a
different job:

1. **Unit tests** (`test_io.py`, `test_labels.py`, `test_abundance.py`,
   `test_stats.py`, `test_compositional.py`, `test_batch.py`,
   `test_classifier.py`, `test_datasets.py`, `test_manifest.py`,
   `test_checksums.py`, `test_download.py`, `test_shared_response.py`) —
   mostly synthetic fixtures checking one module's contract in isolation
   (including every documented edge case: duplicate sample IDs, wrong
   encodings, degenerate/all-tied data, a single batch, zero-variance
   genes); `test_labels.py`, `test_abundance.py`, `test_stats.py`,
   `test_compositional.py`, `test_batch.py`, and `test_checksums.py` are
   fully synthetic with no data or network dependency, while the rest add
   a handful of `pytest.skip`-guarded tests that also check against real
   downloaded data when it's present.
2. **Known-answer tests against published values**
   (`test_differential_m2_verification.py`,
   `test_shared_response_m3_verification.py`,
   `test_table1_counts.py`, `test_supplementary_files.py`,
   `test_supplementary_s5_s2_verification.py`) — the actual reproduction
   claims: does `differential_abundance` reproduce the paper's own
   q-values for the required subset, does the labelling rule reproduce
   both published "non-specific genera" files exactly, does the sample
   count match Table 1.
3. **`*_verification.py` files** — an independent second pass, usually
   written by the `verifier` subagent re-deriving a result a different way
   (e.g. reading raw files with plain pandas instead of `mbhd.io`) to
   catch a bug the original implementation and its own unit tests might
   share. Several real bugs were caught exactly this way (§8).
4. **`test_no_reference_library_imports.py`** — imports a fixed list of
   `mbhd` modules (the ones whose from-scratch statistics/batch-correction
   code scipy/statsmodels/pycombat verify — not every module in the
   package, e.g. `mbhd.classifier` and `mbhd.compositional` are
   deliberately excluded, since they have a legitimate scikit-learn
   dependency) in a subprocess and inspects `sys.modules` afterward,
   enforcing that scipy, statsmodels, and pycombat never got pulled in as
   a side effect — dev-only oracles, never runtime dependencies of the
   code they verify.

## 8. Milestones, in order

- **M1 — download, checksums, Table 1 reconciliation.** 25 of 30 datasets
  reconcile Table 1's published sample counts exactly;
  `mbhd.labels.KNOWN_MISMATCHES` pins the 5 that don't, none forced to
  match.
- **M2 — per-genus differential abundance from scratch.** Exact
  reproduction (0 missing/extra genera, 100% significance and sign
  agreement, q-values agreeing to ~1e-13) on the required 4-dataset
  subset against the published `file-S1`/`file-S5` values
  (`docs/reproduction-m2.md`).
- **M3 — full 30-dataset run, the shared-response finding.** 138 of 142
  lineages both this project's reproduction and the published
  "non-specific genera" file have an opinion on agree — **97.2%** on the
  primary comparison target — closing the loop from raw OTU tables to the
  paper's own headline finding, independently, for the first time in this
  project (`docs/reproduction-m3.md`).
- **M4 — two extension questions.** (A) Does a CLR (compositional) re-
  analysis change the shared-response conclusion? Yes, substantially: the
  CLR pool is 1.6–1.7x larger and 24–35% of common lineages get a
  different label — a real, quantified disagreement between two
  statistically valid analyses of the same counts, not a bug. (B) Does the
  shared-response signature transfer to an unseen study? A leave-one-
  folder-out classifier scores mean AUC-ROC 0.603 (Euclidean baseline,
  leak-free — ~7.0 permutation-null standard deviations above chance),
  rising to 0.635 with CLR-transformed features (~9.1 null standard
  deviations above chance) — modest but real (`docs/reproduction-m4.md`).
- **M5 — batch/study-effect correction.** M4's own named biggest weakness
  (no correction for pooling 27 independently-sequenced studies) is
  addressed with a from-scratch ComBat implementation. Adopted headline:
  **mean LOFO AUC-ROC 0.635 -> 0.647** (leak-free, no covariate — the
  arm named as primary *before* the run, per the adoption rule fixed before the run), 19
  of 27 folds individually improved, only 7 worse. A naive version with a
  label leak was also deliberately measured as a warning: +0.15 AUC-ROC
  inflation, the largest leakage effect found anywhere in this project
  (`docs/reproduction-m5.md`).

## 9. Bugs found along the way

Documented here because how they were found is part of the engineering
story, not just the fix:

- **`SAMPLE_TYPE_FILTER` / `CONDITION_FILTER`** — two rounds of "this
  dataset's count doesn't match Table 1" investigations, each resolved by
  finding a real, previously-unapplied filtering rule (non-stool samples;
  MicrobiomeHD's own upstream `condition:` field) rather than forcing a
  match — shrank `KNOWN_MISMATCHES` from 11 -> 9 -> 5.
- **The blank-`DiseaseState`-sample exclusion bug (M2)** — the first
  implementation excluded unlabelled samples from the study-level
  filtering scope, a reasonable-looking assumption that turned out wrong:
  caught only when the all-30 appendix run showed `crc_zeller` (one of
  M1's *exactly* reconciled datasets) missing 7 of its 193 published
  genera.
- **Feature-selection leakage in the M4 classifier** — the original design
  chose the shared-response feature list once, globally, from all 29
  datasets before the leave-one-folder-out loop began; both `verifier` and
  `code-reviewer` independently flagged this as a leakage path. Fixed by
  recomputing each fold's feature list from training data only; the
  measured leakage effect turned out to be negligible (0.607 vs. 0.603)
  but was fixed rather than only documented.
- **A label leak in M5's batch correction, deliberately measured as a
  warning, not shipped** — fitting ComBat once globally with the
  case/control covariate, including the held-out fold's own labels,
  inflates AUC-ROC by +0.15 — by far the largest leakage effect measured
  in this project. The leak-free hook signature makes this specific
  mistake structurally impossible to reproduce by accident in the shipped
  code.
- **Two stacked Docker bugs**, found only by an actual `docker run`
  against empty volumes (not just `docker build` succeeding): the runtime
  image had neither the `uv` binary nor the pinned Zenodo manifest file.
- **`scripts/download.py` never fetched the GitHub supplementary files**
  — found during a full-project completion review simulating a fresh
  clone; the download script previously covered only the Zenodo archive.

## 10. Limitations, stated plainly

- No claim of a novel biological finding — cross-study microbiome
  prediction has been surveyed elsewhere. This project demonstrates
  reproduction and method competence, stated as such.
- 5 of 30 datasets still don't reconcile Table 1 exactly
  (`mbhd.labels.KNOWN_MISMATCHES`) — documented as "could not verify," not
  forced to match.
- The paper's abstract says "ten diseases"; this project's prefix mapping
  yields 11 groups (NASH/LIV both being liver-related conditions,
  unverified) — the 11-group mapping is what reproduces both published
  "non-specific genera" files exactly, so it's what the analysis uses; the
  discrepancy is noted, not resolved, and doesn't affect reproduction.
- M4/M5's extension results have no published ground truth to check
  against by construction — they are judged against an empirical
  permutation null instead, which is a weaker verification standard than
  M1–M3's exact-value reproduction, and is stated as such throughout
  `docs/reproduction-m4.md`/`-m5.md`.

## 11. How to reproduce every number in this document

```bash
uv sync
uv run scripts/download.py
uv run pytest -q
uv run snakemake --cores all        # M1-M3
uv run scripts/classifier.py --features clr --batch-correct   # M4/M5
```

Full command reference: [`README.md`](../README.md).
