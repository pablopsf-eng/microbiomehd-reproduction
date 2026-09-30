# Project overview

**The question.** Studies linking specific gut bacteria to specific
diseases often don't replicate across studies. In 2017, Duvallet et al.
re-processed 28 published case-control gut microbiome studies (ten
diseases) through one standardized pipeline and found that a lot of what
gets reported as "disease X is linked to bacterium Y" is actually a
generic, shared response to being sick — not disease-specific at all.

**What I did.** I reproduced that analysis independently, from the raw
sequencing-derived count tables, without starting from the authors' own
analysis code — I only used their published result files afterward, as
ground truth to check my own numbers against. Then I asked one extension
question of my own: if a set of genera really is a shared "sick response"
signature, can a classifier trained on it predict case-vs-control status
in a study it has never seen?

**How.** Python, built from scratch where it mattered for the reproduction
to mean something: I implemented Kruskal-Wallis testing and
Benjamini-Hochberg multiple-testing correction myself (not `scipy`),
verified against `scipy`/`statsmodels` in tests but never imported by the
real code path. Same for the batch-effect correction method (ComBat) used
in the extension. The full pipeline — download, statistics, the paper's
own "shared response" labelling rule, figures — runs end-to-end with one
command via Snakemake, in Docker, with CI running the test suite and a
structural check on every push. 30 datasets, 28 studies, ~850 MB of real
data, none of it committed (fetched and checksum-verified on demand).

**What I found.**

- **Reproduction works, with a real, quantified gap, not a false 100%
  match.** 25 of 30 datasets match the paper's published sample counts
  exactly; on the required subset, my independent statistics reproduce the
  paper's own q-values to ~1e-13. Running the full pipeline and applying
  the paper's own labelling rule to my own numbers reproduces **97.2%**
  (138/142) of the paper's published "shared response" gene list.
- **The reproduction gap is itself informative.** Every dataset that
  doesn't reconcile is a real, named data issue I chased down against the
  raw files — not swept under a rug. Some I resolved (a longitudinal study
  needed per-patient counting instead of per-sample; two datasets had
  non-stool samples the paper's own criteria should have excluded; four
  needed an upstream filter field nothing had read before). Five remain
  open, documented as "could not verify," not forced to match.
- **My extension question has a real but modest answer.** A classifier
  trained on the shared-response genus signature, tested on a study it
  never saw during training, scores AUC-ROC 0.60–0.65 depending on how the
  abundance data is transformed and whether study-to-study batch effects
  are corrected for — well above chance (about 7–9 standard deviations
  above an empirical null built from 200 label-shuffle repeats), but not a
  strong discriminator. I also found and fixed a real leakage bug in my
  own first attempt at this classifier, and separately measured — as a
  deliberate warning, never shipped as a result — how badly a *different*
  leak (letting a batch-correction step see held-out labels) would have
  inflated the number if I hadn't guarded against it (+0.15 AUC-ROC,
  the largest such effect found anywhere in the project).

**What this demonstrates.** Independent reproduction of a real published
result from raw data — not just running someone else's code and reporting
a match — plus honest handling of the parts that don't reconcile, plus one
original extension question, verified against an appropriate null rather
than presented as more definitive than it is. No claim of a new biological
discovery is made or implied.

Full technical detail: [`technical-walkthrough.md`](technical-walkthrough.md).
Code: [`../README.md`](../README.md).
