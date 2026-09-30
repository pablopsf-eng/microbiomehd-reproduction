# Multi-stage build: a pinned uv installs pinned, locked dependencies in one
# stage; only the resulting virtualenv and the source tree are copied into
# the final image.
#
# Base image pinned to an exact tag *and* its manifest-list digest -- not
# "latest", not a floating "3.12-slim" -- matching this project's existing
# no-floating-major-tag practice for astral-sh/setup-uv in .github/workflows/
# ci.yml. Digest verified directly against Docker Hub's registry API on
# 2026-09-17 (`docker-content-digest` header for the `3.12.3-slim` tag);
# re-verify if this line is ever bumped to a new Python patch version.
FROM python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63 AS builder

# uv itself pinned to the exact version this project develops against
# (matches pyproject.toml's own build-system requirement).
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# --locked: fail the build if pyproject.toml/uv.lock have drifted, rather
# than silently re-resolving. --no-dev: pytest/ruff/scipy/statsmodels are
# dev-only (used to validate mbhd.stats in tests, never imported by mbhd
# itself -- tests/test_no_reference_library_imports.py enforces this), so
# the runtime image does not need them. README.md must be present here,
# not just for the later --no-install-project-less sync below: pyproject.toml
# declares `readme = "README.md"`, and uv_build refuses to build the mbhd
# package at all without the actual file on disk (caught by code-reviewer
# during M3's review -- verified by reproducing the exact failure with a
# synthetic project before this fix).
RUN uv sync --locked --no-dev --no-install-project

COPY src/ src/
COPY scripts/ scripts/
COPY Snakefile ./
# The pinned Zenodo manifest: scripts/download.py's default --manifest path
# is docs/zenodo_840333_manifest.json (mbhd.manifest.load_manifest reads it
# at run time, not build time) -- the only file under docs/ any runtime code
# actually opens (everything else under docs/ is documentation, referenced
# only in docstrings/comments, never read by a running script). Found the
# same way as the uv-binary gap above: a real `docker run` against empty
# volumes, once uv itself could run, immediately hit
# FileNotFoundError('/app/docs/zenodo_840333_manifest.json') on the very
# first rule -- this file had never been copied into the image at all.
COPY docs/zenodo_840333_manifest.json docs/zenodo_840333_manifest.json
RUN uv sync --locked --no-dev

FROM python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63

WORKDIR /app
COPY --from=builder /app /app
# uv itself, not just the venv it built: every Snakefile rule's `shell:` runs
# `uv run scripts/...`, so uv must exist at container *run* time too, not
# only during the build stage above -- found by a 2026-09-24 review actually
# running `docker run` (every prior verification only checked that `docker
# build` succeeds), which failed on the very first rule with "uv: command
# not found" because uv lives outside /app and this COPY never brought it in.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
ENV PATH="/app/.venv/bin:$PATH"

# Network access is required here, at container *run* time, not at build
# time above: scripts/download.py fetches the real MicrobiomeHD archive from
# Zenodo when the pipeline actually runs. If Zenodo's documented
# intermittent 504 gateway time-out is active when this
# container runs, the run fails with that error directly -- this is a real,
# stated limitation of "one command reproduces everything", not hidden
# behind a silent retry loop.
ENTRYPOINT ["snakemake", "--cores", "all"]
