# SynHydro 0.1.0 Release Checklist

Working document for the first formal, citable release of SynHydro.
Goal: a tagged GitHub release archived on Zenodo with a DOI that can be
cited in publications, and the same version published on PyPI. Version
1.0.0 is reserved for a later JOSS/JORS submission.

Check items off as they are completed. Notes under each item record the
state found during the 2026-09-14 pre-release survey (commit 98751ad).

---

## 0. Decisions

Settled on 2026-09-14:

| Decision | Outcome |
|---|---|
| Version number | `0.1.0`, tag `v0.1.0`. Three components everywhere (PEP 440 treats `0.1` and `0.1.0` as equal, but tags, CHANGELOG, and Zenodo display the literal string). Patch releases `0.1.x` for fixes, `0.2.0` for API changes, `1.0.0` with the JOSS paper. |
| DOI provider | Zenodo via the GitHub release integration (Section 5). |
| Metadata file | `CITATION.cff` only. GitHub renders it as "Cite this repository" and Zenodo reads it. Add `.zenodo.json` only if grants or Zenodo communities are needed; if both exist Zenodo ignores the CFF file. |
| PyPI | Yes. Publish `synhydro` 0.1.0 via Trusted Publishing from GitHub Actions (Section 3.2). The name is unclaimed (checked 2026-09-14). |
| `experiments/` | Not part of the release. Untrack the six tracked files and ignore the whole directory (Section 2.4). |
| AUDIT.md items | All still-open items resolved before the final release (Section 1.7). |
| Python versions | 3.10 to 3.12 only. Do not add 3.13 to the classifiers or the CI matrix (it has broken dependencies in prior experience). Leave `requires-python = ">=3.10"` uncapped; an upper bound makes pip resolution worse and the classifiers plus CI matrix already state what is supported. |
| Development status classifier | Not used. Remove `Development Status :: 3 - Alpha` from `pyproject.toml` (Section 1.4). It is an informational PyPI "Trove" classifier shown in the project sidebar; no tool depends on it. |
| `examples/_dev_*.ipynb` | Untrack both dev notebooks and ignore the pattern (Section 2.4). |

No decisions remain open.

---

## 1. Code and packaging fixes

### 1.1 Example data is not shipped in the wheel (blocker)

- [x] Move `examples/example_data/*.csv` into the package (for example `src/synhydro/data/`) and load them with `importlib.resources`.
- [x] Keep `examples/example_data/` working for the notebooks (copy, or point the retrieval script at the package data).
- [x] Add a test that imports the installed package from a directory outside the repo and calls `load_example_data()`.

Done 2026-09-14: CSVs moved (`git mv`) to `src/synhydro/data/` (a regular package with an `__init__.py`); `synhydro.utils.directories` resolves them with `importlib.resources.files("synhydro.data")` with the public API unchanged; `retrieve_example_data.py` now writes into that directory and no copy is kept under `examples/` (every notebook calls `load_example_data()`); `tests/test_packaging.py` builds the wheel with hatchling, pip-installs it with `--no-deps --target` into a temp directory, and loads both datasets from a subprocess run outside the repo (skips if hatchling or pip is missing); `unzip -l` on the built wheel shows both CSVs.

Notes: `src/synhydro/utils/directories.py` computes
`EXAMPLE_DATA_DIR = PACKAGE_ROOT.parent.parent / "examples" / "example_data"`.
That resolves relative to `site-packages` for a normal install. Verified
by installing the built wheel into a clean target directory:
`load_example_data()` raises `FileNotFoundError`. Every `pip install
git+https://...` user hits this on the README quick example. CI does not
catch it because `tests.yml` uses an editable install. The daily CSV is
2.5 MB and the monthly CSV is 83 KB, which is fine to ship.

### 1.2 sdist contents (blocker for PyPI)

- [x] Add an explicit sdist include list to `pyproject.toml` under `[tool.hatch.build.targets.sdist]`: `src/`, `tests/`, `LICENSE`, `README.md`, `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`.
- [x] Rebuild and confirm the sdist is a few MB at most.

Done 2026-09-14: include list added; the rebuilt sdist is 1.10 MB compressed (4.52 MB uncompressed, 158 entries: `src/`, `tests/`, `LICENSE`, `README.md`, `CHANGELOG.md`, `pyproject.toml`, plus the `.gitignore` and `PKG-INFO` hatchling always adds) with no `cache/`, `docs/`, `examples/`, or `experiments/` entries. `CITATION.cff` is in the list and will be picked up once it exists (Section 2.2).

Notes: `hatchling build` produced a 142 MB sdist because it packed
`cache/aiohttp_cache.sqlite` (301 MB uncompressed). That file is ignored
by `cache/.gitignore`, but hatchling only honors the root `.gitignore`.
The sdist also carried `docs/` (including 1.6 MB of blog PNGs), all
tutorial notebooks, and `experiments/`. The Zenodo archive is unaffected
(it zips the git tag, not the sdist), but PyPI hosts the sdist.

### 1.3 Single-source the version

- [x] Set `dynamic = ["version"]` in `[project]` and add `[tool.hatch.version] path = "src/synhydro/__init__.py"`.
- [x] Remove the stray `__version__ = "2.0.0"` and `__author__ = "SynHydro Development Team"` from `src/synhydro/plotting/__init__.py`.
- [x] Add a test asserting `synhydro.__version__` matches `importlib.metadata.version("synhydro")`.

Done 2026-09-14: the wheel and sdist now build as 0.0.2 from `src/synhydro/__init__.py` alone; `tests/test_packaging.py::TestVersion` checks the attribute against the installed metadata in-process (after bumping the version, re-run `pip install -e .` so the editable metadata catches up) and the wheel test repeats the check inside the installed wheel.

Notes: the version is duplicated in `pyproject.toml` and
`src/synhydro/__init__.py` (both `0.0.2`).

### 1.4 Dependency and license metadata

- [ ] Pin a minimum for `spei` (currently unpinned; venv has 0.8.2).
- [ ] Confirm `pygeohydro` belongs in `dev` only (used by `examples/example_data/retrieve_example_data.py`).
- [ ] Modernize license metadata to PEP 639 (`license = "MIT"`, `license-files = ["LICENSE"]`) and drop the `License ::` classifier. Hatchling 1.27 supports it, and PyPI now warns on the old form.
- [ ] Remove the `Development Status :: 3 - Alpha` classifier (decision: not used).
- [ ] Leave `requires-python = ">=3.10"` and the 3.10, 3.11, 3.12 classifiers as they are. Do not add 3.13 anywhere.

### 1.5 README as the PyPI landing page

- [ ] Make the `CONTRIBUTING.md` link absolute (relative links break on PyPI).
- [ ] Replace the git install command with `pip install synhydro`; keep `pip install git+...@v0.1.0` as the alternative.
- [ ] Add a "Citing SynHydro" section with a DOI badge placeholder (filled in after the first Zenodo record exists, see 7.1).
- [ ] Check the rendered README with `twine check dist/*` after building.

### 1.6 Small code items found in the survey

- [ ] `synhydro/plotting/correlation.py:151` and `synhydro/plotting/timeseries.py:147` use unseeded `np.random.choice` to pick which realizations to draw. Accept an optional `seed` or `rng` argument so figures are reproducible. Low priority.
- [ ] `synhydro/droughts/distributions.py:198-214` uses `print()` in a help-style display function. Either keep it (user-facing output, not logging) or return a string. Decide and move on.
- [ ] Add `synhydro_version` to `EnsembleMetadata` / HDF5 attributes at generation time so archived ensembles record which release produced them. Small change, high value for reproducibility (the Kirsch fix already changed seed-level output relative to earlier code, so the version matters).

### 1.7 AUDIT.md closeout (all items resolved before release)

Source: `AUDIT.md` (2026-08-20 audit plus resolution log). Work these in a
dedicated session; they touch generator code and tests, so keep them
separate from the packaging commits above.

- [x] Statistical regression test for `ARFIMAGenerator`: `TestARFIMAGeneratedOutput` (recovered d from generated output, Hosking ACF, Gaussian-domain variance, pooled moments).
- [x] Statistical regression test for `SMARTAGenerator`: `TestSMARTAStatisticalReproduction` (target ACF, observed ACF consistency, cross-correlation, marginal moments).
- [x] Statistical regression test for `KirschGenerator`: `TestKirschStatisticalReproduction` on the packaged USGS monthly record (per-period mean/std, within-half-year Corr(Y), per-month cross-site correlation).
- [x] Statistical regression test for `WARMGenerator`: spectral-peak, variance, and epoch-timing tests already existed; added `test_refit_on_generated_output_redetects_band`.
- [x] Statistical regression test for `PhaseRandomizationGenerator`: `TestPhaseRandomizationSpectralAndMarginal` (exact amplitude spectrum, flow ACF, kappa L-moment round-trip, day-of-year marginal).
- [x] Statistical regression test for `MultisitePhaseRandomizationGenerator`: `TestMultisitePhaseRandomizationSpearman` (Spearman cross-correlation within 0.10, within-day-of-year dependence control).
- [x] Kirsch `generate_from_residuals`: fixed 2026-09-14. Now requires `(n_years + 1, P, S)` residuals like `generate_from_indices`; padding removed, docstring and CHANGELOG corrected, exact-equality test against `generate_from_indices` added.
- [ ] Low-severity items deferred by maintainer decision (2026-09-14). Verified closed: negative lag-1 clipping, Nataf zero-inflated note, Valencia-Schaake partial-year check, and the rest of the original Low list. Still open and listed in `AUDIT.md` under "Open: Low": WARM significance-test size, two undocumented PRSim deviations in phase randomization, KNN bootstrap module docstring, plus method-limitation doc notes surfaced by the new statistical tests.
- [x] `ThomasFieringGenerator`: reviewed manually by the maintainer; no `AUDIT.md` entry required (decision 2026-09-14).
- [x] AUDIT.md rewritten 2026-09-14 to list only open (Low) items; full suite re-run with the new tests: 1118 tests, all passing (see Section 4 for the final-commit re-run).
- [ ] Anything deliberately deferred goes into CHANGELOG under "Known limitations" so the release notes are honest.

---

## 2. Documentation and release metadata

### 2.1 CHANGELOG

- [ ] Rename `## [Unreleased]` to `## [0.1.0] - YYYY-MM-DD` and add a fresh empty `[Unreleased]` above it.
- [ ] Add a short "Highlights" paragraph at the top of 0.1.0 (the section is long; reviewers and Zenodo readers need a summary).
- [ ] Add a note that `0.0.1` and `0.0.2` were never tagged (do not retro-tag).
- [ ] Add compare links at the bottom (`[0.1.0]: https://github.com/TrevorJA/SynHydro/releases/tag/v0.1.0`).

### 2.2 CITATION.cff (new file, repo root)

- [ ] Create from the template in Appendix A. Fill in ORCID and affiliation.
- [ ] Validate with `pip install cffconvert && cffconvert --validate`.
- [ ] Confirm GitHub shows "Cite this repository" in the sidebar after pushing.

Notes: no `CITATION.cff`, `.zenodo.json`, or `codemeta.json` exists today.
Without one, Zenodo falls back to the GitHub contributor list for
authorship. Explicit metadata is what makes the DOI record correct.

### 2.3 Docs site

- [ ] Update `docs/getting-started.md` ("not yet published on PyPI") to the PyPI install command.
- [ ] Add a "How to cite" page (or section on the index) with the version DOI and concept DOI, once they exist.
- [ ] Run `/check-citations` on every `docs/algorithms/*.md` page.
- [ ] Re-execute all `examples/*.ipynb` with the release code and commit the outputs (the docs build uses `execute: false`, so committed outputs are what readers see).
- [ ] Add `--strict` to the `mkdocs build` step in `.github/workflows/docs.yml`.

Notes: `mkdocs build --strict` passed on 2026-09-14 with no warnings from
the project (one upstream notice from Material about MkDocs 2.0, which
the `mkdocs-material<9.7` pin already handles).

### 2.4 Repo hygiene visible in the archive

- [x] Untrack `experiments/`: `git rm -r --cached experiments`, then replace the eight partial `experiments/...` lines in `.gitignore` (lines 179 to 186) with a single `experiments/`. Local copies stay on disk.
- [ ] Confirm nothing copyright-restricted is tracked. Today: reference PDFs, `external_source_codes/`, `AUDIT.md`, and `docs/dev/` are all git-ignored, so they will not be in the Zenodo zip. Blog figure PNGs and example CSVs are tracked and are yours.
- [x] Untrack the dev notebooks: `git rm --cached examples/_dev_drought_distributions.ipynb examples/_dev_parameter_system.ipynb`, then add `examples/_dev_*.ipynb` to `.gitignore`. Local copies stay on disk. The docs hook only copies `NN_*.ipynb` and nothing else references them, so the docs build is unaffected.
- [ ] Optional: add `codemeta.json` for machine-readable metadata (not needed for Zenodo).

Notes: the six tracked files are `experiments/model_comparison/.README`
and `experiments/model_diagnostics/{config.py, plotting.py, run_all.sh,
run_all_mpi.py, run_diagnostic.py}`. `experiments/release_smoke/run_smoke.py`
is already ignored; consider promoting it to `tests/test_smoke.py` before
the directory disappears from the archive.

---

## 3. CI and release automation

### 3.1 Test workflow

- [x] Add a non-editable install job to `tests.yml`: `pip install .`, then run the smoke check from a different working directory. This would have caught 1.1.
- [ ] Optional: enable branch protection on `main` requiring the test matrix to pass.
- [ ] Optional: add a codecov badge (coverage upload already exists in `tests.yml`).

Notes: current workflows are `tests.yml` (3 OS x Python 3.10 to 3.12,
editable install) and `docs.yml` (deploys on push to `main`). There is no
release or publish workflow. `tests.yml` also lists a `dev` branch that
does not exist; harmless.

### 3.2 PyPI via Trusted Publishing (about an hour of one-time setup)

Trusted Publishing lets GitHub Actions upload without API tokens. PyPI
supports registering a "pending publisher" before the project exists, so
the first upload can already come from the workflow.

- [ ] Create a PyPI account (pypi.org) and enable two-factor authentication (required for all uploaders).
- [ ] Create a TestPyPI account (test.pypi.org) the same way. The two sites are separate.
- [ ] On PyPI: Account settings, Publishing, "Add a new pending publisher". Project name `synhydro`, owner `TrevorJA`, repository `SynHydro`, workflow `release.yml`, environment `pypi`.
- [ ] On TestPyPI: same, with environment `testpypi`.
- [ ] On GitHub: Settings, Environments, create `pypi` and `testpypi`. Optionally add yourself as a required reviewer on `pypi` so publishing waits for a click.
- [x] Add `.github/workflows/release.yml` from Appendix B. The workflow file name and environment names must match the PyPI configuration exactly.
- [ ] Rehearse: bump to `0.1.0rc1`, push tag `v0.1.0rc1` (no GitHub release, so Zenodo does nothing), confirm the TestPyPI upload, then `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ synhydro==0.1.0rc1` in a fresh venv and run the quick example.

Notes: a version number can never be re-uploaded to PyPI or TestPyPI. A
broken `0.1.0` upload is fixed by yanking it and releasing `0.1.1`. That
is why the rc rehearsal on TestPyPI is worth doing once.

---

## 4. Pre-release verification (run locally on the release commit)

- [ ] `venv/Scripts/python -m pytest tests/` passes. Survey run 2026-09-14: 1093 passed (Appendix C); 1118 passed later that day after the audit session added tests. Re-run on the final release commit.
- [ ] `pre-commit run --all-files` is clean, or decide explicitly that black is not enforced for 0.1.0 (running it repo-wide may produce a large diff; do it in its own commit if so).
- [ ] `venv/Scripts/python -m hatchling build` produces a wheel and a small sdist; inspect both (`unzip -l`, `tar tzf`).
- [ ] `twine check dist/*` passes (README renders, metadata valid).
- [ ] Install the wheel into a fresh venv, `cd` elsewhere, run the README quick example end to end.
- [ ] Run the smoke script (`experiments/release_smoke/run_smoke.py`, or its promoted `tests/test_smoke.py`).
- [ ] `mkdocs build --strict` passes.
- [ ] `cffconvert --validate` passes.
- [ ] CI matrix is green on the release commit before tagging.
- [ ] AUDIT.md closeout (1.7) is complete and logged.

---

## 5. Zenodo setup (one-time, before the release)

### Why Zenodo

- Free, CERN-operated, DataCite DOIs, accepted by AGU/WRR and other journals for software citation.
- GitHub integration mints a DOI automatically for every release.
- Issues a concept DOI (resolves to the latest version) plus a version DOI per release. Cite the version DOI in papers; put the concept DOI in the README.
- JOSS requires the accepted release to be archived on Zenodo (or figshare) anyway, so starting here means 0.1.0 through 1.0.0 live under one concept record.

Alternatives considered: Software Heritage (permanent archive with SWHIDs, complements Zenodo but does not issue DOIs), figshare and OSF (DOIs, weaker GitHub automation), Cornell eCommons (institutional DOI, manual upload). Zenodo is the community default.

### Steps

- [ ] Create or log in to a Zenodo account (sign in with GitHub, or link GitHub in profile settings).
- [ ] Profile menu, GitHub, "Sync now", toggle `TrevorJA/SynHydro` ON. The repository must be public and you need admin rights on it.
- [ ] Do this BEFORE publishing the GitHub release. Releases published before the toggle is on are not archived retroactively.
- [ ] Make sure `CITATION.cff` is on `main` before tagging so the record's authors, license, keywords, and description are correct on first archive. Metadata can be edited later in Zenodo, but the archived files cannot.
- [ ] Optional rehearsal: enable the repo on `sandbox.zenodo.org` (separate account and GitHub app) or use a throwaway fork, publish a test release, check the record, then disable. Do not publish a pre-release (`v0.1.0rc1`) as a GitHub release on the real repo: Zenodo archives pre-releases too and published records cannot be deleted by the user. Pushing the rc tag alone (Section 3.2) is safe; only a published GitHub release triggers Zenodo.
- [ ] Decide whether to add the record to a Zenodo community (optional).

Gotchas:

- Draft GitHub releases are not archived; published releases and pre-releases are.
- The DOI is minted at release time and cannot be pre-reserved through the GitHub flow, so the README and `CITATION.cff` get the DOI in a follow-up commit (Section 7). This is normal.
- Zenodo archives GitHub's source zip of the tag: tracked files only.

---

## 6. Release day (in order)

1. [ ] All work merged to `main`, working tree clean, CI green, AUDIT closeout done.
2. [ ] Bump version to `0.1.0` (single source), set the CHANGELOG date, set `version` and `date-released` in `CITATION.cff`.
3. [ ] Commit: `Release 0.1.0`.
4. [ ] Annotated tag: `git tag -a v0.1.0 -m "SynHydro 0.1.0"`.
5. [ ] Push `main` and the tag (you manage pushes). The tag push runs `release.yml`: build, `twine check`, smoke-install, upload to TestPyPI.
6. [ ] Confirm the TestPyPI upload and the build artifacts look right. If anything is wrong, fix it now: no GitHub release exists yet, so nothing is archived and PyPI is untouched. Delete the tag, fix, re-tag.
7. [ ] GitHub: Releases, "Draft a new release", choose tag `v0.1.0`, title `SynHydro v0.1.0`, paste the CHANGELOG 0.1.0 section as notes, Publish. Publishing triggers two things at once: Zenodo archives the tag, and `release.yml` attaches the wheel and sdist to the release and publishes to PyPI (after the environment approval if you set one).
8. [ ] Wait a few minutes; open Zenodo, GitHub tab, confirm the record. Record both DOIs here:
   - Version DOI (0.1.0): `10.5281/zenodo.________`
   - Concept DOI (all versions): `10.5281/zenodo.________`
9. [ ] Review the Zenodo record metadata (title, authors, ORCID, license, description) and edit in Zenodo if anything is off.
10. [ ] Confirm https://pypi.org/project/synhydro/0.1.0/ exists and the README renders.
11. [ ] `pip install synhydro==0.1.0` in a fresh venv, `cd` elsewhere, run the quick example one more time.

---

## 7. Post-release

### 7.1 Repo updates

- [ ] Add the Zenodo DOI badge and a PyPI version badge to `README.md` and `docs/index.md` (Zenodo shows the badge markdown on its GitHub page; it uses the concept DOI so it always resolves to the latest version).
- [ ] Add `doi` (concept DOI) to `CITATION.cff`, and `identifiers` for the version DOI.
- [ ] Add the "How to cite" page to the docs with a ready-to-paste reference (Appendix D).
- [ ] Bump `__version__` to `0.2.0.dev0` so development installs are distinguishable from the release.
- [ ] Update downstream pins (for example Pywr-DRB) to `synhydro==0.1.0` or `synhydro>=0.1,<0.2`.

### 7.2 Using the DOI in papers

- Cite the version DOI of the exact release used to generate the ensembles, not the concept DOI.
- Put the version number and DOI in the paper's data and software availability statement.
- Keep, per paper, the SynHydro version, generator class, parameters, and seeds. The `synhydro_version` metadata item (1.6) makes this automatic for HDF5 ensembles.
- Each later release gets its own version DOI under the same concept DOI, so citations stay unambiguous.

---

## 8. Suggested session plan

The audit work and the packaging work touch different files, so they can
run in separate sessions in either order.

| Session | Scope | Files touched |
|---|---|---|
| A. Audit closeout (done 2026-09-14, changes uncommitted) | Section 1.7 | generator modules, `tests/test_*_generator.py`, `docs/algorithms/*.md`, `AUDIT.md` |
| B. Packaging | 1.1 to 1.6, 2.4 | `pyproject.toml`, `src/synhydro/utils/`, `src/synhydro/data/`, `plotting/__init__.py`, `core/ensemble.py`, `.gitignore`, one new test |
| C. Metadata and docs | 2.1 to 2.3, README | `CHANGELOG.md`, `CITATION.cff`, `README.md`, `docs/` |
| D. Automation and accounts | 3.1, 3.2, 5 | `.github/workflows/`, PyPI, TestPyPI, Zenodo, GitHub environments; ends with the `v0.1.0rc1` TestPyPI rehearsal |
| E. Release | 4, 6, 7 | version bump, tag, release, DOI follow-up commit |

Session E cannot start until A through D are done. B and C are each an
hour or two. D is mostly account setup in a browser.

---

## 9. Toward 1.0.0 and JOSS (not part of this release)

Items to keep in view so 0.x releases build toward them:

- JOSS requires: an OSI license (done), a `paper.md` with statement of need, installation and usage docs (done), automated tests (done), community guidelines (CONTRIBUTING.md exists; add a code of conduct), and a release archived on Zenodo with a version DOI at acceptance.
- JOSS reviewers check that the software is a substantial scholarly effort; the algorithm pages and audit history support that.
- Consider JORS if you want a longer-form paper describing the verification framework.
- API stability commitments should be stated before 1.0.0; the 0.x series is where breaking changes such as the `NowakDisaggregator` argument renames belong.

---

## Appendix A. CITATION.cff template

```yaml
cff-version: 1.2.0
message: "If you use SynHydro in your research, please cite it using these metadata."
type: software
title: "SynHydro: Synthetic hydrologic timeseries generation in Python"
abstract: "Python library for generating synthetic hydrologic timeseries. Provides parametric, hybrid, and non-parametric stochastic streamflow generators and temporal disaggregators under a unified fit/generate API, with ensemble management, verification, and drought validation tools."
authors:
  - family-names: Amestoy
    given-names: Trevor
    email: tja73@cornell.edu
    orcid: "https://orcid.org/0000-0000-0000-0000"
    affiliation: "Cornell University"
version: 0.1.0
date-released: "2026-XX-XX"
license: MIT
repository-code: "https://github.com/TrevorJA/SynHydro"
url: "https://trevorja.github.io/SynHydro/"
keywords:
  - hydrology
  - synthetic streamflow
  - stochastic generation
  - time series
  - disaggregation
  - water resources
# Add after the first Zenodo record exists:
# doi: 10.5281/zenodo.XXXXXXX          (concept DOI)
# identifiers:
#   - type: doi
#     value: 10.5281/zenodo.YYYYYYY    (version DOI for 0.1.0)
#     description: "Zenodo archive of version 0.1.0"
```

## Appendix B. Release workflow (`.github/workflows/release.yml`)

Tag push: build, check, smoke-install, upload to TestPyPI.
GitHub release published: attach artifacts to the release, publish to PyPI.

```yaml
name: Release

on:
  push:
    tags: ["v*"]
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - name: Smoke-install the wheel from outside the repo
        run: |
          pip install dist/*.whl
          cd /tmp
          python -c "import synhydro; synhydro.load_example_data(); print(synhydro.__version__)"
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  testpypi:
    needs: build
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  pypi:
    needs: build
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Attach artifacts to the GitHub release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Notes: `pypa/gh-action-pypi-publish` refuses to re-upload an existing
version, so re-running a failed job is safe. The `environment:` names must
match the Trusted Publisher configuration on each index.

## Appendix C. Survey results, 2026-09-14 (commit 98751ad)

| Check | Result |
|---|---|
| Git tags | none (CHANGELOG lists 0.0.1 and 0.0.2 but they were never tagged) |
| Version sources | `pyproject.toml` 0.0.2, `src/synhydro/__init__.py` 0.0.2, stray `plotting/__init__.py` 2.0.0 |
| Citation metadata | no `CITATION.cff`, `.zenodo.json`, or `codemeta.json` |
| PyPI name `synhydro` | unclaimed |
| Wheel build | OK, 95 files, 319 KB, no example data inside |
| Non-editable install | `load_example_data()` fails with `FileNotFoundError` |
| sdist build | 142 MB, includes `cache/aiohttp_cache.sqlite` |
| `mkdocs build --strict` | passes |
| Full test suite | see below |
| Tracked non-library dirs | `examples/` (14 files), `experiments/` (6 files, to be untracked), `docs/` (35 files) |
| Git-ignored (absent from archive) | reference PDFs, `external_source_codes/`, `AUDIT.md`, `docs/dev/`, `cache/`, `site/` |
| Commit history | some commits carry `Co-Authored-By: Claude Sonnet 4.6` trailers; already published, leave as is. They do not affect Zenodo authorship when `CITATION.cff` is present. |

Full test suite (`pytest tests/`, venv Python 3.10.10, Windows): 1093 passed,
0 failed, 1 warning (joblib physical-core lookup on Windows, harmless), 185 s.
AUDIT.md reported 1148 on 2026-08-20; the difference is the `tests/_dev/`
suite, which `norecursedirs` now excludes.

## Appendix D. Suggested citation text (fill in after release)

```
Amestoy, T. (2026). SynHydro: Synthetic hydrologic timeseries generation
in Python (Version 0.1.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.YYYYYYY
```

BibTeX:

```bibtex
@software{amestoy_synhydro_2026,
  author    = {Amestoy, Trevor},
  title     = {{SynHydro}: Synthetic hydrologic timeseries generation in Python},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.YYYYYYY},
  url       = {https://github.com/TrevorJA/SynHydro}
}
```
