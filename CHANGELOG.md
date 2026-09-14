# Changelog

All notable changes to SynHydro are documented in this file.

## [Unreleased]

### Highlights

SynHydro 0.1.0 is the first tagged, citable release. Every generator and
disaggregator shares one workflow: `preprocessing()`, `fit()`, and
`generate()` (or `disaggregate()`), returning an `Ensemble` that can be
verified, plotted, and saved. The release ships eleven generators across
the parametric (Thomas-Fiering, Matalas, ARFIMA, SPARTA, SMARTA, multisite
HMM), hybrid (Kirsch, WARM, phase randomization, multisite phase
randomization), and non-parametric (KNN bootstrap) families, plus two
temporal disaggregators (Nowak, Valencia-Schaake). Each method has an
algorithm page that states where the implementation deviates from its
source paper, and statistical regression tests check the generators
against the properties those papers claim. The verification suite
(`synhydro.verify()`) scores an ensemble against the observed record
across nine statistical categories, and the validation suite
(`synhydro.validate()`) evaluates fitness for drought studies. Ensembles
are stored as compact HDF5 files that record the SynHydro version that
produced them. Packaging fixes make `pip install synhydro` work outside a
repository checkout: the example data ships inside the package, the
version is single-sourced, the sdist contains only the library, tests,
and metadata, and the license metadata follows PEP 639.

### Removed
- `HMMKNNGenerator` moved from `synhydro.methods.generation.hybrid` to the
  unexported `synhydro.methods.generation._dev` module and removed from the
  public API, README, and documentation. The regime-conditioned HMM + KNN
  combination is an experimental prototype that has not been verified
  against a published method.

### Added
- `EnsembleMetadata.synhydro_version` records the library version that
  produced an ensemble (default `synhydro.__version__`). `Ensemble.to_hdf5`
  writes it inside the JSON metadata and as a top-level HDF5 attribute
  `synhydro_version`, so h5py users can read it without parsing JSON.
  Files written before the field existed load with `synhydro_version`
  set to None and stay unversioned when re-saved.
- `plot_timeseries` and `plot_autocorrelation` accept a keyword-only
  `seed` argument (int or `np.random.Generator`) that fixes which ensemble
  members are drawn for `show_members`, so figures are reproducible. The
  default `seed=None` keeps the previous unseeded selection.
- Verification suite (`synhydro.verification`) for statistical property
  preservation, following the Stedinger and Taylor (1982) terminology:
  registered per-metric functions across nine categories (marginal,
  temporal, seasonal, annual, spatial, fdc, lmoments, extremes,
  spectral), a `verify()` orchestrator, and a `VerificationResult` with
  tidy `to_dataframe()`, `summary()` (including the observed value's
  rank position within the ensemble), and `category_summary()`.
  Metric selection is explicit (`metrics="all"`, names, categories, or
  callables); custom metrics register via `register_metric()`.
- Validation suite (`synhydro.validation`) for fit-for-purpose drought
  evaluation: `validate()` runs the threshold-drought and SSI-drought
  categories, relocated from the old validation framework with
  computations unchanged (verified bit-identical on a fixed-seed
  ensemble).
- Frequency-aware evaluation: ensemble frequency is normalized from
  metadata (including plain-word values), checked against the index,
  and used to gate metrics (7-day low flows daily-only, Hurst on
  annual aggregation with a 20-year minimum) and to report events per
  year.
- Rewritten `bootstrap_metric_ci` (vectorized) and `compare_methods`
  (true paired bootstrap when realization counts match) operating on
  the tidy result frame.
- Plotting: `plot_metric_distributions` (per-metric boxplots with
  observed overlay and rank annotation) and `plot_metric_curve`
  (ensemble band for FDC, ACF, monthly statistics, spectra).

- `NowakDisaggregator` generalized to arbitrary discrete timescale pairs via
  `input_timestep` / `output_timestep` constructor arguments: any input in
  {annual, monthly, weekly} disaggregates to any finer output in
  {monthly, weekly, daily} (six pairs). Weekly timesteps follow the ISO
  52-week convention (Sunday anchors, week 53 folded into week 52 pools),
  consistent with `KirschGenerator`. For monthly-to-weekly, weeks are
  assigned to the calendar month containing their Sunday anchor. Monthly to
  daily remains the default and its output is bit-identical to the previous
  implementation for a given seed (guarded by a golden-file regression test
  in `tests/test_nowak_regression.py`).
- `KirschGenerator` now supports weekly resolution in addition to monthly.
  `preprocessing(..., timestep="weekly")` aggregates daily input to ISO weekly
  (W-SUN) buckets, drops ISO week 53, and runs the Kirsch (2013) algorithm with
  52 periods per year and a 26-period half-shift. Monthly remains the default
  and is auto-detected when no `timestep` is supplied. Pre-aggregated weekly
  input (W-SUN family aliases) is also accepted.
- ARFIMA MA(q) component with CSS estimation and BIC-based order selection
- KNN Bootstrap generator for nonparametric multi-site generation
- Valencia-Schaake temporal disaggregation method
- Validation framework with 12 metric categories:
  - L-moment ratios, GEV extreme-value metrics
  - CRPS and SSI drought metrics
  - Bootstrap hypothesis testing module
- Month blending for Nowak disaggregation
- Input validation with warnings for poor data quality
- Pre-commit lint hook
- CONTRIBUTING.md
- MkDocs documentation site with algorithm reference pages

### Changed
- Packaging metadata: the license is declared per PEP 639
  (`license = "MIT"`, `license-files = ["LICENSE"]`, the `License ::`
  classifier dropped), the `Development Status` classifier is removed,
  `spei` is pinned to `>=0.8.2` (the only version tested), and the build
  backend is pinned to `hatchling>=1.27` so isolated builds understand
  the new license fields.
- README: `pip install synhydro` is the primary install command, with the
  tagged git install kept as the alternative; the CONTRIBUTING link is
  absolute so it resolves on PyPI; and a "Citing SynHydro" section with a
  Zenodo DOI placeholder follows the quick example. `docs/getting-started.md`
  shows the same install commands.
- Packaging: the version is single-sourced from `synhydro.__version__`
  (`dynamic = ["version"]` in `pyproject.toml`), the stray
  `synhydro.plotting.__version__` and `__author__` were removed, and the
  sdist is restricted to `src/`, `tests/`, and the top-level metadata
  files. A packaging test builds the wheel, installs it into a temporary
  directory, and loads the example data from outside the repository.
- `Ensemble.to_hdf5` writes substantially smaller files (about 35 percent
  smaller on a 10-realization, 5-site, 30-year daily benchmark; 5.9 MB to
  3.8 MB) with faster writes. Dates are now stored once as a fixed-length
  byte-string dataset and hard-linked into every group (variable-length
  strings bypassed HDF5 compression and were duplicated per site), and
  numeric datasets use the shuffle filter when compression is enabled.
  The on-disk layout (group per site with `date`, per-realization
  datasets, and a `column_labels` attribute) is unchanged and remains
  readable by pywrdrb; files written by older versions still load with
  `Ensemble.from_hdf5`.
- `Ensemble.to_hdf5` gained a `dtype` argument, defaulting to
  `"float32"`, which roughly halves file size again (about 68 percent
  total reduction on the benchmark) while keeping about 7 significant
  digits. Pass `dtype="float64"` (or None for the data's native dtype)
  to restore full-precision storage.
- Breaking: `validate_ensemble` and `compute_realization_metrics` are
  removed. Use `synhydro.verify()` for statistical verification and
  `synhydro.validate()` for drought validation. `ValidationResult` now
  refers to the fit-for-purpose result object.
- Breaking: `plot_validation_panel` is now `plot_verification_panel`.
  Rank-sum and Levene panels test each realization separately instead
  of pooling realizations (pooled p-values shrank with ensemble size),
  and the observed-year resampling accepts a `seed`.
- `fit_gev` returns the shape parameter in a single convention for
  both fitting methods (Hosking kappa, equal to the scipy `genextreme`
  c parameter). The L-moment fit previously had a sign error in the
  location parameter and the MLE fit negated the shape; GEV
  return-level callers no longer negate the shape.
- `compute_lmoments` (public) returns the first two L-moments and the
  L-skewness and L-kurtosis ratios.
- GEV metrics are named by return period (`gev_rp10`, `gev_rp50`,
  `gev_rp100`), replacing the mislabeled `gev_q10`/`gev_q50`/`gev_q100`.
- Breaking: `NowakDisaggregator` constructor arguments renamed for
  timescale-neutral clarity: `max_month_shift` is now
  `max_knn_pool_shift_timesteps` and `blend_days` is now
  `boundary_blend_timesteps`, both in units of output timesteps.
  Per-pair defaults apply when `max_knn_pool_shift_timesteps` is omitted
  (7 for monthly-to-daily, unchanged from before). The `KirschNowakPipeline`
  and `ThomasFieringNowakPipeline` public `max_month_shift` argument is
  unchanged. The legacy `disaggregate_monthly_flows()` method has been
  removed; use `disaggregate(ensemble, seed=...)`.
- `NowakDisaggregator` boundary smoothing now defaults to off
  (`boundary_blend_timesteps=0`, previously 2) so the default reproduces the
  published Nowak et al. (2010) method, which applies no boundary correction.
  Boundary smoothing remains available as an opt-in SynHydro extension.
- `KNNBootstrapGenerator` no longer accepts sub-monthly input. Restricted to
  monthly (Lall & Sharma, 1996; Prairie et al., 2006) and annual (Prairie et
  al., 2008) per the primary streamflow literature. Sub-monthly input now
  raises `ValueError`; for daily output, generate an annual realization and
  disaggregate with `NowakDisaggregator`. Daily KNN bootstrap in the
  literature is established only for weather variables (Rajagopalan and Lall,
  1999) or as a disaggregation step (Nowak et al., 2010), not as a standalone
  daily streamflow generator.
- Updated pandas frequency strings to current preferred names: `'AS'` ->
  `'YS'`, `'A'` -> `'YE'`, `'M'` -> `'ME'`. Input-alias sets in ARFIMA,
  WARM, and SMARTA still accept both old and new aliases for compatibility
  with `pd.infer_freq()` on older pandas versions.
- Reorganized generators into three classification bins based on the
  mathematical character of their generative mechanism (Studnicka and Panu,
  2025): `synhydro.methods.generation.parametric` (Thomas-Fiering, Matalas,
  ARFIMA, SPARTA, SMARTA, MS-HMM), `synhydro.methods.generation.hybrid`
  (Kirsch, WARM, Phase Randomization, MS Phase Randomization, HMM-KNN), and
  `synhydro.methods.generation.nonparametric` (KNN-Bootstrap). Top-level
  `synhydro.<Generator>` imports are unchanged.
- **Breaking:** `KirschGenerator` internal attribute renames to be
  resolution-agnostic: `n_months` -> `n_periods_per_year`,
  `mean_month` -> `mean_period`, `std_month` -> `std_period`, and the public
  property `Q_obs_monthly` -> `Q_obs_aggregated`. Downstream callers
  referencing these attributes (e.g., custom plotting code) must update.
  `MatalasGenerator` and `ThomasFieringGenerator` retain their original
  `Q_obs_monthly` naming.
- Migrated all generators to `np.random.Generator` (replaces legacy `np.random`)
- Replaced all `print()` with `logging.getLogger(__name__)`
- Major API refactor: standardized preprocessing/fit/generate interface
- Improved ensemble HDF5 loading performance
- Stedinger transform lower bound now correctly falls back to tau=0
  when the formula produces values outside the valid range
- Pinned minimum dependency versions in pyproject.toml
- Test fixtures produce physically realistic (non-negative) streamflow data
- SSI clip floor is now configurable via `cdf_epsilon` (keyword-only); the
  default changed from an implicit +/- 6.36 bound to +/- 4.5 to suppress
  gamma-tail extrapolation artifacts. Pre-existing values previously below
  -4.5 or above +4.5 will now be clipped to those bounds.
- Renamed package from SGLib to SynHydro

### Fixed
- Example datasets are now package data. `usgs_daily_streamflow_cms.csv`
  and `usgs_monthly_streamflow_cms.csv` moved from `examples/example_data/`
  to `src/synhydro/data/` and are located with `importlib.resources`, so
  `load_example_data()`, `list_example_datasets()`, and
  `get_example_data_path()` work for regular `pip install` users. The old
  lookup resolved a path relative to the repository checkout and raised
  `FileNotFoundError` on any non-editable install.
  `examples/example_data/retrieve_example_data.py` now writes into the
  package data directory; no copy is kept under `examples/`.
- Matalas correlation matrix repair via shared `repair_correlation_matrix`
- KNN Bootstrap `block_size` and `index_site` parameters
- Multisite `ValueError` for univariate generators
- Nowak non-leap-year February bug
- Kirsch correlation matrix bug
- Kirsch `generate()` and `generate_from_indices()` now build X' as the
  deterministic 6-month shift of X per Kirsch et al. (2013), p. 6, rather
  than an independent bootstrap (`generate()`) or a shared-index lookup
  (`generate_from_indices()`). All three entry points
  (`generate`/`generate_single_series`, `generate_from_indices`,
  `generate_from_residuals`) are now statistically equivalent and route
  through a shared `_pipeline_from_X` helper. Breaking for seed-level
  reproducibility: `.generate(seed=S)` produces different numerical output
  than prior releases (the corrected distribution).
- Kirsch `generate_from_residuals` now requires residuals of shape
  `(n_years + 1, n_periods_per_year, n_sites)`, matching
  `generate_from_indices`, instead of duplicating the last residual row,
  which made the last two synthetic years share an identical second half.
  Breaking for external callers that passed exactly `n_years` rows. The
  docstring now states that residuals must already be in the normal-score
  space when `generate_using_log_flow=True` and that only the inverse
  transform is applied.
- Kirsch weekly output DatetimeIndex is now built per synthetic year using
  `pd.Timestamp.fromisocalendar(y, w, 7)` instead of a plain
  `pd.date_range(freq="W-SUN")`. The previous range marched in 7-day steps
  across year boundaries, accumulating a ~1.25 day/year backward drift (since
  52 x 7 = 364 days), which manifested as a multi-week leftward shift in
  seasonal-cycle plots over multi-decade horizons. Each synthetic position
  `k` of year `y` now lands on the Sunday of ISO week `(k+1)` of `y` so the
  per-period mean/std applied at fit time aligns with the calendar week of
  the labeled date.
- SSI Python version compatibility issue
- SSI deprecated pandas `"M"` frequency string (now `"ME"`)
- Valencia-Schaake divide-by-zero in correlation matrix computation
- RuntimeWarnings from `np.log` on edge-case data in Stedinger transform
  and HMM preprocessing

### Removed
- CRPS metrics (`crps_mean`, `crpss`): probabilistic-forecast scores do
  not apply to stochastic generation, where realizations are
  independent draws rather than date-aligned forecasts.
- Ideal-value and ensemble-size-dependent metrics: `ks_pvalue`, pooled
  `monthly_wilcoxon_pvalue`, `fdc_envelope_coverage`,
  `spectral_correlation`, `low_freq_ratio`, `variance_ratio`,
  `acf_rmse`, `monthly_*_bias` aggregates, and the cross-category
  `mean_absolute_relative_error` summary score, which averaged
  incommensurable quantities.
- The `core/validation/` package, replaced by `synhydro.verification`
  and `synhydro.validation`.
- Dead legacy `synhydro.verification` package (pre-refactor SSI code).
- Deprecated Kirsch-Nowak combined generator
- Outdated `core/validation.py` monolith (replaced by `core/validation/` package)

### Known limitations

Deferred by maintainer decision on 2026-09-14. Each is a property of the
published method or a cosmetic issue rather than an implementation error;
method properties are also documented on the algorithm pages.

- `KirschGenerator` preserves cross-site (lag-0) correlation only
  approximately. The bootstrap indices are shared across sites, but each
  site's Cholesky factor mixes periods, so the synthetic cross-site
  correlation for a given month is a blend of the historical values that
  feed that column of the residual matrix. On the packaged USGS monthly
  record the deviation reaches 0.09 (December). If month-specific
  cross-site correlation matters, check the spatial category of
  `synhydro.verify()` and consider a parametric multisite generator
  (`MatalasGenerator`, `SPARTAGenerator`) that targets the lag-0
  cross-correlation matrix directly.
- `SMARTAGenerator` under-detects long-range dependence on records of
  about 100 years or fewer. The CAS fit minimizes the error against the
  biased sample ACF over lags 1 to n/3 with no bias correction, so a
  long-memory record can be fitted as short-range (a fractional Gaussian
  noise fixture with H = 0.75 is fitted as short-range on two of three
  sites), and the generator then faithfully reproduces that short-range
  target. This matters when multi-decadal persistence drives the analysis.
  For short records, `acf_model="hurst"` falls back to H = 0.6 when the
  CAS fit finds no long memory, which retains mild persistence; an
  externally estimated Hurst coefficient (for example from a climacogram)
  cannot be supplied in this release.
- `SMARTAGenerator.fit()` does not populate `fitted_params_`, so
  `get_fitted_params()` raises "has not been fitted yet" even after a
  successful fit, unlike the other generators. The per-site CAS fit is
  logged at debug level (`debug=True`).
- When a SMARTA site's fitted CAS autocorrelation decays to numerically
  zero (beta near 0, a short-memory site), the Nataf inversion logs a
  warning that target correlations are "below the attainable
  Frechet-Hoeffding lower bound" and were clipped. The message is harmless
  in this situation: the targets are below the resolution of the
  polynomial fit, not outside the attainable range, and the output is
  unaffected.
- `ARFIMAGenerator.log_transform.params_["tau"]` (the Stedinger lower
  bounds, one per month for monthly input) is an object-dtype DataFrame,
  so NumPy ufuncs applied to its `.values` fail. Call `.astype(float)`
  first; `get_fitted_params()` exposes the same values under
  `stedinger_tau`.
- `PhaseRandomizationGenerator` dilutes flow-domain autocorrelation on
  short records. The back-transform draws, for each day of year, an
  independent kappa sample with one value per observed year and assigns
  it by rank, so with 6 observed years the lag-1 flow autocorrelation
  drops from 0.93 to 0.75 even though the normal-score spectrum is
  preserved exactly. The gap shrinks with record length (0.12, 0.08, and
  0.05 at 12, 30, and 60 years) and follows PRSim's construction. Use the
  longest record available and check the temporal category of
  `synhydro.verify()` before relying on fits to records shorter than
  about 30 years.
- `PhaseRandomizationGenerator` departs from PRSim in two details of the
  kappa fit: invalid (kappa, h) combinations return a large penalty to
  the optimizer instead of aborting the fit as PRSim does, and the
  Nelder-Mead settings (`xatol` and `fatol` of 1e-6, `maxiter` of 1000)
  are not PRSim's. Fitted kappa parameters can therefore differ from
  PRSim's in the last digits, and days on which PRSim would abort are
  handled by the fit-rejection rule instead. This matters only when
  reproducing PRSim output digit for digit.
- `MultisitePhaseRandomizationGenerator` imposes nearly the same
  within-season cross-site dependence whatever the observed dependence
  was, because every site shares one random phase field and differs only
  in its wavelet amplitude envelope. The within-day-of-year rank
  correlation between sites is about 0.8 to 0.9 in the output both for
  sites whose observed anomalies are independent (observed 0.00) and for
  strongly dependent sites (observed 0.61 to 0.73). This is inherent to
  the PRSim.wave design, which targets spatially coherent basins. Use the
  generator for sites within one hydrologically coherent basin; for
  weakly dependent sites, fit `PhaseRandomizationGenerator` per site
  instead.
- `WARMGenerator`'s chi-squared significance test is conservative: on
  white-noise input about 2 percent of scales exceed the 95 percent
  threshold instead of the nominal 5 percent, and no significant band is
  found below roughly 8 years. Weak or short-period bands can therefore
  be missed. Pass the band explicitly with the `bands` argument or lower
  `significance_level` when a known band is not detected.
- `KNNBootstrapGenerator` implements only the Lall and Sharma (1996)
  bootstrap. The Prairie et al. (2006) local-polynomial modification and
  the Prairie et al. (2008) paleo-state conditioning listed under "See
  Also" in the module docstring are not implemented; the algorithm page
  is the authoritative description.

## [0.0.2] - 2025-06-09

Versions 0.0.1 and 0.0.2 were never tagged in git and will not be
retro-tagged; their entries are kept for the record. 0.1.0 is the first
tagged release.

### Added
- Thomas-Fiering monthly generator
- Kirsch nonparametric generator
- Nowak temporal disaggregation
- Multisite HMM generator
- WARM wavelet-based generator
- Phase randomization generator
- Matalas multivariate autoregressive generator
- SSI drought index calculation
- Ensemble management with HDF5 storage
- Pipeline for chaining generation and disaggregation
- Basic package structure with `pip install` support

## [0.0.1] - 2023-06-15

- Initial commit with project scaffolding

[Unreleased]: https://github.com/TrevorJA/SynHydro/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TrevorJA/SynHydro/releases/tag/v0.1.0
