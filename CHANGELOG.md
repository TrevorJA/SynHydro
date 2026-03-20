# Changelog

All notable changes to SynHydro are documented in this file.

## [Unreleased]

### Added
- Gaussian/t-copula generator with parametric and empirical marginals
- ARFIMA MA(q) component with CSS estimation and BIC-based order selection
- KNN Bootstrap generator for nonparametric multi-site generation
- Grygier-Stedinger temporal disaggregation method
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
- Migrated all generators to `np.random.Generator` (replaces legacy `np.random`)
- Replaced all `print()` with `logging.getLogger(__name__)`
- Major API refactor: standardized preprocessing/fit/generate interface
- Improved ensemble HDF5 loading performance
- Stedinger transform lower bound now correctly falls back to tau=0
  when the formula produces values outside the valid range
- Pinned minimum dependency versions in pyproject.toml
- Test fixtures produce physically realistic (non-negative) streamflow data
- Renamed package from SGLib to SynHydro

### Fixed
- Matalas correlation matrix repair via shared `repair_correlation_matrix`
- KNN Bootstrap `block_size` and `index_site` parameters
- Multisite `ValueError` for univariate generators
- Nowak non-leap-year February bug
- Kirsch correlation matrix bug
- SSI Python version compatibility issue
- SSI deprecated pandas `"M"` frequency string (now `"ME"`)
- Valencia-Schaake divide-by-zero in correlation matrix computation
- RuntimeWarnings from `np.log` on edge-case data in Stedinger transform
  and HMM preprocessing

### Removed
- Deprecated Kirsch-Nowak combined generator
- Outdated `core/validation.py` monolith (replaced by `core/validation/` package)

## [0.0.2] - 2025-06-09

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
