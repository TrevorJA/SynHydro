# Phase Randomization (Brunner et al. 2019)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Daily |
| **Sites** | Univariate |
| **Class** | `PhaseRandomizationGenerator` |

## Overview

Phase randomization generates synthetic daily streamflow by randomizing the Fourier phase spectrum while preserving the amplitude (power) spectrum. This maintains both short-range (daily autocorrelation) and long-range (Hurst phenomenon) temporal dependence. A four-parameter kappa distribution fitted per day-of-year allows extrapolation beyond the observed range.

## Algorithm

### Preprocessing

1. **Remove leap days** — ensures consistent 365-day years.
2. **Create day-of-year index** (1–365).
3. **Validate** — minimum 730 days (2 complete years), no missing observations.

### Fitting

1. **Marginal distribution fitting** (if `marginal='kappa'`) — for each day d:
   - Define moving window of +/- `win_h_length` days (circular, wraps at year boundary)
   - Extract all observations in window across all years
   - Fit four-parameter kappa distribution via L-moment matching
2. **Normal score transform** — for each day d:
   - Rank all observations across years
   - Map to standard normal quantiles via rank matching
   - Result: normalized series with N(0,1) marginals per day
3. **Fourier transform** of normalized series:
   - Compute FFT; extract modulus (amplitude) and phases
   - Store first-half indices and mirror indices for conjugate symmetry

### Generation

1. **Phase randomization**:
   - Keep DC component (index 0) unchanged
   - For positive frequencies: generate random phases from Uniform(-pi, pi)
   - Construct `FT_new[k] = modulus[k] * exp(i * phase_random[k])`
   - Apply conjugate symmetry for negative frequencies
2. **Inverse FFT** — produces phase-randomized series in normalized domain.
3. **Back-transform to original distribution** — for each day d:
   - If kappa: generate kappa sample, rank-match against normalized values
   - If empirical: rank-match directly against observed (no extrapolation)
4. **Non-negativity** — replace negative values with `Uniform(0, min_obs_d)`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marginal` | `str` | `'kappa'` | Marginal distribution: `'kappa'` (extrapolation) or `'empirical'` |
| `win_h_length` | `int` | `15` | Half-window for daily distribution fitting (total = 2*h + 1 days) |

## Properties Preserved

- Full power spectrum (all temporal autocorrelations, in expectation)
- Long-range dependence (Hurst coefficient)
- Day-of-year marginal distributions (via kappa or empirical)
- Seasonal patterns

**Not preserved:**
- Phase coherence (randomized by design)
- Exact autocorrelations (preserved only in expectation across ensemble)

## Limitations

- Univariate only — no spatial correlations between sites
- Generated series has same length as observed (no temporal extrapolation)
- Output excludes February 29 dates
- Kappa fitting may fail for some days — falls back to adjacent day parameters
- Minimum 2 years of data; 10+ recommended for stable kappa fits

## References

**Primary:**
Brunner, M.I., Bárdossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019

**See also:**
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94.
- Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society Series B*, 52, 105-124.
- Hosking, J.R.M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38, 251-258.

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/phase_randomization.py`
**Tests:** `tests/test_phase_randomization_generator.py`
