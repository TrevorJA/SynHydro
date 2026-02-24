# WARM — Wavelet Auto-Regressive Method (Nowak et al. 2011)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Univariate |
| **Class** | `WARMGenerator` |

## Overview

WARM combines continuous wavelet transforms with autoregressive modeling to generate synthetic annual streamflow. Its key innovation is the Scale Averaged Wavelet Power (SAWP), which captures time-varying spectral characteristics. This makes WARM particularly suited for non-stationary flows with regime shifts or low-frequency variability (e.g., climate oscillations, multi-decadal droughts).

## Algorithm

### Preprocessing

1. Validate input (univariate time series, minimum ~20 years recommended).
2. Resample to annual frequency if needed (sum monthly or daily).

### Fitting

1. **Continuous Wavelet Transform** — decompose observed flows into time-frequency components using mother wavelet (default: Morlet):
   ```
   W(s, t) = CWT(Q_obs, scales, wavelet)
   ```
   Result: coefficient matrix of shape (n_scales, n_years).

2. **Scale Averaged Wavelet Power** — compute time-varying power across all scales:
   ```
   SAWP(t) = (1/S) * sum_s |W(s, t)|^2
   ```
   High SAWP indicates high energy/variability at time t.

3. **Normalize by SAWP** — remove time-varying power to produce stationary components:
   ```
   W_norm(s, t) = W(s, t) / sqrt(SAWP(t) + epsilon)
   ```

4. **Fit AR models** — for each scale s, fit an AR(p) model to normalized coefficients via Yule-Walker equations. Store parameters: mean, AR coefficients, innovation variance.

### Generation

1. **Generate normalized coefficients** — for each scale, run the fitted AR(p) process forward with Gaussian innovations.
2. **Resample SAWP** — bootstrap SAWP values from the historical record with replacement.
3. **Rescale coefficients** — restore time-varying power:
   ```
   W_syn(s, t) = W_norm_syn(s, t) * sqrt(SAWP_syn(t) + epsilon)
   ```
4. **Inverse wavelet transform** — reconstruct streamflow via weighted summation across scales. Adjust mean and variance to match observed, then clip negatives to zero.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelet` | `str` | `'morl'` | Mother wavelet (Morlet recommended for hydrology) |
| `scales` | `int` | `64` | Number of frequency scales (rule of thumb: T/2 to T) |
| `ar_order` | `int` | `1` | AR model order (1 or 2 typically sufficient) |

## Properties Preserved

- Mean and variance (exactly, via post-reconstruction adjustment)
- Lag-1 autocorrelation (approximately, via AR models)
- Multi-year persistence and low-frequency variability (via wavelet decomposition)
- Non-stationary variance structure (via SAWP)
- Power spectrum (via wavelet amplitude preservation)

**Not preserved:**
- Exact marginal distribution shape (Gaussian innovations)
- Skewness and higher moments (not explicitly modeled)
- Spatial correlations (univariate method)

## Limitations

- Annual frequency only — for finer resolution, couple with a disaggregation method
- Univariate (single site) — no native multi-site support
- Inverse CWT is approximate (PyWavelets); corrected by mean/variance adjustment
- Edge effects in CWT for short records (< 20 years)
- Gaussian AR innovations may underrepresent extremes

## References

**Primary:**
Nowak, K., Rajagopalan, B., and Zagona, E. (2011). A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary trends. *Journal of Hydrology*, 410(1-2), 1-12. https://doi.org/10.1016/j.jhydrol.2011.08.049

**See also:**
- Kwon, H.-H., Lall, U., and Khalil, A.F. (2007). Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition. *Water Resources Research*, 43(5).
- Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78.

---

**Implementation:** `src/synhydro/methods/generation/parametric/warm.py`
**Tests:** `tests/test_warm_generator.py`
