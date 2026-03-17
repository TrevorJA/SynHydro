# ARFIMA - Autoregressive Fractionally Integrated Moving Average (Hosking 1984)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate |
| **Class** | `ARFIMAGenerator` |

## Overview

ARFIMA(p,d,q) extends classical ARMA models by allowing the differencing parameter d to take fractional values in (0, 0.5), enabling the model to reproduce long-range dependence (LRD) observed in hydrologic timeseries. The fractional differencing operator produces hyperbolic decay in the autocorrelation function, matching the Hurst phenomenon that standard AR/ARMA models fundamentally cannot capture. This makes ARFIMA particularly important for generating synthetic flows that preserve multi-year drought persistence and low-frequency variability.

The Hurst exponent H is related to the fractional differencing parameter by H = d + 0.5, providing a direct link between the model parameter and the observed long-memory behavior of the timeseries.

## Algorithm

### Preprocessing

1. **Validate input** as univariate timeseries with at least 30 timesteps (50+ recommended for reliable d estimation).
2. **Deseasonalize** if monthly data: remove monthly means and divide by monthly standard deviations to produce stationary residuals.

### Fitting

1. **Estimate fractional differencing parameter d** using one of:
   - **Whittle estimator** (frequency-domain MLE, recommended; Fox & Taqqu 1986):
     ```
     d_hat = argmin_d sum_j [ log f(w_j; d) + I(w_j) / f(w_j; d) ]
     ```
     where I(w_j) is the periodogram and the spectral density is:
     ```
     f(w; d) ∝ [2(1 - cos(w))]^{-d}
     ```
     derived from |1 - e^{-iw}|^{-2d} = [2(1 - cos(w))]^{-d} (Hosking 1981, eq. 2.3).
   - **R/S analysis** for Hurst exponent: compute H, then d = H - 0.5.
   - **GPH (Geweke-Porter-Hudak)** log-periodogram regression (Geweke & Porter-Hudak 1983):
     ```
     log I(w_j) = c - d * log[2(1 - cos(w_j))] + u_j
     ```
     OLS regression on the m = sqrt(n) lowest Fourier frequencies; slope = -d.

2. **Apply fractional differencing** to obtain the fractionally differenced series:
   ```
   (1 - B)^d X_t = sum_{k=0}^{inf} pi_k X_{t-k}
   ```
   where the fractional differencing coefficients are:
   ```
   pi_0 = 1
   pi_k = pi_{k-1} * (k - 1 - d) / k,  for k >= 1
   ```
   Truncate the infinite sum at lag K (default: 100).

3. **Fit AR(p) to the differenced series** via Yule-Walker equations:
   - p is user-specified (default: 1). The MA component (q > 0) is not yet implemented.
   - Store AR coefficients phi and innovation variance sigma_eps^2.

4. **Store all fitted parameters**: d, phi, theta, sigma_eps^2, seasonal means/stds (if monthly), truncation lag K.

### Generation

1. **Generate AR innovations** for the differenced series:
   ```
   eps_t ~ N(0, sigma_eps^2)
   W_t = phi_1 * W_{t-1} + ... + phi_p * W_{t-p} + eps_t
   ```

2. **Invert fractional differencing** via MA convolution (FIR filter) to recover the long-memory process:
   ```
   X_t = sum_{k=0}^{K} psi_k * W_{t-k}
   ```
   where the inverse coefficients are:
   ```
   psi_0 = 1
   psi_k = psi_{k-1} * (k - 1 + d) / k,  for k >= 1
   ```
   Note: this is a moving-average (convolution) over the ARMA series W, not an autoregressive recursion over X.

3. **Re-seasonalize** if monthly: multiply by monthly standard deviations and add monthly means.

4. **Enforce non-negativity**: clip to zero or apply back-transformation if log-space was used.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Q_obs` | `pd.Series` or `pd.DataFrame` | - | Observed streamflow with DatetimeIndex |
| `p` | `int` | `1` | AR order for the short-memory component |
| `q` | `int` | `0` | MA order (currently only q=0 is supported) |
| `d_method` | `str` | `'whittle'` | Estimation method for d: `'whittle'`, `'gph'`, or `'rs'` |
| `truncation_lag` | `int` | `100` | Truncation lag K for fractional differencing coefficients |
| `deseasonalize` | `bool` | `True` | Remove monthly seasonality before fitting (set False for annual data) |
| `name` | `str` | `None` | Optional name identifier for this generator instance |
| `debug` | `bool` | `False` | Enable debug logging |

## Properties Preserved

- Long-range dependence / Hurst exponent (directly parameterized via d)
- Lag-1 through lag-p short-memory autocorrelation (via AR component)
- Monthly means and standard deviations (via deseasonalization)
- Power spectrum at low frequencies (hyperbolic decay)

**Not preserved:**
- Spatial correlations (univariate method)
- Non-Gaussian marginal distributions (Gaussian innovations assumed)
- Non-stationarity or trends

## Limitations

- Univariate only. For multisite applications, combine with spatial correlation methods
- Requires long records (50+ years) for reliable estimation of d
- Gaussian innovation assumption may underrepresent extreme events
- Truncation of infinite fractional differencing series introduces approximation error
- MA component (q > 0) not yet implemented; only AR short-memory component is available

## References

**Primary:**
Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using fractional differencing. *Water Resources Research*, 20(12), 1898-1908. https://doi.org/10.1029/WR020i012p01898

**See also:**
- Granger, C.W.J., and Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15-29. https://doi.org/10.1111/j.1467-9892.1980.tb00297.x
- Geweke, J., and Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis*, 4(4), 221-238. https://doi.org/10.1111/j.1467-9892.1983.tb00371.x
- Fox, R., and Taqqu, M.S. (1986). Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series. *The Annals of Statistics*, 14(2), 517-532.
- Montanari, A., Rosso, R., and Taqqu, M.S. (1997). Fractionally differenced ARIMA models applied to hydrologic time series: Identification, estimation, and simulation. *Water Resources Research*, 33(5), 1035-1044. https://doi.org/10.1029/97WR00043
- Koutsoyiannis, D. (2002). The Hurst phenomenon and fractional Gaussian noise made easy. *Hydrological Sciences Journal*, 47(4), 573-595.

---

**Implementation:** `src/synhydro/methods/generation/parametric/arfima.py`
**Tests:** `tests/test_arfima_generator.py`
