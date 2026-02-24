# Matalas (1967) Multi-Site MAR(1)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |
| **Class** | `MATALASGenerator` |

## Overview

The Matalas MAR(1) model is the classical parametric baseline for multi-site stochastic streamflow generation. It extends the univariate Thomas-Fiering model to n sites by fitting a matrix autoregressive model to standardized monthly flows. A separate pair of coefficient matrices is estimated for each of the 12 calendar-month transitions.

## Algorithm

### Preprocessing

1. Validate input; resample to monthly if daily data are provided.
2. Clip values to 1e-6 to avoid log of zero.
3. Optionally apply `log(Q + 1)` transformation to reduce skewness.

### Fitting

1. **Standardize** observed flows by monthly means and standard deviations:
   ```
   Z(t) = (Q(t) - mu_m) / sigma_m,    m = month(t)
   ```
2. **Estimate cross-correlation matrices** for each transition m to m+1:
   ```
   S0(m) = (1/(n-1)) * Z(m)^T * Z(m)         # lag-0 covariance
   S1(m) = (1/(n-1)) * Z(m+1)^T * Z(m)        # lag-1 cross-covariance
   ```
3. **Solve for coefficient matrices:**
   ```
   A(m) = S1(m) * S0(m)^{-1}                   # AR coefficient matrix
   M(m) = S0(m+1) - A(m) * S0(m) * A(m)^T      # innovation covariance
   B(m) = cholesky(M(m))                         # lower Cholesky factor
   ```
   If M(m) is not positive semi-definite, project to nearest PSD matrix via eigenvalue clipping before Cholesky decomposition.

   The December to January transition wraps across the year boundary.

### Generation

1. **Initialize**: `Z_0 ~ N(0, I)`
2. **Recurse** for each subsequent month:
   ```
   Z(t+1) = A(m) * Z(t) + B(m) * epsilon(t+1),    epsilon ~ N(0, I)
   ```
3. **Back-transform** to flow space:
   ```
   Q(t) = sigma_m * Z(t) + mu_m
   ```
4. If log transform was applied: `Q = exp(Q) - 1`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_transform` | `bool` | `True` | Apply log(Q+1) before standardization |

## Properties Preserved

- Monthly means and standard deviations at each site
- Lag-1 serial correlation at each site
- Contemporaneous cross-site correlations

**Not preserved:**
- Higher-order autocorrelation (lag > 1)
- Non-Gaussian marginal distributions

## Limitations

- First-order memory only (lag-1)
- Normality assumption after transformation
- Requires sufficient record length for stable covariance estimation
- Covariance matrices may require PSD repair with limited data

## References

**Primary:**
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology. *Water Resources Research*, 3(4), 937-945. https://doi.org/10.1029/WR003i004p00937

**See also:**
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/generation/parametric/matalas.py`
**Tests:** `tests/test_matalas_generator.py`
