# Valencia-Schaake Temporal Disaggregation (Valencia and Schaake 1973)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual to Monthly (or Monthly to Weekly) |
| **Sites** | Univariate / Multisite |
| **Class** | `ValenciaSchaakeDisaggregator` |

## Overview

The Valencia-Schaake method is the foundational parametric temporal disaggregation approach in stochastic hydrology. It disaggregates an aggregate flow volume (e.g., annual total) into sub-period values (e.g., 12 monthly flows) using a linear regression model that preserves the conditional mean and covariance structure of the sub-periods given the aggregate. The method models sub-period flows as a multivariate normal distribution conditioned on the known aggregate, then samples from this conditional distribution.

This is the classical baseline against which all subsequent disaggregation methods are compared.

## Algorithm

### Preprocessing

1. **Validate input**: observed flows at the finer resolution (e.g., daily or monthly) with DatetimeIndex.
2. **Aggregate observed flows** to the coarser resolution (e.g., monthly to annual by summation).
3. **Organize into matrices**: for each year y, form the vector of sub-period flows:
   ```
   X_y = [Q_{y,1}, Q_{y,2}, ..., Q_{y,m}]^T
   ```
   where m is the number of sub-periods (e.g., 12 months).
4. **Optional transformation**: apply log or Box-Cox transform to improve normality.

### Fitting

1. **Compute sub-period statistics**:
   - Mean vector: `mu_X = E[X]` (m x 1)
   - Covariance matrix: `S_XX = Cov(X, X)` (m x m)
2. **Compute aggregate statistics**:
   - Mean: `mu_Y = E[Y]` where Y = sum(X)
   - Variance: `sigma_Y^2 = Var(Y)`
3. **Compute cross-covariance** between sub-periods and aggregate:
   ```
   S_XY = Cov(X, Y) = S_XX * 1_m
   ```
   where 1_m is an m-vector of ones.
4. **Compute regression parameters**:
   - Regression coefficients: `A = S_XY / sigma_Y^2` (m x 1)
   - Conditional covariance: `S_e = S_XX - A * sigma_Y^2 * A^T` (m x m)
5. **Cholesky decomposition** of S_e:
   ```
   S_e = C * C^T
   ```
   If S_e is not positive semi-definite, apply spectral repair (set negative eigenvalues to small positive value).
6. **Store** mu_X, mu_Y, sigma_Y^2, A, C, and transformation parameters.

### Disaggregation

For each synthetic aggregate value Y_syn:

1. **Compute conditional mean**:
   ```
   mu_X|Y = mu_X + A * (Y_syn - mu_Y)
   ```
2. **Sample residuals**:
   ```
   Z ~ N(0, I_m)
   X_syn = mu_X|Y + C * Z
   ```
3. **Proportional adjustment** to enforce consistency (sub-periods sum to aggregate):
   ```
   X_adj = X_syn * (Y_syn / sum(X_syn))
   ```
   Note: This multiplicative adjustment is simple but can distort the conditional covariance. See Grygier-Stedinger (1988) for a more rigorous conservation correction.
4. **Inverse transform** if log/Box-Cox was applied.
5. **Enforce non-negativity**: clip to zero.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_subperiods` | `int` | `12` | Number of sub-periods per aggregate period |
| `transform` | `str` | `'log'` | Transformation before fitting: `'log'`, `'boxcox'`, or `'none'` |
| `conservation_method` | `str` | `'proportional'` | Method to enforce sum consistency: `'proportional'` or `'none'` |

## Properties Preserved

- Conditional mean of sub-periods given aggregate (by construction)
- Conditional covariance structure (via S_e)
- Monthly means and standard deviations (approximately)
- Cross-correlations between sub-periods (via full covariance S_XX)
- Aggregate total (exactly, via proportional adjustment)

**Not preserved:**
- Exact conditional covariance after proportional adjustment
- Non-Gaussian features of sub-period distributions
- Inter-annual temporal correlations between sub-periods

## Limitations

- Assumes multivariate normality of sub-period flows (often violated for daily/monthly data)
- Proportional adjustment distorts the conditional covariance
- Does not model serial correlation between consecutive years' sub-periods
- For high m (many sub-periods), covariance estimation requires long records
- Better suited for annual-to-monthly than monthly-to-daily (daily data requires many more sub-periods)

## References

**Primary:**
Valencia, R.D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585. https://doi.org/10.1029/WR009i003p00580

**See also:**
- Stedinger, J.R., and Vogel, R.M. (1984). Disaggregation procedures for generating serially correlated flow vectors. *Water Resources Research*, 20(1), 47-56. https://doi.org/10.1029/WR020i001p00047
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/valencia_schaake.py`
**Tests:** `tests/test_valencia_schaake_disaggregator.py`
