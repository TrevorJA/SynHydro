# Grygier-Stedinger Condensed Disaggregation (Grygier and Stedinger 1988)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual to Monthly |
| **Sites** | Univariate / Multisite |
| **Class** | `GrygierStedingerDisaggregator` |

## Overview

The Grygier-Stedinger method extends the Valencia-Schaake disaggregation framework with two key improvements: (1) a condensed parameter set that reduces the number of parameters to be estimated, improving reliability with short records, and (2) a rigorous conservation correction that ensures sub-period values sum exactly to the aggregate without distorting the conditional covariance structure. This method forms the basis of the widely-used SPIGOT software (Stedinger et al., U.S. Army Corps of Engineers).

The conservation correction replaces the simple proportional adjustment of Valencia-Schaake with an adjustment that accounts for the covariance between the sum of generated sub-periods and each individual sub-period, producing statistically consistent disaggregated flows.

## Algorithm

### Preprocessing

1. **Validate input**: observed flows at the finer resolution (e.g., monthly) with DatetimeIndex.
2. **Aggregate to annual** by summation across sub-periods.
3. **Organize sub-period vectors** X_y for each year y.
4. **Apply transformation** (log or Wilson-Hilferty) to improve normality.

### Fitting

1. **Compute sub-period statistics** (same as Valencia-Schaake):
   - Mean vector mu_X, covariance matrix S_XX.
   - Aggregate mean mu_Y, variance sigma_Y^2.
   - Cross-covariance S_XY = S_XX * 1_m.

2. **Condensed parameterization**: rather than estimating the full m x m covariance S_XX, use a reduced parameter set:
   - Lag-0 cross-correlations between sub-periods and the aggregate
   - Lag-1 serial correlations between consecutive sub-periods
   - Sub-period means and standard deviations

   This reduces the parameter count from O(m^2) to O(m), making estimation feasible with shorter records.

3. **Compute regression coefficients** A and conditional covariance S_e as in Valencia-Schaake, but using the condensed parameter estimates.

4. **Compute conservation correction matrix** D:
   ```
   D = S_e * 1_m * (1_m^T * S_e * 1_m)^{-1}
   ```
   This is the key innovation. D is an m x 1 vector of correction weights that distributes the conservation error across sub-periods proportionally to their conditional variances.

5. **Cholesky decomposition** of S_e: `S_e = C * C^T`.

6. **Store** mu_X, mu_Y, A, C, D, and transformation parameters.

### Disaggregation

For each synthetic aggregate value Y_syn:

1. **Compute conditional mean**:
   ```
   mu_X|Y = mu_X + A * (Y_syn - mu_Y)
   ```

2. **Generate uncorrected sub-periods**:
   ```
   Z ~ N(0, I_m)
   X_raw = mu_X|Y + C * Z
   ```

3. **Apply conservation correction**:
   ```
   delta = Y_syn - sum(X_raw)
   X_syn = X_raw + D * delta
   ```
   The correction distributes the discrepancy delta across sub-periods using the weights D, which are derived from the conditional covariance structure. This preserves the statistical properties of the disaggregation while ensuring exact summation.

4. **Inverse transform** if log or Wilson-Hilferty was applied.

5. **Enforce non-negativity**: if any X_syn < 0, set to zero and redistribute the deficit across remaining sub-periods proportionally.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Q_obs` | `pd.Series` or `pd.DataFrame` | - | Observed streamflow at sub-period resolution with DatetimeIndex |
| `n_subperiods` | `int` | `12` | Number of sub-periods per aggregate period |
| `transform` | `str` | `'log'` | Transformation: `'log'`, `'wilson_hilferty'`, or `'none'` |
| `name` | `Optional[str]` | `None` | Optional name identifier for this disaggregator instance |
| `debug` | `bool` | `False` | Enable debug logging |

## Properties Preserved

- Conditional mean of sub-periods given aggregate (by construction)
- Conditional covariance structure (not distorted by conservation correction)
- Aggregate total (exactly, by construction via correction matrix D)
- Monthly means and standard deviations
- Lag-1 serial correlation between consecutive sub-periods (approximately)

**Not preserved:**
- Non-Gaussian marginal features
- Higher-order temporal correlations
- Non-stationarity

## Advantages over Valencia-Schaake

- Conservation correction preserves conditional covariance (proportional adjustment does not)
- Condensed parameterization requires fewer estimated parameters, improving reliability
- Explicitly handles the statistical consequences of forcing sub-periods to sum to the aggregate
- Better performance with short historical records

## Limitations

- Still assumes multivariate normality of transformed sub-period flows
- Condensed parameterization may miss complex cross-correlations between non-adjacent sub-periods
- Conservation correction assumes linear relationships; strong nonlinearity in the data may not be captured
- Transformation choice (log vs. Wilson-Hilferty) can affect results significantly

## References

**Primary:**
Grygier, J.C., and Stedinger, J.R. (1988). Condensed disaggregation procedures and conservation corrections for stochastic hydrology. *Water Resources Research*, 24(10), 1574-1584. https://doi.org/10.1029/WR024i010p01574

**See also:**
- Valencia, R.D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585. https://doi.org/10.1029/WR009i003p00580
- Stedinger, J.R., and Vogel, R.M. (1984). Disaggregation procedures for generating serially correlated flow vectors. *Water Resources Research*, 20(1), 47-56. https://doi.org/10.1029/WR020i001p00047
- Lane, W.L. (1979). Applied stochastic techniques (LAST computer package). User Manual, Division of Planning Technical Services, Bureau of Reclamation, Denver, CO.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/grygier_stedinger.py`
**Tests:** `tests/test_grygier_stedinger_disaggregator.py`
