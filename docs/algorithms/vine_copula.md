# Vine Copula Generator (Yu et al. 2025; Wang & Shen 2023; Pereira et al. 2017)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |
| **Class** | `VineCopulaGenerator` |

## Overview

Extends the PAR(1)-plus-copula framework of `GaussianCopulaGenerator` by
replacing the single elliptical copula with a vine copula.  Vine copulas
decompose a multivariate copula into a hierarchy of bivariate copulas
arranged in tree structures (R-vine, C-vine, or D-vine), allowing
heterogeneous pairwise dependence and asymmetric tail dependence.

Marginals and temporal dependence are identical to `GaussianCopulaGenerator`:
per-(month, site) parametric or empirical marginals with a Periodic AR(1)
temporal model.  Spatial dependence among the PAR(1) residuals is captured
by a monthly-periodic vine copula fitted via the Dissmann algorithm
(pyvinecopulib).

## Algorithm

### Preprocessing

1. Validate input data; resample daily/weekly to monthly if needed.
2. Optionally apply `log(Q + offset)` transform.

### Fitting

Steps 1-3 are identical to `GaussianCopulaGenerator`.

1. **Marginal fitting** (per calendar month, per site):
   - Parametric: fit gamma and log-normal via MLE; select by BIC.
   - Empirical: fit `NormalScoreTransform` (Hazen plotting position).

2. **Probability Integral Transform (PIT):**
   ```
   u[t,s] = F_{m,s}(Q[t,s])
   z[t,s] = Phi^{-1}(u[t,s])
   ```

3. **PAR(1) temporal model** (Pereira et al. 2017):
   ```
   rho_{m,s} = corr(z[month=m, s], z[month=m-1, s])
   e[t,s] = (z[t,s] - rho_{m,s} * z[t-1,s]) / sqrt(1 - rho_{m,s}^2)
   ```

4. **Vine copula per month** (new step):
   - Transform PAR residuals to uniform: `u = Phi(e)`
   - For each calendar month m = 1..12, fit a vine copula on the
     uniform PAR residuals using pyvinecopulib:
     - Structure: R-vine (automatic Dissmann), C-vine, or D-vine
     - Bivariate families: selected per edge by AIC/BIC from the
       configured family set (Gaussian, Student-t, Clayton, Gumbel, etc.)

### Generation

For each timestep t (calendar month m):

1. Draw from the vine copula for month m:
   ```
   u_new = vine_m.simulate(1)    # uniform [0,1]^n_sites
   e = Phi^{-1}(u_new)           # PAR residuals in normal space
   ```

2. PAR(1) temporal recursion:
   ```
   z[t,s] = rho_{m,s} * z[t-1,s] + sqrt(1 - rho_{m,s}^2) * e[s]
   ```

3. Map to uniform and invert marginal CDF:
   ```
   u[t,s] = Phi(z[t,s])
   Q[t,s] = F_{m,s}^{-1}(u[t,s])
   ```

4. Enforce non-negativity.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vine_type` | str | `"rvine"` | `"rvine"`, `"cvine"`, or `"dvine"`. Controls the vine tree structure. |
| `family_set` | str or list | `"all"` | Bivariate copula families to consider per edge. |
| `selection_criterion` | str | `"aic"` | `"aic"` or `"bic"` for bivariate family selection. |
| `marginal_method` | str | `"parametric"` | `"parametric"` (BIC-selected gamma/log-normal) or `"empirical"` (Hazen CDF). |
| `log_transform` | bool | `False` | Apply log(Q + offset) before fitting. |
| `offset` | float | `1.0` | Additive offset for log transform. |
| `trunc_level` | int or None | `None` | Vine tree truncation level. `None` fits all trees. |

## Properties Preserved

- Marginal distributions per (month, site)
- Spatial cross-site correlation via vine copula
- Asymmetric tail dependence (when using Clayton, Gumbel, Joe, etc.)
- Heterogeneous pairwise dependence (different copula families per pair)
- Lag-1 temporal autocorrelation via PAR(1)
- Seasonal cycle (implicit in per-month marginals and PAR parameters)

**Not preserved:**

- Higher-order temporal dependence (only lag-1 via PAR(1))
- Long-range dependence / Hurst exponent
- Non-stationarity

## Limitations

- Requires the optional dependency `pyvinecopulib` (`pip install synhydro[vine]`).
- With short records (< 20 years), each month has fewer than 20 observations
  for vine fitting.  Vine structure selection may overfit.  Use `trunc_level`
  and a restricted `family_set` to mitigate.
- Vine complexity grows quadratically with the number of sites.  Practical
  limit is approximately 15-20 sites.

## References

**Primary:**
Yu, X., Xu, Y.-P., Guo, Y., Chen, S., and Gu, H. (2025). Synchronization
frequency analysis and stochastic simulation of multi-site flood flows based
on the complicated vine copula structure. Hydrology and Earth System Sciences,
29, 179-214. https://doi.org/10.5194/hess-29-179-2025

Wang, X., and Shen, Y.-M. (2023). R-statistic based predictor variables
selection and vine structure determination approach for stochastic streamflow
generation. Journal of Hydrology, 617, 129093.
https://doi.org/10.1016/j.jhydrol.2023.129093

Wang, W., Dong, Z., Zhang, T., Ren, L., Xue, L., Wu, T. (2024). Mixed
D-vine copula-based conditional quantile model for stochastic monthly
streamflow simulation. Water Science and Engineering, 17(1), 13-20.
https://doi.org/10.1016/j.wse.2023.05.004

**See also:**
- Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic
  spatial vine copula model for multi-site streamflow simulation. Electric
  Power Systems Research, 152, 9-17.
  https://doi.org/10.1016/j.epsr.2017.06.017
- Czado, C., and Nagler, T. (2022). Vine copula based modeling. Annual Review
  of Statistics and Its Application, 9, 453-477.

---

**Implementation:** `src/synhydro/methods/generation/parametric/vine_copula.py`
**Tests:** `tests/test_vine_copula_generator.py`
