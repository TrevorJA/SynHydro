# Gaussian / Student-t Copula Generator (Chen et al. 2015; Pereira et al. 2017)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |
| **Class** | `GaussianCopulaGenerator` |

## Overview

Separates the marginal distributions from the dependence structure using
Sklar's theorem.  Per-(month, site) marginals are fitted parametrically
(gamma or log-normal, selected by BIC) or empirically (Hazen plotting
position).  Temporal dependence is captured by a Periodic AR(1) model in
normal-score space (Pereira et al. 2017), and spatial dependence among
the PAR residuals is modelled by a Gaussian or Student-t copula.

The t-copula adds symmetric tail dependence via a degrees-of-freedom
parameter, addressing the Gaussian copula's known zero tail dependence
limitation (Tootoonchi et al. 2022).

## Algorithm

### Preprocessing

1. Validate input data via `_store_obs_data()`; resample daily/weekly to monthly if needed.
2. Optionally apply `log(Q + offset)` transform.

### Fitting

1. **Marginal fitting** (per calendar month, per site):
   - Parametric: fit gamma and log-normal via MLE; select winner by BIC
     (Tootoonchi et al. 2022).  Store shape/scale parameters.
   - Empirical: fit `NormalScoreTransform` (Hazen plotting position).

2. **Probability Integral Transform (PIT):**
   ```
   u[t,s] = F_{m,s}(Q[t,s])      # CDF of fitted marginal
   z[t,s] = Phi^{-1}(u[t,s])     # normal scores
   ```

3. **PAR(1) temporal model** (Pereira et al. 2017, Section 3.1):
   - For each monthly transition m-1 -> m and each site s, estimate
     lag-1 autocorrelation:
     ```
     rho_{m,s} = corr(z[month=m, s], z[month=m-1, s])
     ```
   - Compute PAR residuals (approximately i.i.d.):
     ```
     e[t,s] = (z[t,s] - rho_{m,s} * z[t-1,s]) / sqrt(1 - rho_{m,s}^2)
     ```

4. **Copula correlation matrices** (per month):
   - Compute n_sites x n_sites Pearson correlation of PAR residuals e.
   - Repair via spectral method to ensure positive definiteness.
   - Cholesky decompose: R_m = L_m L_m^T.

5. **t-copula df estimation** (if `copula_type="t"`):
   - Grid search over df in [2, 50] maximising the multivariate-t
     log-likelihood on PAR residuals.

### Generation

For each timestep t (calendar month m):

1. Draw independent innovations:
   - Gaussian: `eps ~ N(0, I_{n_sites})`
   - t-copula: `eps ~ t(df, I_{n_sites})`

2. Impose spatial correlation:
   ```
   e[t] = L_m @ eps
   ```

3. PAR(1) temporal recursion:
   ```
   z[t,s] = rho_{m,s} * z[t-1,s] + sqrt(1 - rho_{m,s}^2) * e[t,s]
   ```

4. Map to uniform:
   - Gaussian: `u[t,s] = Phi(z[t,s])`
   - t-copula: `u[t,s] = T_df(z[t,s])`

5. Inverse marginal CDF:
   ```
   Q[t,s] = F_{m,s}^{-1}(u[t,s])
   ```

6. Enforce non-negativity.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `copula_type` | str | `"gaussian"` | `"gaussian"` or `"t"`. t-copula adds tail dependence. |
| `marginal_method` | str | `"parametric"` | `"parametric"` (BIC-selected gamma/log-normal) or `"empirical"` (Hazen CDF). |
| `log_transform` | bool | `False` | Apply log(Q + offset) before fitting. |
| `offset` | float | `1.0` | Additive offset for log transform. |
| `matrix_repair_method` | str | `"spectral"` | Method for repairing non-PSD correlation matrices. |

## Properties Preserved

- Marginal distributions per (month, site) via parametric or empirical CDF
- Spatial cross-site correlation via copula correlation matrix
- Lag-1 temporal autocorrelation via PAR(1) in normal-score space
- Seasonal cycle (implicit in per-month marginals and PAR parameters)

**Not preserved:**

- Higher-order temporal dependence (only lag-1 via PAR(1))
- Asymmetric tail dependence (Gaussian copula only; t-copula provides symmetric tail dependence)
- Long-range dependence / Hurst exponent (no fractional differencing)

## Limitations

- The Gaussian copula has zero upper and lower tail dependence (Renard and
  Lang, 2007).  Use `copula_type="t"` to model symmetric tail dependence.
- PAR(1) captures only lag-1 persistence.  Higher-order temporal structure
  (e.g., multi-month drought persistence) is not explicitly modelled.
- Vine copulas (Czado and Nagler, 2022) offer more flexible dependence
  structures but are beyond the scope of this implementation.
- With short records (< 20 years per month), parametric marginal fitting
  and copula parameter estimation may be unreliable.

## References

**Primary:**
Genest, C., and Favre, A.-C. (2007). Everything you always wanted to know
about copula modeling but were afraid to ask. Journal of Hydrologic
Engineering, 12(4), 347-368.
https://doi.org/10.1061/(ASCE)1084-0699(2007)12:4(347)

Chen, L., Singh, V.P., Guo, S., Zhou, J., and Zhang, J. (2015). Copula-based
method for multisite monthly and daily streamflow simulation. Journal of
Hydrology, 526, 360-381.
https://doi.org/10.1016/j.jhydrol.2015.05.018

Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic
spatial vine copula model for multi-site streamflow simulation. Electric
Power Systems Research, 152, 9-17.
https://doi.org/10.1016/j.epsr.2017.06.017

**See also:**
- Tootoonchi, F. et al. (2022). Copulas for hydroclimatic analysis: A
  practice-oriented overview. WIREs Water, 9(2), e1579.
- Nelsen, R.B. (2006). An Introduction to Copulas. 2nd ed. Springer.
- Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs
  marges. Publ. Inst. Statist. Univ. Paris, 8, 229-231.

---

**Implementation:** `src/synhydro/methods/generation/parametric/gaussian_copula.py`
**Tests:** `tests/test_gaussian_copula_generator.py`
