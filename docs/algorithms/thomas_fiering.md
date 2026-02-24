# Thomas-Fiering (1962) with Stedinger-Taylor Normalization

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Univariate |
| **Class** | `ThomasFieringGenerator` |

## Overview

The Thomas-Fiering method generates synthetic monthly streamflow using a seasonal AR(1) model with month-specific parameters. The Stedinger-Taylor (1982) normalization applies a lower-bound adjustment before log transformation, reducing skewness and improving the normality assumption. The method preserves monthly means, standard deviations, and lag-1 serial correlations.

## Algorithm

### Preprocessing

1. **Resample to monthly** if input is daily or sub-monthly (sum within each month).
2. **Univariate enforcement** — use first column if multi-site DataFrame is provided.
3. **Stedinger-Taylor normalization** — for each month m:
   - Estimate lower bound:
     ```
     tau_m = (Q_max * Q_min - Q_median^2) / (Q_max + Q_min - 2 * Q_median)
     tau_m = max(tau_m, 0)
     ```
   - Transform: `X_m = log(Q_m - tau_m)`

### Fitting

1. **Monthly statistics** — for each month m, compute from transformed flows X:
   - Mean: `mu_m`
   - Standard deviation: `sigma_m`
2. **Lag-1 serial correlations** — for each transition m to m+1:
   - `rho_m = corr(X_m, X_{m+1})` (Dec-Jan wraps across year boundary)
3. **Store parameters** — 48 total (12 months x 4: mu, sigma, rho, tau).

### Generation

1. **Initialize** first month: `X_1 = mu_1 + epsilon * sigma_1` where `epsilon ~ N(0,1)`.
2. **AR(1) recursion** for each subsequent month:
   ```
   X_m = mu_m + rho_m * (sigma_m / sigma_{m-1}) * (X_{m-1} - mu_{m-1})
         + sqrt(1 - rho_m^2) * sigma_m * epsilon
   ```
3. **Inverse transform**: `Q_m = exp(X_m) + tau_m`
4. **Non-negativity**: clip to observed monthly minimum.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Q_obs` | `pd.DataFrame` | — | Observed streamflow with DatetimeIndex |
| `debug` | `bool` | `False` | Enable debug logging |

## Properties Preserved

- Monthly means and standard deviations (in transformed space)
- Lag-1 serial correlation (month-to-month)
- Seasonal flow patterns (via month-specific parameters)

**Not preserved:**
- Higher-order autocorrelation (lag > 1)
- Spatial correlations (univariate method)
- Exact marginal distributions (normality imposed via transformation)

## Limitations

- Univariate only — no spatial correlation modeling
- First-order memory — misses multi-month drought persistence
- Requires at least 2 complete years; 10+ recommended
- Assumes stationarity (no trends or regime shifts)

## References

**Primary:**
Thomas, H.A., and Fiering, M.B. (1962). Mathematical synthesis of streamflow sequences for the analysis of river basins by simulation. In *Design of Water Resource Systems* (eds. Maass et al.), pp. 459-493. Harvard University Press.

**See also:**
- Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation: 1. Model verification and validation. *Water Resources Research*, 18(4), 909-918. https://doi.org/10.1029/WR018i004p00909

---

**Implementation:** `src/synhydro/methods/generation/parametric/thomas_fiering.py`
**Tests:** `tests/test_thomas_fiering_generator.py`
