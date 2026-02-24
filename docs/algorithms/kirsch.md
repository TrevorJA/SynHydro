# Kirsch (2013) Monthly Bootstrap Generator

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |
| **Class** | `KirschGenerator` |

## Overview

The Kirsch method generates synthetic multi-site monthly streamflow by bootstrapping standardized residuals and imposing fitted correlation structure via Cholesky decomposition. A cross-year shifted matrix preserves December-to-January correlations. An optional normal score transform prevents bias when working in log-space.

## Algorithm

### Preprocessing

1. **Aggregate to monthly** — group by (year, month) and sum.
2. **Optional log transform** — if `generate_using_log_flow=True`, apply `log(Q)` (clipped at 1e-6).

### Fitting

1. **Monthly statistics** — for each month m and site s, compute mean and standard deviation.
2. **Standardized residuals**:
   ```
   Z_h[y, m, s] = (Q[y, m, s] - mean[m, s]) / std[m, s]
   ```
3. **Normal score transform** (if log-flow) — for each month-site pair:
   - Rank residuals, map to normal quantiles via Hazen plotting positions
   - Store mapping for inverse transform during generation
   - Result: `Y` in standard normal space
4. **Cross-year shifted matrix** `Y_prime` — preserves inter-year correlations:
   ```
   Y_prime[:, 0:6, :]  = Y[:-1, 6:12, :]   # Jul-Dec of year i
   Y_prime[:, 6:12, :] = Y[1:, 0:6, :]      # Jan-Jun of year i+1
   ```
5. **Cholesky decomposition** — for each site, compute 12x12 correlation matrix of Y (and Y_prime), repair if not PSD (spectral method), then Cholesky factor.

### Generation

1. **Bootstrap** — sample random year indices for each (year, month) position.
2. **Cholesky mixing** — multiply bootstrap samples by Cholesky factor to impose correlation:
   ```
   Z[:, :, s] = X[:, :, s] @ U[s]
   ```
3. **Combine** Z and Z_prime to preserve intra-year correlations:
   ```
   ZC[i, 0:6, :]  = Z_prime[i, 6:12, :]    # first half from shifted
   ZC[i, 6:12, :] = Z[i+1, 6:12, :]         # second half from regular
   ```
4. **Inverse normal score transform** (if log-flow) — map back using stored mappings with linear tail extrapolation.
5. **Destandardize**: `Q_syn = ZC * std[m] + mean[m]`
6. **Back-transform** from log space if applicable: `Q_syn = exp(Q_syn)`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_using_log_flow` | `bool` | `True` | Log-transform before processing (recommended for skewed data) |
| `matrix_repair_method` | `str` | `'spectral'` | Method for repairing non-PSD correlation matrices |

## Properties Preserved

- Monthly means and standard deviations (per site)
- Spatial cross-site correlations (via Cholesky decomposition)
- Intra-annual temporal correlation (within and across year boundaries)
- Empirical marginal distributions (nonparametric)

**Not preserved:**
- Values outside observed range (bootstrap limitation, except via NST tail extrapolation)

## Limitations

- Requires complete years (all 12 months present per year)
- Y_prime construction loses one year of data
- Sample correlation matrices may require PSD repair, which can inflate correlations
- Bootstrap resampling bounded by historical range

## References

**Primary:**
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: A practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396-406. https://doi.org/10.1061/(ASCE)WR.1943-5452.0000287

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/kirsch.py`
**Tests:** `tests/test_kirsch_generator.py`
