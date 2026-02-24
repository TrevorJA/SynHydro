# Kirsch Monthly Bootstrap Generator

**Classification:** Nonparametric
**Temporal Resolution:** Monthly
**Site Compatibility:** Multisite

---

## Technical Specifications

| Property | Specification |
|----------|---------------|
| Input data | Monthly streamflow, multisite capable, aggregated from any finer resolution |
| Output frequency | MS (month start) |
| Distributional assumption | Nonparametric (empirical) |
| Correlation structure | Spatial (cross-site) and temporal (intra-annual) via Cholesky decomposition |

---

## Algorithm Description

### Preprocessing

1. **Data aggregation to monthly resolution**
   - Group input data by (year, month) and sum
   - Create MultiIndex DataFrame with levels: ['year', 'month']
   - Store as `Qm` with shape (n_years × 12, n_sites)

2. **Optional log-transformation**
   - If `generate_using_log_flow=True`: Apply `log(Qm)` with clipping at 1e-6 to prevent log(0)
   - This improves handling of skewed distributions and prevents bias in back-transformation

### Fitting/Calibration

1. **Compute monthly statistics**
   - For each month m ∈ {1, 2, ..., 12} and site s:
     - `mean_month[m, s]` = mean of all month-m values across years
     - `std_month[m, s]` = standard deviation of all month-m values

2. **Create standardized residuals**
   - For each year y, month m, site s:
     ```
     Z_h[y, m, s] = (Qm[y, m, s] - mean_month[m, s]) / std_month[m, s]
     ```
   - Shape: (n_years, 12, n_sites)

3. **Apply normal score transform** (conditional on log-flow option)
   - If `generate_using_log_flow=True`:
     - For each month m and site s:
       - Sort `Z_h[:, m, s]` to get empirical CDF
       - Compute plotting positions: `pp = (rank - 0.5) / n_years` (Hazen formula)
       - Map to normal quantiles: `nscores = Φ⁻¹(pp)`
       - Store sorted values and normal scores for inverse mapping
       - Create `Y` by interpolating `Z_h` onto normal scores
   - Else: `Y = Z_h`
   - Purpose: Prevents bias from interaction between non-Gaussian residuals and exp() back-transform

4. **Create cross-year shifted matrix Y_prime**
   - Preserves correlations between late months of year y and early months of year y+1
   - Construction (shape: n_years-1 × 12 × n_sites):
     ```
     Y_prime[:, 0:6, :] = Y[:-1, 6:12, :]   # Jul-Dec of year i → Jan-Jun position
     Y_prime[:, 6:12, :] = Y[1:, 0:6, :]     # Jan-Jun of year i+1 → Jul-Dec position
     ```

5. **Compute correlation matrices and Cholesky decomposition** (per site)
   - For each site s:
     - Compute 12×12 correlation matrix from `Y[:, :, s]` (shape: n_years × 12)
       - `corr_s = np.corrcoef(Y[:, :, s].T)`
     - Compute 12×12 correlation matrix from `Y_prime[:, :, s]`
       - `corr_prime_s = np.corrcoef(Y_prime[:, :, s].T)`
     - Apply matrix repair if not positive semi-definite (spectral method by default)
     - Compute Cholesky: `U_site[s] = cholesky(corr_s).T`
     - Compute Cholesky: `U_prime_site[s] = cholesky(corr_prime_s).T`

### Generation

1. **Bootstrap sampling of indices**
   - Generate random year indices `M` with shape (n_years+1, 12)
   - Each `M[i, m]` is a random integer in [0, n_historic_years)
   - Generate separate indices `M_prime` for Y_prime (max index = n_historic_years - 1)

2. **Create bootstrap tensors**
   - For standard tensor:
     ```
     X[i, m, s] = Y[M[i, m], m, s]
     ```
   - For shifted tensor:
     ```
     X_prime[i, m, s] = Y_prime[M_prime[i, m], m, s]
     ```

3. **Apply Cholesky mixing** (per site)
   - For each site s:
     ```
     Z[:, :, s] = X[:, :, s] @ U_site[s]
     Z_prime[:, :, s] = X_prime[:, :, s] @ U_prime_site[s]
     ```
   - This imposes the fitted correlation structure on the bootstrap samples

4. **Combine Z and Z_prime to preserve intra-year correlations**
   - For years 0 to n_years-1:
     ```
     ZC[i, 0:6, :] = Z_prime[i, 6:12, :]    # Late months from shifted
     ZC[i, 6:12, :] = Z[i+1, 6:12, :]       # Late months from regular
     ```
   - This ensures the second half of each synthetic year has proper correlation with the first half

5. **Inverse normal score transform** (conditional)
   - If log-flow option was used:
     - For each month m, site s:
       - Use stored sorted values and normal scores
       - Linear extrapolation at tails for out-of-sample values
       - Map `ZC[:, m, s]` back to original standardized space

6. **Destandardize**
   - For each month m:
     ```
     Q_syn[:, m, :] = ZC[:, m, :] * std_month[m, :] + mean_month[m, :]
     ```

7. **Back-transform from log space** (conditional)
   - If log-flow: `Q_syn = exp(Q_syn)`

8. **Reshape and return**
   - Flatten to (n_years × 12, n_sites) with DatetimeIndex

---

## Key Parameters

- **`generate_using_log_flow`**: Apply log transformation before processing
  - Type: bool
  - Default: `True`
  - Notes: Improves handling of skewed distributions; prevents bias when using normal score transform with exp() back-transform. Set to `False` for symmetric or normally-distributed flows.

- **`matrix_repair_method`**: Method for repairing non-positive-definite correlation matrices
  - Type: str
  - Default: `'spectral'`
  - Options: `'spectral'` (eigenvalue adjustment)
  - Notes: Required when sample correlations yield non-PSD matrices; may cause correlation inflation

---

## Algorithmic Details

### Normal Score Transform (NST)

Applied when generating in log-space to prevent bias. Without NST, standardized residuals that are non-Gaussian would interact poorly with the exp() back-transformation, creating systematic bias.

**Forward transform** (during fitting):
```
For each month m, site s:
  1. Sort Z_h[:, m, s] to get sorted_vals
  2. Compute plotting positions: pp[k] = (k - 0.5) / n_years
  3. Map to normal quantiles: nscores[k] = Φ⁻¹(pp[k])
  4. Store {sorted_vals, nscores} for month-site pair
  5. Y[:, m, s] = interp(Z_h[:, m, s], sorted_vals, nscores)
```

**Inverse transform** (during generation):
```
For each month m, site s:
  1. Retrieve {sorted_vals, nscores} for month-site pair
  2. Extend with linear extrapolation at tails:
     - Lower tail slope: (sorted[1] - sorted[0]) / (nscore[1] - nscore[0])
     - Upper tail slope: (sorted[-1] - sorted[-2]) / (nscore[-1] - nscore[-2])
  3. ZC_orig[:, m, s] = interp(ZC[:, m, s], extended_nscores, extended_sorted)
```

### Cross-Year Correlation Preservation

The Y_prime tensor enables preservation of correlations between successive years (e.g., December to January).

**Construction logic:**
- Y contains standardized flows in calendar order
- Y_prime contains the same data but with 6-month phase shift
- When Y and Y_prime are sampled independently and combined, the result has proper within-year correlation without artificially coupling all 12 months to the same historic year

**Combination in generation:**
- First 6 months of synthetic year i: Use months 7-12 from Y_prime sample i
- Last 6 months of synthetic year i: Use months 7-12 from Y sample i+1
- This creates a "seam" at mid-year where cross-year correlation is preserved

### Matrix Repair for Non-PSD Correlation Matrices

Sample correlation matrices may not be positive semi-definite, especially with limited data.

**Spectral method:**
1. Eigendecomposition: `corr = Q Λ Q^T`
2. Clip negative eigenvalues: `Λ_fixed = max(Λ, ε)` where ε is small positive value
3. Reconstruct: `corr_repaired = Q Λ_fixed Q^T`
4. Rescale to unit diagonal: `corr_repaired[i,i] = 1`

---

## Algorithm Variations

- **Standard (log-space with NST)**: `generate_using_log_flow=True` - Recommended for typical skewed streamflow data
- **Linear-space**: `generate_using_log_flow=False` - Appropriate for symmetric distributions or when log-transform is inappropriate

---

## Implementation Notes

### Computational complexity
- Fitting: O(n_sites × n_years × 12²) for correlation computation and Cholesky decomposition
- Generation: O(n_sites × n_synthetic × 12²) for Cholesky mixing per realization

### Limitations
- Requires complete years (all 12 months present for each year)
- Cannot handle missing months within a year
- Y_prime construction loses one year of data (n_years - 1 available for Y_prime)
- Bootstrap resampling cannot generate values outside observed range (except via extrapolation in NST)

### Special handling
- Negative values after generation: Clipped to minimum observed value per month
- Non-PSD matrices: Automatic repair with user warning about potential correlation inflation
- Index bounds: M_prime indices limited to Y_prime.shape[0] to prevent out-of-bounds access

---

## References

**Primary:**
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: A practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396-406.

**Implementation details:**
- Normal score transform: Prevents bias in log-normal generation
- Matrix repair: Uses spectral method from `synhydro.core.statistics.repair_correlation_matrix`
- Monthly statistics: Computed via `synhydro.core.statistics.compute_monthly_statistics`

---

**SynHydro Implementation:** [`src/synhydro/methods/generation/nonparametric/kirsch.py`](https://github.com/TrevorJA/SynHydro/blob/main/src/synhydro/methods/generation/nonparametric/kirsch.py)
