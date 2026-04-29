# Kirsch Monthly Bootstrap (Kirsch et al., 2013)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

The Kirsch method generates synthetic multi-site monthly streamflow by bootstrapping standardized residuals and imposing fitted temporal and spatial correlation structure through Cholesky decomposition. A cross-year shifted matrix construction preserves December-to-January continuity. An optional normal score transform reduces bias when operating in log-transformed space. The method is nonparametric in that the generated values are drawn from the historical record rather than from a parametric distribution.

### Generalization note

Kirsch et al. (2013) describe the method on weekly timesteps (52 columns per year, 26-week halves for the cross-year shift). SynHydro implements it on monthly timesteps (12 columns per year, 6-month halves). The mathematical structure is identical; only the column dimension and the half-year split change. The implementation is straightforwardly generalizable to any sub-annual period count $n$, with the half split at $n/2$.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{y,m,s}$ | Observed monthly flow for year $y$, month $m$, site $s$ |
| $\hat{Q}_{y,m,s}$ | Synthetic monthly flow |
| $\mu_{m,s}$ | Sample mean of flows at site $s$ in month $m$ |
| $\sigma_{m,s}$ | Sample standard deviation of flows at site $s$ in month $m$ |
| $Z_{y,m,s}$ | Standardized residual |
| $Y_{y,m,s}$ | Normal score-transformed residual |
| $\mathbf{Y}^{(s)} \in \mathbb{R}^{N \times 12}$ | Matrix of normal scores for site $s$ (years by months) |
| $\mathbf{Y}'^{(s)}$ | Cross-year shifted matrix for site $s$ |
| $\mathbf{R}^{(s)}, \mathbf{R}'^{(s)}$ | $12 \times 12$ correlation matrices of $\mathbf{Y}^{(s)}$ and $\mathbf{Y}'^{(s)}$ |
| $\mathbf{U}^{(s)}$ | Upper Cholesky factor, $\mathbf{R}^{(s)} = (\mathbf{U}^{(s)})^\top \mathbf{U}^{(s)}$ |
| $N$ | Number of complete years in the record |
| $S$ | Number of sites |

## Formulation

### Standardization and Normal Score Transform

Monthly flows are first (optionally) log-transformed: $Q' = \ln(\max(Q, 10^{-6}))$. Standardized residuals are computed for each year $y$, month $m$, and site $s$:

$$
Z_{y,m,s} = \frac{Q'_{y,m,s} - \mu_{m,s}}{\sigma_{m,s}}
$$

When operating in log space, a normal score transform (NST) is applied to each $(m, s)$ pair. The residuals are ranked, mapped to Hazen plotting positions $p_i = (i - 0.5)/n$, and converted to standard normal quantiles:

$$
Y_{y,m,s} = \Phi^{-1}\!\left(\frac{r(Z_{y,m,s}) - 0.5}{N}\right)
$$

where $r(\cdot)$ denotes the rank among the $N$ values and $\Phi^{-1}$ is the standard normal inverse CDF.

**Note:** The normal score transform is a SynHydro-specific extension to the original Kirsch (2013) method. Kirsch (2013, eq. 3) only applies z-score standardization. NST is added in log-space here to prevent bias in the back-transformed marginal distribution when standardized log-residuals are non-Gaussian; the inverse NST (with linear tail extrapolation) maps Cholesky-mixed values back to the empirical $(m, s)$ marginal. To run the algorithm closer to the published version, set `generate_using_log_flow=False` (which also skips NST).

### Cross-Year Shifted Matrix

To preserve inter-annual correlation (particularly the December-January transition), a shifted matrix $\mathbf{Y}'^{(s)}$ is constructed for each site $s$:

$$
\mathbf{Y}'^{(s)}_{y,\,1:6} = \mathbf{Y}^{(s)}_{y,\,7:12}, \qquad \mathbf{Y}'^{(s)}_{y,\,7:12} = \mathbf{Y}^{(s)}_{y+1,\,1:6}
$$

This re-indexes the data so that July-December of year $y$ is paired with January-June of year $y+1$, enabling the Cholesky factor to capture cross-year correlations.

### Cholesky Decomposition

For each site $s$, the $12 \times 12$ sample correlation matrices $\mathbf{R}^{(s)}$ and $\mathbf{R}'^{(s)}$ are computed from $\mathbf{Y}^{(s)}$ and $\mathbf{Y}'^{(s)}$ respectively. If either matrix is not positive definite, it is repaired via spectral projection (negative eigenvalues are clipped to a small positive constant and the matrix is rescaled to unit diagonal). The upper Cholesky factors $\mathbf{U}^{(s)}$ and $\mathbf{U}'^{(s)}$ are then computed.

Per Kirsch et al. (2013, eqs. 4-5), the correlation matrix and its Cholesky factor are intra-annual operators defined for a single site at a time. Cross-site correlation is preserved through the **shared bootstrap index matrix $\mathbf{M}$** (Kirsch et al. 2013, p. 7), not through a joint multi-site correlation matrix. This implementation therefore computes per-site Cholesky factors $\mathbf{U}^{(s)}, \mathbf{U}'^{(s)}$ and reuses one $\mathbf{M}$ across all sites in the synthesis step below.

### Synthesis Procedure

1. **Bootstrap**: For each synthetic realization, draw a single matrix $\mathbf{M} \in \{1, \dots, N\}^{(N_{\text{syn}}+1) \times 12}$ of year indices sampled with replacement from the historical record. The same $\mathbf{M}$ is reused across all sites so that cross-site correlation is preserved (Kirsch et al. 2013, p. 7). Construct the bootstrap matrix $\mathbf{X}^{(s)}$ by extracting the normal scores of site $s$ at the sampled indices. Construct the corresponding shifted matrix $\mathbf{X}'^{(s)}$ **deterministically** from $\mathbf{X}^{(s)}$ using the same 6-month shift that produced $\mathbf{Y}'^{(s)}$ from $\mathbf{Y}^{(s)}$:

$$
\mathbf{X}'^{(s)}_{y,\,1:6} = \mathbf{X}^{(s)}_{y,\,7:12}, \qquad \mathbf{X}'^{(s)}_{y,\,7:12} = \mathbf{X}^{(s)}_{y+1,\,1:6}
$$

Following Kirsch et al. (2013, p. 6): "The matrix X is converted to X' just as Y was converted to Y'". There is exactly one bootstrap draw per realization; $\mathbf{X}'^{(s)}$ is not resampled independently.

2. **Impose correlation**: Multiply each bootstrap matrix by its Cholesky factor:

$$
\tilde{\mathbf{Z}}^{(s)} = \mathbf{X}^{(s)} \mathbf{U}^{(s)}, \qquad \tilde{\mathbf{Z}}'^{(s)} = \mathbf{X}'^{(s)} \mathbf{U}'^{(s)}
$$

3. **Combine shifted and unshifted results**: The final correlated matrix $\mathbf{Z}_C$ takes the first half of each year from the shifted result and the second half from the unshifted result:

$$
\mathbf{Z}_{C,y,1:6,s} = \tilde{\mathbf{Z}}'_{y,7:12,s}, \qquad \mathbf{Z}_{C,y,7:12,s} = \tilde{\mathbf{Z}}_{y+1,7:12,s}
$$

4. **Inverse normal score transform**: Map back from normal space to the original residual space using the stored rank mappings, with linear extrapolation in the tails for values outside the historical range.

5. **Destandardize and back-transform**:

$$
\hat{Q}'_{y,m,s} = Z_{C,y,m,s} \cdot \sigma_{m,s} + \mu_{m,s}, \qquad \hat{Q}_{y,m,s} = \exp(\hat{Q}'_{y,m,s})
$$

## Statistical Properties

The method preserves monthly means and standard deviations at each site (by construction through standardization and destandardization), spatial cross-site correlations (all sites share the same bootstrap indices for each year), and intra-annual temporal correlation (through the Cholesky decomposition of the 12-month correlation matrix). The cross-year shifted matrix construction maintains continuity across the December-January boundary.

Because the method resamples from the historical record, the empirical marginal distribution is approximately preserved. The normal score transform and its inverse allow modest extrapolation beyond the observed range in the tails. However, the generated values remain close to the historical envelope, and genuinely novel extremes cannot be produced.

## Limitations

- Generated values are bounded near the historical range (bootstrap limitation).
- Requires complete years; the cross-year shifted matrix loses one year of data.
- Sample correlation matrices may need positive-definiteness repair, which can inflate apparent correlations.
- The method does not model long-range persistence or nonstationarity.

## References

**Primary:**
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: A practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396-406. https://doi.org/10.1061/(ASCE)WR.1943-5452.0000287

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/kirsch.py`
