# Matalas Multi-Site MAR(1) (Matalas, 1967)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

The Matalas model extends the univariate Thomas-Fiering seasonal AR(1) to multiple sites by fitting a matrix autoregressive process to standardized monthly flows. A separate pair of transition matrices (autoregressive coefficients and innovation structure) is estimated for each of the 12 calendar-month transitions, capturing both temporal persistence and contemporaneous spatial dependence across a network of gauges.

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{Q}_t \in \mathbb{R}^S$ | Observed monthly flow vector at time $t$ across $S$ sites |
| $\hat{\mathbf{Q}}_t$ | Synthetic monthly flow vector at time $t$ |
| $\mathbf{Z}_t \in \mathbb{R}^S$ | Standardized flow vector at time $t$ |
| $m(t)$ | Calendar month corresponding to time $t$, $m \in \{1, \ldots, 12\}$ |
| $\boldsymbol{\mu}_m \in \mathbb{R}^S$ | Vector of site means for month $m$ |
| $\boldsymbol{\sigma}_m \in \mathbb{R}^S$ | Vector of site standard deviations for month $m$ |
| $\mathbf{S}_0^{(m)} \in \mathbb{R}^{S \times S}$ | Lag-0 cross-correlation matrix for month $m$ |
| $\mathbf{S}_1^{(m)} \in \mathbb{R}^{S \times S}$ | Lag-1 cross-correlation matrix (month $m+1$ on month $m$) |
| $\mathbf{A}^{(m)} \in \mathbb{R}^{S \times S}$ | Autoregressive coefficient matrix for the transition from month $m$ to $m+1$ |
| $\mathbf{B}^{(m)} \in \mathbb{R}^{S \times S}$ | Lower Cholesky factor of the innovation covariance |
| $\boldsymbol{\varepsilon}_t \in \mathbb{R}^S$ | Independent standard normal innovation vector |
| $N$ | Number of complete years in the historical record |

## Formulation

### Standardization

An optional log transformation $Q \mapsto \ln(Q + 1)$ may be applied first to reduce skewness. Flows are then standardized by monthly statistics:

$$
Z_{t,s} = \frac{Q_{t,s} - \mu_{m(t),s}}{\sigma_{m(t),s}}, \qquad s = 1, \ldots, S
$$

where $\mu_{m,s}$ and $\sigma_{m,s}$ are the sample mean and standard deviation of site $s$ in month $m$.

### Model Structure

The standardized flow vectors follow a periodic MAR(1) process:

$$
\mathbf{Z}_{t+1} = \mathbf{A}^{(m)} \mathbf{Z}_t + \mathbf{B}^{(m)} \boldsymbol{\varepsilon}_{t+1}, \qquad \boldsymbol{\varepsilon}_{t+1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

where $m = m(t)$ is the calendar month of time $t$ and the month indices wrap cyclically ($m = 12$ transitions to $m = 1$).

### Parameter Estimation

For each month $m$, let $\mathbf{Z}^{(m)}$ denote the $N \times S$ matrix of standardized observations falling in month $m$. The lag-0 and lag-1 cross-correlation matrices are:

$$
\mathbf{S}_0^{(m)} = \frac{1}{N - 1} \left(\mathbf{Z}^{(m)}\right)^\top \mathbf{Z}^{(m)}, \qquad \mathbf{S}_1^{(m)} = \frac{1}{N - 1} \left(\mathbf{Z}^{(m+1)}\right)^\top \mathbf{Z}^{(m)}
$$

The autoregressive coefficient matrix is obtained by:

$$
\mathbf{A}^{(m)} = \mathbf{S}_1^{(m)} \left(\mathbf{S}_0^{(m)}\right)^{-1}
$$

The innovation covariance is the residual after accounting for the autoregressive component:

$$
\mathbf{M}^{(m)} = \mathbf{S}_0^{(m+1)} - \mathbf{A}^{(m)} \mathbf{S}_0^{(m)} \left(\mathbf{A}^{(m)}\right)^\top
$$

$\mathbf{M}^{(m)}$ is symmetrized and, if necessary, repaired to positive semi-definiteness via spectral projection (setting negative eigenvalues to zero). The Cholesky factorization then yields:

$$
\mathbf{B}^{(m)} = \text{chol}(\mathbf{M}^{(m)}), \qquad \mathbf{M}^{(m)} = \mathbf{B}^{(m)} \left(\mathbf{B}^{(m)}\right)^\top
$$

### Synthesis Procedure

1. Initialize $\mathbf{Z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
2. For each time step $t = 0, 1, \ldots, T-1$, with $m = m(t)$:

$$
\mathbf{Z}_{t+1} = \mathbf{A}^{(m)} \mathbf{Z}_t + \mathbf{B}^{(m)} \boldsymbol{\varepsilon}_{t+1}
$$

3. Back-transform to flow space:

$$
\hat{Q}_{t,s} = \sigma_{m(t),s} \cdot Z_{t,s} + \mu_{m(t),s}
$$

4. If a log transformation was applied, invert: $\hat{Q}_{t,s} \leftarrow \exp(\hat{Q}_{t,s}) - 1$, then enforce non-negativity.

## Statistical Properties

The MAR(1) model preserves the first two moments (mean and variance) and the lag-1 autocorrelation at each site, as well as the contemporaneous cross-site correlation structure, all at the monthly scale. The seasonal cycle of these statistics is captured through the 12 sets of month-specific matrices.

Higher-order temporal autocorrelations (lag $> 1$) emerge only indirectly through the chain of first-order transitions and are generally underestimated. The model assumes that the standardized residuals are multivariate Gaussian, which may inadequately represent heavy-tailed or skewed marginal distributions. Long-range persistence (Hurst phenomenon) is not captured.

## Limitations

- First-order memory only; multi-month drought persistence is underrepresented.
- Multivariate Gaussian assumption may not hold for strongly skewed flows.
- Covariance matrices may require positive-definiteness repair when the record is short relative to the number of sites.
- Stationarity is assumed; the model does not accommodate trends or regime shifts.

## References

**Primary:**
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology. *Water Resources Research*, 3(4), 937-945. https://doi.org/10.1029/WR003i004p00937

**See also:**
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/generation/parametric/matalas.py`
