# Matalas (1967) Multi-Site MAR(1)

## Technical Specifications

| Property | Value |
|----------|-------|
| **Class** | `MATALASGenerator` |
| **Type** | Parametric, Stochastic |
| **Frequency** | Monthly |
| **Sites** | Multi-site |
| **Reference** | Matalas (1967) |

## Overview

The Matalas MAR(1) model is the classical parametric baseline for multi-site stochastic
streamflow generation. It extends the univariate Thomas-Fiering model to $n$ sites by
fitting a matrix autoregressive model to the standardized monthly flows. A separate pair
of coefficient matrices is estimated for each of the 12 calendar-month transitions.

## Algorithm Description

### Preprocessing

1. Validate input; resample to monthly if daily data are provided.
2. Clip values to $10^{-6}$ (avoid log of zero).
3. Optionally apply $\log(Q + 1)$ transformation to reduce skewness.

### Fitting

**Standardize** observed flows by monthly means $\mu_m$ and standard deviations $\sigma_m$:

$$Z(t) = \frac{Q(t) - \mu_m}{\sigma_m}, \quad m = \text{month}(t)$$

**Estimate cross-correlation matrices** for each monthly transition $m \to m+1$:

$$S_0(m) = \frac{1}{n-1} \mathbf{Z}(m)^T \mathbf{Z}(m)$$

$$S_1(m) = \frac{1}{n-1} \mathbf{Z}(m+1)^T \mathbf{Z}(m)$$

where rows of $\mathbf{Z}(m)$ are the standardized flow vectors across all sites for month
$m$ across all observed years.

**Solve for coefficient matrices:**

$$A(m) = S_1(m) \cdot S_0(m)^{-1}$$

$$M(m) = S_0(m+1) - A(m) \cdot S_0(m) \cdot A(m)^T$$

$$B(m) = \text{chol}\!\left(M(m)\right) \quad \text{(lower Cholesky factor)}$$

!!! note "Numerical stability"
    If $M(m)$ is not positive semi-definite (due to finite-sample noise), it is first
    projected to the nearest PSD matrix via eigenvalue clipping before Cholesky decomposition.

The December→January transition wraps across the year boundary: $Z$(Dec, year $y$)
is paired with $Z$(Jan, year $y+1$).

### Generation

Initialize $Z_0 \sim \mathcal{N}(\mathbf{0}, I)$, then recurse:

$$Z(t+1) = A(m) \cdot Z(t) + B(m) \cdot \varepsilon(t+1), \quad \varepsilon \sim \mathcal{N}(\mathbf{0}, I)$$

Back-transform to flow space:

$$Q(t) = \sigma_m \cdot Z(t) + \mu_m$$

If `log_transform=True`, apply $Q \leftarrow e^Q - 1$.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_transform` | `True` | Apply $\log(Q+1)$ before standardization |

## Properties Preserved

- Monthly means and standard deviations at each site
- Lag-1 serial correlation at each site
- Contemporaneous cross-site correlations

## References

Matalas, N. C. (1967). Mathematical assessment of synthetic hydrology.
*Water Resources Research*, 3(4), 937–945.

Salas, J. D., Delleur, J. W., Yevjevich, V., & Lane, W. L. (1980).
*Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

**SGLib Implementation:** [`src/sglib/methods/generation/parametric/matalas.py`](https://github.com/Pywr-DRB/SGLib/blob/main/src/sglib/methods/generation/parametric/matalas.py)
