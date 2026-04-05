# Grygier-Stedinger Condensed Disaggregation (Grygier and Stedinger, 1988)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual to Monthly |
| **Sites** | Univariate / Multisite |

## Overview

The Grygier-Stedinger method extends the [Valencia-Schaake](valencia_schaake.md) disaggregation framework with two key improvements: a condensed parameter set that reduces the number of estimated parameters from $O(m^2)$ to $O(m)$, improving reliability with short records; and a rigorous conservation correction matrix that ensures sub-period values sum exactly to the aggregate without distorting the conditional covariance structure. This method forms the basis of the widely used SPIGOT software (U.S. Army Corps of Engineers).

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{X}_y \in \mathbb{R}^{m}$ | Vector of sub-period flows for year $y$ |
| $Y_y$ | Aggregate flow, $Y_y = \mathbf{1}^\top \mathbf{X}_y$ |
| $\boldsymbol{\mu}_X$ | Mean vector of sub-period flows |
| $\mu_Y, \sigma_Y^2$ | Mean and variance of the aggregate flow |
| $\mathbf{S}_{XX}$ | Covariance matrix of sub-period flows |
| $\mathbf{S}_{XY}$ | Cross-covariance between sub-periods and aggregate |
| $\mathbf{A}$ | Regression coefficient vector |
| $\mathbf{S}_e$ | Conditional covariance matrix |
| $\mathbf{C}$ | Lower Cholesky factor, $\mathbf{S}_e = \mathbf{C}\mathbf{C}^\top$ |
| $\mathbf{D} \in \mathbb{R}^{m}$ | Conservation correction vector |
| $m$ | Number of sub-periods |
| $\mathbf{1}$ | $m$-vector of ones |

## Formulation

### Model Structure

As in Valencia-Schaake, the sub-period flows are modeled as multivariate normal conditioned on the aggregate:

$$
\mathbf{X} \mid Y \;\sim\; \mathcal{N}\!\left(\boldsymbol{\mu}_X + \mathbf{A}(Y - \mu_Y),\; \mathbf{S}_e\right)
$$

with regression coefficients $\mathbf{A} = \mathbf{S}_{XY} / \sigma_Y^2$ and conditional covariance $\mathbf{S}_e = \mathbf{S}_{XX} - \mathbf{S}_{XY}\mathbf{S}_{XY}^\top / \sigma_Y^2$.

### Condensed Parameterization

Rather than estimating the full $m \times m$ covariance matrix $\mathbf{S}_{XX}$, the condensed approach uses a reduced set of statistics:

- Sub-period means $\mu_{X,j}$ and standard deviations $\sigma_{X,j}$ ($j = 1, \ldots, m$)
- Lag-0 correlations between each sub-period and the aggregate
- Lag-1 serial correlations between consecutive sub-periods

This reduces the parameter count from $m(m+1)/2$ to $O(m)$, making estimation feasible with shorter historical records while retaining the essential first- and second-order structure.

### Conservation Correction

The key innovation is the conservation correction vector $\mathbf{D}$, which distributes the volume discrepancy across sub-periods in a manner that preserves the conditional covariance:

$$
\mathbf{D} = \frac{\mathbf{S}_e\,\mathbf{1}}{\mathbf{1}^\top \mathbf{S}_e\,\mathbf{1}}
$$

The weights in $\mathbf{D}$ are proportional to the conditional covariance of each sub-period with the total, ensuring that sub-periods with greater conditional variance absorb a larger share of the correction. This avoids the covariance distortion introduced by the proportional adjustment used in Valencia-Schaake.

### Synthesis Procedure

For each synthetic aggregate value $Y^{\text{syn}}$:

1. Compute the conditional mean:

$$
\boldsymbol{\mu}_{X|Y} = \boldsymbol{\mu}_X + \mathbf{A}(Y^{\text{syn}} - \mu_Y)
$$

2. Generate uncorrected sub-period values:

$$
\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m), \qquad \mathbf{X}^{\text{raw}} = \boldsymbol{\mu}_{X|Y} + \mathbf{C}\,\mathbf{Z}
$$

3. Compute the volume discrepancy and apply the conservation correction:

$$
\delta = Y^{\text{syn}} - \mathbf{1}^\top \mathbf{X}^{\text{raw}}
$$

$$
\mathbf{X}^{\text{syn}} = \mathbf{X}^{\text{raw}} + \mathbf{D}\,\delta
$$

Since $\mathbf{1}^\top \mathbf{D} = 1$ by construction, this guarantees $\mathbf{1}^\top \mathbf{X}^{\text{syn}} = Y^{\text{syn}}$.

4. Invert the transformation if one was applied (log or Wilson-Hilferty).
5. Enforce non-negativity: if any $X_j^{\text{syn}} < 0$, set to zero and redistribute the deficit proportionally across positive sub-periods.

## Statistical Properties

The method preserves the conditional mean and covariance of sub-period flows given the aggregate. Unlike the Valencia-Schaake proportional adjustment, the conservation correction does not distort the conditional covariance structure, producing statistically consistent disaggregated flows. The aggregate total is preserved exactly by construction.

Monthly means, standard deviations, and lag-1 serial correlations between consecutive sub-periods are approximately preserved through the condensed parameterization. The condensed approach sacrifices complex cross-correlations between non-adjacent sub-periods in exchange for more reliable estimation from short records.

## Limitations

- Multivariate normality assumption remains; strongly skewed distributions require transformation.
- Condensed parameterization may miss complex cross-correlations between non-adjacent sub-periods.
- Conservation correction assumes linear relationships; strong nonlinearity may not be captured.
- Transformation choice (log vs. Wilson-Hilferty) can affect results significantly.
- Does not model inter-annual serial correlations between sub-periods of consecutive years.

## References

**Primary:**
Grygier, J.C., and Stedinger, J.R. (1988). Condensed disaggregation procedures and conservation corrections for stochastic hydrology. *Water Resources Research*, 24(10), 1574-1584. https://doi.org/10.1029/WR024i010p01574

**See also:**
- Valencia, R.D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585. https://doi.org/10.1029/WR009i003p00580
- Stedinger, J.R., and Vogel, R.M. (1984). Disaggregation procedures for generating serially correlated flow vectors. *Water Resources Research*, 20(1), 47-56. https://doi.org/10.1029/WR020i001p00047

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/grygier_stedinger.py`
