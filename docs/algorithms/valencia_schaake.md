# Valencia-Schaake Temporal Disaggregation (Valencia and Schaake, 1973)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual to Monthly |
| **Sites** | Univariate / Multisite |

## Overview

The Valencia-Schaake method is the foundational parametric temporal disaggregation approach in stochastic hydrology. It disaggregates an aggregate flow volume (e.g., annual total) into sub-period values (e.g., 12 monthly flows) by modeling the sub-periods as a multivariate normal distribution conditioned on the known aggregate. A linear regression relates the sub-period vector to the aggregate, and the conditional covariance captures the remaining uncertainty. This is the classical baseline against which subsequent disaggregation methods, including the [Grygier-Stedinger](grygier_stedinger.md) extension, are compared.

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{X}_y \in \mathbb{R}^{m}$ | Vector of sub-period flows for year $y$, $X_{y,j}$ is the flow in sub-period $j$ |
| $Y_y$ | Aggregate (annual) flow for year $y$, $Y_y = \mathbf{1}^\top \mathbf{X}_y$ |
| $\boldsymbol{\mu}_X \in \mathbb{R}^{m}$ | Mean vector of sub-period flows |
| $\mu_Y$ | Mean of the aggregate flow |
| $\sigma_Y^2$ | Variance of the aggregate flow |
| $\mathbf{S}_{XX} \in \mathbb{R}^{m \times m}$ | Covariance matrix of sub-period flows |
| $\mathbf{S}_{XY} \in \mathbb{R}^{m}$ | Cross-covariance between sub-periods and aggregate |
| $\mathbf{A} \in \mathbb{R}^{m}$ | Regression coefficient vector |
| $\mathbf{S}_e \in \mathbb{R}^{m \times m}$ | Conditional covariance of sub-periods given the aggregate |
| $\mathbf{C}$ | Lower Cholesky factor, $\mathbf{S}_e = \mathbf{C}\mathbf{C}^\top$ |
| $m$ | Number of sub-periods (e.g., 12 months) |
| $N$ | Number of complete years in the historical record |
| $\mathbf{1}$ | $m$-vector of ones |

## Formulation

### Model Structure

The sub-period vector $\mathbf{X}$ and the aggregate $Y = \mathbf{1}^\top \mathbf{X}$ are assumed to follow a joint multivariate normal distribution. The conditional distribution of $\mathbf{X}$ given $Y$ is:

$$
\mathbf{X} \mid Y \;\sim\; \mathcal{N}\!\left(\boldsymbol{\mu}_{X|Y},\; \mathbf{S}_e\right)
$$

where the conditional mean is:

$$
\boldsymbol{\mu}_{X|Y} = \boldsymbol{\mu}_X + \mathbf{A}(Y - \mu_Y)
$$

### Parameter Estimation

The sub-period covariance and aggregate statistics are estimated from the historical record. The cross-covariance between sub-periods and the aggregate is:

$$
\mathbf{S}_{XY} = \text{Cov}(\mathbf{X}, Y) = \mathbf{S}_{XX}\,\mathbf{1}
$$

The regression coefficient vector is:

$$
\mathbf{A} = \frac{\mathbf{S}_{XY}}{\sigma_Y^2}
$$

The conditional covariance matrix is:

$$
\mathbf{S}_e = \mathbf{S}_{XX} - \frac{\mathbf{S}_{XY}\,\mathbf{S}_{XY}^\top}{\sigma_Y^2}
$$

If $\mathbf{S}_e$ is not positive semi-definite, it is repaired via spectral projection. The Cholesky factorization $\mathbf{S}_e = \mathbf{C}\mathbf{C}^\top$ is then computed.

An optional transformation (log or Box-Cox) may be applied to the sub-period flows before fitting to improve the normality assumption.

### Synthesis Procedure

For each synthetic aggregate value $Y^{\text{syn}}$:

1. Compute the conditional mean:

$$
\boldsymbol{\mu}_{X|Y} = \boldsymbol{\mu}_X + \mathbf{A}(Y^{\text{syn}} - \mu_Y)
$$

2. Sample from the conditional distribution:

$$
\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_m), \qquad \mathbf{X}^{\text{syn}} = \boldsymbol{\mu}_{X|Y} + \mathbf{C}\,\mathbf{Z}
$$

3. Apply a proportional adjustment to enforce volume conservation:

$$
X_j^{\text{adj}} = X_j^{\text{syn}} \cdot \frac{Y^{\text{syn}}}{\displaystyle\sum_{k=1}^{m} X_k^{\text{syn}}}
$$

4. Invert the transformation if one was applied.
5. Enforce non-negativity.

Note: The proportional adjustment ensures exact volume conservation but distorts the conditional covariance structure. The [Grygier-Stedinger](grygier_stedinger.md) method provides a correction that preserves this structure.

## Statistical Properties

The method preserves the conditional mean and covariance of sub-period flows given the aggregate, the cross-correlations among sub-periods, and the aggregate total (exactly, after proportional adjustment). Monthly means and standard deviations are approximately preserved.

However, the proportional adjustment introduces distortion into the conditional covariance. The multivariate normal assumption may be violated for strongly skewed flow distributions, particularly at the monthly or daily scale. Inter-annual serial correlations between sub-periods of consecutive years are not modeled.

## Limitations

- Multivariate normality assumption may be violated for skewed sub-period distributions.
- Proportional adjustment distorts the conditional covariance (see Grygier-Stedinger for improvement).
- Does not model serial correlations between sub-periods of consecutive years.
- Full covariance estimation requires long records relative to $m$.
- Better suited for annual-to-monthly than monthly-to-daily disaggregation.

## References

**Primary:**
Valencia, R.D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585. https://doi.org/10.1029/WR009i003p00580

**See also:**
- Stedinger, J.R., and Vogel, R.M. (1984). Disaggregation procedures for generating serially correlated flow vectors. *Water Resources Research*, 20(1), 47-56. https://doi.org/10.1029/WR020i001p00047
- Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/valencia_schaake.py`
