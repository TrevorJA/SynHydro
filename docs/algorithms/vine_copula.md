# Vine Copula Generator (Yu et al., 2025; Wang and Shen, 2023)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

The vine copula generator extends the PAR(1)-plus-copula framework of the [Gaussian copula method](gaussian_copula.md) by replacing the single elliptical copula with a vine copula. A vine copula decomposes the multivariate dependence structure into a hierarchy of bivariate copulas arranged on a sequence of trees, allowing each pair of variables to be modeled by a different copula family. This enables heterogeneous pairwise dependence and asymmetric tail dependence, which the Gaussian and Student-t copulas cannot represent.

Marginal fitting and temporal modeling are identical to the Gaussian copula approach: per-(month, site) parametric or empirical marginals with a Periodic AR(1) temporal model (Pereira et al., 2017). Spatial dependence among the PAR(1) residuals is captured by a monthly-periodic vine copula fitted via the Dissmann algorithm.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{t,s}$ | Observed monthly flow at time $t$, site $s$ |
| $\hat{Q}_{t,s}$ | Synthetic monthly flow |
| $m(t)$ | Calendar month at time $t$ |
| $F_{m,s}$ | Marginal CDF for month $m$, site $s$ |
| $z_{t,s}$ | Normal score, $z = \Phi^{-1}(F_{m,s}(Q_{t,s}))$ |
| $\rho_{m,s}$ | PAR(1) lag-1 autocorrelation for month $m$, site $s$ |
| $e_{t,s}$ | PAR(1) residual |
| $\mathbf{u}_t \in [0,1]^S$ | Uniform vector of transformed PAR residuals |
| $C_m$ | Vine copula for month $m$ |
| $c_{ij;\mathbf{D}}$ | Bivariate copula density for the pair $(i, j)$ conditional on set $\mathbf{D}$ |
| $S$ | Number of sites |
| $N$ | Number of complete years |

## Formulation

### Marginals, PIT, and PAR(1)

The first three stages are identical to the Gaussian copula method. For each (month, site), a marginal distribution $F_{m,s}$ is fitted (gamma vs. log-normal by BIC, or empirical). The probability integral transform and normal score mapping yield:

$$
z_{t,s} = \Phi^{-1}(F_{m(t),s}(Q_{t,s}))
$$

The PAR(1) residuals are:

$$
e_{t,s} = \frac{z_{t,s} - \rho_{m(t),s}\,z_{t-1,s}}{\sqrt{1 - \rho_{m(t),s}^2}}
$$

### Vine Copula Structure

A vine copula factorizes the $S$-dimensional copula density into a product of bivariate copula densities arranged on a sequence of $S - 1$ trees $T_1, \ldots, T_{S-1}$. At tree level $\ell$, each edge connects two nodes that share $\ell - 1$ conditioning variables:

$$
c(\mathbf{u}) = \prod_{\ell=1}^{S-1} \prod_{(i,j;\mathbf{D}) \in T_\ell} c_{ij;\mathbf{D}}\!\left(u_{i|\mathbf{D}},\; u_{j|\mathbf{D}}\right)
$$

where $u_{i|\mathbf{D}}$ denotes the conditional distribution of variable $i$ given the conditioning set $\mathbf{D}$, evaluated via sequential application of the $h$-function (conditional CDF of a bivariate copula).

Three vine structures are supported:

- **R-vine**: Most general; the tree structure is selected automatically by the Dissmann algorithm, which greedily maximizes pairwise dependence at each tree level.
- **C-vine**: Star structure with a single central variable at each tree level.
- **D-vine**: Linear chain structure.

### Bivariate Family Selection

At each edge of the vine, a bivariate copula family is selected from a configurable set (e.g., Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1, BB6, BB7, BB8, independence) by minimizing AIC or BIC. This allows different types of dependence for different pairs: for example, Clayton copulas for lower tail dependence between drought-prone sites and Gumbel copulas for upper tail dependence between flood-prone sites.

### Vine Copula Fitting

For each calendar month $m$, the PAR(1) residuals are transformed to uniform variates:

$$
u_{t,s} = \Phi(e_{t,s})
$$

A vine copula $C_m$ is then fitted to $\{\mathbf{u}_t : m(t) = m\}$ using the Dissmann sequential algorithm, which proceeds tree by tree, selecting the structure and bivariate families that best describe the data at each level. An optional truncation level limits the number of trees, reducing complexity for high-dimensional problems.

### Synthesis Procedure

For each time step $t$ with calendar month $m$:

1. Simulate from the vine copula:

$$
\mathbf{u}_t^{\text{new}} = C_m.\text{simulate}(1), \qquad \mathbf{u}_t^{\text{new}} \in [0,1]^S
$$

and convert to normal-space PAR residuals: $e_{t,s} = \Phi^{-1}(u_{t,s}^{\text{new}})$.

2. Apply the PAR(1) temporal recursion:

$$
z_{t,s} = \rho_{m,s}\,z_{t-1,s} + \sqrt{1 - \rho_{m,s}^2}\;\,e_{t,s}
$$

3. Map to uniform and invert the marginal CDF:

$$
\hat{Q}_{t,s} = F_{m,s}^{-1}(\Phi(z_{t,s}))
$$

4. Enforce non-negativity.

## Statistical Properties

The vine copula preserves marginal distributions at each (month, site) pair, spatial cross-site dependence with potentially heterogeneous and asymmetric tail behavior, and lag-1 temporal autocorrelation through the PAR(1) model. The seasonal cycle is captured through month-specific marginals, PAR parameters, and vine copula structures.

Unlike the Gaussian copula, the vine copula can represent asymmetric tail dependence (e.g., stronger co-occurrence of low flows than of high flows) through appropriate bivariate family selection. Each pair of sites can exhibit a different type of dependence structure. However, higher-order temporal dependence and long-range persistence are not modeled.

## Limitations

- Requires the optional dependency `pyvinecopulib`.
- With short records (fewer than 20 years), each month has fewer than 20 observations for vine fitting, increasing the risk of overfitting. Truncation and restricted family sets help mitigate this.
- Vine complexity grows quadratically with the number of sites; practical limit is approximately 15-20 sites.
- Higher-order temporal dependence and long-range persistence are not captured.

## References

**Primary:**
Yu, X., Xu, Y.-P., Guo, Y., Chen, S., and Gu, H. (2025). Synchronization frequency analysis and stochastic simulation of multi-site flood flows based on the complicated vine copula structure. *Hydrology and Earth System Sciences*, 29, 179-214. https://doi.org/10.5194/hess-29-179-2025

Wang, X., and Shen, Y.-M. (2023). R-statistic based predictor variables selection and vine structure determination approach for stochastic streamflow generation. *Journal of Hydrology*, 617, 129093. https://doi.org/10.1016/j.jhydrol.2023.129093

**See also:**
- Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic spatial vine copula model for multi-site streamflow simulation. *Electric Power Systems Research*, 152, 9-17. https://doi.org/10.1016/j.epsr.2017.06.017
- Czado, C., and Nagler, T. (2022). Vine copula based modeling. *Annual Review of Statistics and Its Application*, 9, 453-477.

---

**Implementation:** `src/synhydro/methods/generation/parametric/vine_copula.py`
