# Gaussian / Student-t Copula Generator (Chen et al., 2015; Pereira et al., 2017)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Multisite |

## Overview

This method separates marginal distributions from the dependence structure via Sklar's theorem. Per-(month, site) marginals are fitted parametrically (gamma or log-normal, selected by BIC) or empirically. Temporal dependence is captured by a Periodic AR(1) model in normal-score space (Pereira et al., 2017), and spatial dependence among the PAR residuals is modeled by a Gaussian or Student-t copula. The Student-t copula adds symmetric tail dependence through a degrees-of-freedom parameter, addressing the Gaussian copula's known limitation of zero tail dependence.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{t,s}$ | Observed monthly flow at time $t$, site $s$ |
| $\hat{Q}_{t,s}$ | Synthetic monthly flow |
| $m(t)$ | Calendar month at time $t$, $m \in \{1, \ldots, 12\}$ |
| $F_{m,s}$ | Marginal CDF for month $m$, site $s$ |
| $u_{t,s}$ | Uniform score, $u = F_{m,s}(Q_{t,s})$ |
| $z_{t,s}$ | Normal score, $z = \Phi^{-1}(u)$ |
| $\rho_{m,s}$ | PAR(1) lag-1 autocorrelation for month $m$, site $s$ |
| $e_{t,s}$ | PAR(1) residual |
| $\mathbf{R}_m \in \mathbb{R}^{S \times S}$ | Copula correlation matrix for month $m$ |
| $\mathbf{L}_m$ | Lower Cholesky factor, $\mathbf{R}_m = \mathbf{L}_m \mathbf{L}_m^\top$ |
| $\nu$ | Degrees of freedom for the Student-t copula |
| $\Phi, \Phi^{-1}$ | Standard normal CDF and its inverse |
| $T_\nu, T_\nu^{-1}$ | Student-t CDF and its inverse with $\nu$ degrees of freedom |
| $S$ | Number of sites |
| $N$ | Number of complete years |

## Formulation

### Marginal Fitting

For each month $m$ and site $s$, the marginal distribution $F_{m,s}$ is selected from the gamma and log-normal families. Both are fitted by maximum likelihood, and the model with the lower BIC is retained:

$$
\text{BIC} = 2\,\text{NLL} + k \ln n
$$

where NLL is the negative log-likelihood, $k$ is the number of parameters (2 for both gamma and log-normal), and $n$ is the number of observations. Alternatively, an empirical CDF based on Hazen plotting positions may be used.

### Probability Integral Transform and Normal Scores

Observed flows are mapped to uniform variates via the fitted marginal CDF, then transformed to standard normal space:

$$
u_{t,s} = F_{m(t),s}(Q_{t,s}), \qquad z_{t,s} = \Phi^{-1}(u_{t,s})
$$

### Periodic AR(1) Temporal Model

For each monthly transition $m-1 \to m$ and each site $s$, the lag-1 autocorrelation is estimated:

$$
\rho_{m,s} = \text{Corr}(z_{m-1,s},\; z_{m,s})
$$

The PAR(1) residuals, which are approximately temporally independent, are computed as:

$$
e_{t,s} = \frac{z_{t,s} - \rho_{m(t),s}\, z_{t-1,s}}{\sqrt{1 - \rho_{m(t),s}^2}}
$$

### Copula Estimation

For each month $m$, the $S \times S$ Pearson correlation matrix of the PAR residuals is computed:

$$
\mathbf{R}_m = \text{Corr}(\mathbf{e}_m)
$$

If $\mathbf{R}_m$ is not positive definite, it is repaired via spectral projection (negative eigenvalues are set to a small positive value and the matrix is rescaled to unit diagonal). The Cholesky factorization is then computed: $\mathbf{R}_m = \mathbf{L}_m \mathbf{L}_m^\top$.

For the Student-t copula, the degrees of freedom $\nu$ are estimated by maximizing the profile log-likelihood of the multivariate $t$-distribution over a grid $\nu \in \{2, 3, \ldots, 50\}$:

$$
\ell(\nu) = \sum_{t} \left[\ln \Gamma\!\left(\frac{\nu + S}{2}\right) - \ln \Gamma\!\left(\frac{\nu}{2}\right) - \frac{S}{2}\ln(\nu\pi) - \frac{1}{2}\ln|\mathbf{R}_m| - \frac{\nu + S}{2}\ln\!\left(1 + \frac{\mathbf{z}_t^\top \mathbf{R}_m^{-1}\mathbf{z}_t}{\nu}\right)\right]
$$

### Synthesis Procedure

For each time step $t$ with calendar month $m$:

1. Draw independent innovations:
   - Gaussian copula: $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
   - Student-t copula: $\boldsymbol{\varepsilon}_t \sim t_\nu(\mathbf{0}, \mathbf{I})$

2. Impose spatial correlation via the Cholesky factor:

$$
\mathbf{e}_t = \mathbf{L}_m\,\boldsymbol{\varepsilon}_t
$$

3. Apply the PAR(1) temporal recursion for each site $s$:

$$
z_{t,s} = \rho_{m,s}\,z_{t-1,s} + \sqrt{1 - \rho_{m,s}^2}\;\,e_{t,s}
$$

4. Map from normal scores to uniform variates:
   - Gaussian: $u_{t,s} = \Phi(z_{t,s})$
   - Student-t: $u_{t,s} = T_\nu(z_{t,s})$

5. Apply the inverse marginal CDF:

$$
\hat{Q}_{t,s} = F_{m,s}^{-1}(u_{t,s})
$$

6. Enforce non-negativity.

## Statistical Properties

The method preserves marginal distributions at each (month, site) pair through parametric or empirical CDF fitting. Spatial cross-site correlation is captured by the copula correlation matrix applied to PAR residuals. The PAR(1) model preserves lag-1 temporal autocorrelation in normal-score space, and the seasonal cycle is implicit in the month-specific marginals and correlation parameters.

The Gaussian copula has zero tail dependence, meaning it underestimates the probability of joint extreme events. The Student-t copula provides symmetric tail dependence controlled by $\nu$, with stronger tail dependence as $\nu$ decreases. Neither copula captures asymmetric tail dependence (e.g., stronger co-occurrence of droughts than of floods). Higher-order temporal structure and long-range persistence are not modeled.

## Limitations

- The Gaussian copula has zero upper and lower tail dependence.
- PAR(1) captures only lag-1 temporal persistence; multi-month drought dynamics are not explicitly modeled.
- Parametric marginal fitting and copula estimation may be unreliable with fewer than 20 years of data.
- The Student-t copula provides only symmetric tail dependence; for asymmetric tail dependence, consider the [vine copula](vine_copula.md) extension.

## References

**Primary:**
Genest, C., and Favre, A.-C. (2007). Everything you always wanted to know about copula modeling but were afraid to ask. *Journal of Hydrologic Engineering*, 12(4), 347-368. https://doi.org/10.1061/(ASCE)1084-0699(2007)12:4(347)

Chen, L., Singh, V.P., Guo, S., Zhou, J., and Zhang, J. (2015). Copula-based method for multisite monthly and daily streamflow simulation. *Journal of Hydrology*, 526, 360-381. https://doi.org/10.1016/j.jhydrol.2015.05.018

Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic spatial vine copula model for multi-site streamflow simulation. *Electric Power Systems Research*, 152, 9-17. https://doi.org/10.1016/j.epsr.2017.06.017

**See also:**
- Tootoonchi, F. et al. (2022). Copulas for hydroclimatic analysis: A practice-oriented overview. *WIREs Water*, 9(2), e1579.
- Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges. *Publ. Inst. Statist. Univ. Paris*, 8, 229-231.

---

**Implementation:** `src/synhydro/methods/generation/parametric/gaussian_copula.py`
