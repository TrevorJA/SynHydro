# Thomas-Fiering Seasonal AR(1) (Thomas and Fiering, 1962)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly |
| **Sites** | Univariate |

## Overview

The Thomas-Fiering method generates synthetic monthly streamflow by fitting a first-order autoregressive model with month-specific parameters. A Stedinger-Taylor (1982) lower-bound normalization is applied prior to fitting to improve the Gaussianity of the transformed flows. The model preserves seasonal means, standard deviations, and lag-1 serial correlations while maintaining computational simplicity, making it one of the foundational methods in stochastic hydrology.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{t}$ | Observed monthly streamflow at time $t$ |
| $\hat{Q}_{t}$ | Synthetic monthly streamflow at time $t$ |
| $X_{t}$ | Transformed (normalized) streamflow at time $t$ |
| $m$ | Calendar month index, $m \in \{1, 2, \ldots, 12\}$ |
| $m(t)$ | Calendar month corresponding to time $t$ |
| $\mu_m$ | Mean of transformed flows in month $m$ |
| $\sigma_m$ | Standard deviation of transformed flows in month $m$ |
| $\rho_m$ | Lag-1 serial correlation between months $m-1$ and $m$ |
| $\tau_m$ | Stedinger-Taylor lower bound for month $m$ |
| $\varepsilon_t$ | Independent standard normal variate, $\varepsilon_t \sim \mathcal{N}(0, 1)$ |
| $N$ | Number of complete years in the historical record |

## Formulation

### Stedinger-Taylor Normalization

The observed monthly flows are transformed to approximate normality via a shifted logarithmic transformation. For each month $m$, a lower bound $\tau_m$ is estimated from the sample maximum $Q_m^{\max}$, minimum $Q_m^{\min}$, and median $\tilde{Q}_m$:

$$
\tau_m = \frac{Q_m^{\max} \cdot Q_m^{\min} - \tilde{Q}_m^2}{Q_m^{\max} + Q_m^{\min} - 2\,\tilde{Q}_m}
$$

If $\tau_m < 0$ or $\tau_m \geq Q_m^{\min}$, the estimate is set to zero (reducing to a plain log transform). The forward and inverse transformations are:

$$
X_t = \ln(Q_t - \tau_{m(t)}), \qquad \hat{Q}_t = \exp(\hat{X}_t) + \tau_{m(t)}
$$

### Model Structure

In transformed space, the model follows a seasonal AR(1) process. The synthetic value $\hat{X}_t$ in month $m$ depends on the previous month's value $\hat{X}_{t-1}$ in month $m-1$:

$$
\hat{X}_t = \mu_m + \rho_m \frac{\sigma_m}{\sigma_{m-1}} \left( \hat{X}_{t-1} - \mu_{m-1} \right) + \sigma_m \sqrt{1 - \rho_m^2}\;\varepsilon_t
$$

where $\varepsilon_t \sim \mathcal{N}(0,1)$ are independent innovations. The month indices wrap cyclically ($m = 0$ corresponds to $m = 12$).

### Parameter Estimation

The model requires 36 parameters: 12 monthly means, 12 monthly standard deviations, and 12 lag-1 correlations (plus 12 transformation bounds $\tau_m$).

For each month $m$, let $\{X_m^{(1)}, \ldots, X_m^{(N)}\}$ be the $N$ transformed observations falling in that month. The parameters are estimated as:

$$
\hat{\mu}_m = \frac{1}{N} \sum_{k=1}^{N} X_m^{(k)}, \qquad \hat{\sigma}_m = \sqrt{\frac{1}{N-1} \sum_{k=1}^{N} \left(X_m^{(k)} - \hat{\mu}_m\right)^2}
$$

The lag-1 correlation $\hat{\rho}_m$ is computed as the Pearson correlation between the set of transformed values in month $m-1$ and their paired successors in month $m$, with the December-January transition wrapping across the year boundary.

### Synthesis Procedure

1. Estimate $\tau_m$ and transform the historical record to obtain $\{X_t\}$.
2. Compute $\hat{\mu}_m$, $\hat{\sigma}_m$, and $\hat{\rho}_m$ for all $m$.
3. Initialize: draw $\hat{X}_1 = \hat{\mu}_{m(1)} + \hat{\sigma}_{m(1)} \varepsilon_1$.
4. For each subsequent time step $t = 2, 3, \ldots, T$, generate $\hat{X}_t$ using the AR(1) recursion.
5. Back-transform: $\hat{Q}_t = \exp(\hat{X}_t) + \tau_{m(t)}$.
6. Enforce non-negativity: replace any $\hat{Q}_t < 0$ with the historical monthly minimum.

## Statistical Properties

The model preserves the first two moments (mean and variance) and lag-1 autocorrelation of the transformed flows at the monthly scale. Because the parameters are month-specific, the seasonal cycle of flow statistics is reproduced.

Higher-order autocorrelations (lag $> 1$) are not explicitly modeled; they arise only indirectly through the chain of lag-1 transitions. The log transformation with lower-bound adjustment reduces positive skewness but does not guarantee that the marginal distributions match the historical ones exactly. The method does not model spatial dependence and is restricted to univariate applications.

## Limitations

- Univariate only; cannot represent inter-site dependence.
- First-order memory limits the representation of multi-month drought persistence.
- Assumes stationarity of the underlying process; trends or regime shifts are not captured.
- Requires a minimum of roughly 10 complete years for stable monthly parameter estimates.

## References

**Primary:**
Thomas, H.A., and Fiering, M.B. (1962). Mathematical synthesis of streamflow sequences for the analysis of river basins by simulation. In *Design of Water Resource Systems* (eds. Maass et al.), pp. 459-493. Harvard University Press.

**See also:**
- Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation: 1. Model verification and validation. *Water Resources Research*, 18(4), 909-918. https://doi.org/10.1029/WR018i004p00909

---

**Implementation:** `src/synhydro/methods/generation/parametric/thomas_fiering.py`
