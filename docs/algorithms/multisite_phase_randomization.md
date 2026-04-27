# Multisite Wavelet Phase Randomization (Brunner and Gilleland, 2020)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Daily |
| **Sites** | Multisite |

## Overview

The Multisite Wavelet Phase Randomization generator extends the univariate Fourier phase randomization method of Brunner et al. (2019) to multiple sites using the continuous wavelet transform (CWT). The key innovation is drawing a single set of random phases from a white-noise CWT and applying those shared phases to all sites simultaneously. Because the phase perturbation is identical across sites, the spatial dependence structure of the original record is approximately preserved while the per-site power spectra (wavelet amplitudes) remain intact. The method is well suited for generating long daily streamflow ensembles where both temporal autocorrelation and inter-site synchrony -- such as basin-wide wet or dry spells -- must be reproduced.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_{t,s}$ | Observed daily streamflow at time $t$, site $s$ |
| $\hat{Q}_{t,s}$ | Synthetic daily streamflow at time $t$, site $s$ |
| $N$ | Length of the historical record (days, multiple of 365, no leap days) |
| $S$ | Number of sites |
| $d$ | Day of year (1 to 365) |
| $w$ | Half-window length for marginal fitting (default 15 days) |
| $\bar{Q}_s$ | Global mean of observed flow at site $s$ |
| $y_{t,s}$ | Pre-CWT transformed value of $Q_{t,s}$ (mean-centered or normal-score) |
| $\hat{y}_{t,s}$ | Synthetic transformed value for site $s$ |
| $a$ | CWT scale index |
| $W_s(a, t)$ | CWT coefficient at scale $a$, time $t$, site $s$ |
| $\phi_\varepsilon(a, t)$ | Phase of the white-noise CWT at scale $a$, time $t$ |
| $(\xi_d^s, \alpha_d^s, k_d^s, h_d^s)$ | Four-parameter kappa distribution fitted to day-of-year $d$, site $s$ |

## Formulation

### Model Structure

The generator operates in a transformed domain. For each site $s$, the observed flow series is pre-processed before the wavelet transform using one of two options controlled by the `transform` parameter:

- **`mean_center`** (default): subtract the global site mean, $\tilde{Q}_{t,s} = Q_{t,s} - \bar{Q}_s$. This matches the PRSim reference implementation of Brunner and Gilleland (2020).
- **`normal_score`**: apply the per-day-of-year Van der Waerden normal-score transform (see Parameter Estimation below), producing a more Gaussian-distributed CWT input.

In both cases the transformed series is denoted $y_{t,s}$ and the CWT is computed:

$$
W_s(a, t) = \int_{-\infty}^{\infty} y_{u,s} \, \frac{1}{\sqrt{a}} \psi^*\!\left(\frac{u - t}{a}\right) du
$$

where $\psi$ is the complex Morlet wavelet (pywt identifier `cmor1.5-1.0`, bandwidth 1.5, center frequency 1.0). In the discrete implementation, scales are spaced log-uniformly from 2 to $N/8$ over $n_{\text{scales}} = 100$ values.

Phase randomization replaces the observed phase at each scale and time with a shared random phase:

$$
\hat{W}_s(a, t) = |W_s(a, t)| \cdot e^{i \phi_\varepsilon(a, t)}
$$

where $\phi_\varepsilon(a, t) = \arg\!\left[W_\varepsilon(a, t)\right]$ is derived from the CWT of a single white-noise realization $\varepsilon \sim \mathcal{N}(0, 1)$ of length $N$. Sharing $\phi_\varepsilon$ across all $S$ sites is the mechanism that preserves spatial correlation.

The synthetic normal-score series is recovered by an inverse CWT approximation:

$$
\hat{y}_{t,s} \propto \Delta_j \sum_{a} \frac{\operatorname{Re}\!\left[\hat{W}_s(a, t)\right]}{\sqrt{a}}
$$

where $\Delta_j = \ln(a_{j+1}/a_j)$ is the log-scale spacing (constant for geometrically spaced scales).

### Parameter Estimation

**Marginal distributions.** For each site $s$ and day of year $d$, the four-parameter kappa distribution is fitted by the method of L-moments. All observations within a $\pm w$-day circular window around $d$ (pooled across all years) are used, giving approximately $2w + 1$ pooling days and $N_{\text{yr}} \times (2w + 1)$ samples. The kappa quantile function is:

$$
F^{-1}(p;\, \xi, \alpha, k, h) =
\xi + \frac{\alpha}{k} \left[ 1 - \left(\frac{1 - p^h}{h}\right)^k \right]
$$

with the GEV limit when $h = 0$. Parameters $(k, h)$ are found by minimizing the squared difference between sample and theoretical L-skewness ($\tau_3$) and L-kurtosis ($\tau_4$), and $(\xi, \alpha)$ are then determined analytically from the first two L-moments.

**Normal score transform** (`transform='normal_score'`). For each day $d$ at each site $s$, the $N_{\text{yr}}$ observations are ranked, and the Van der Waerden scores are assigned:

$$
y_{t,s} = \Phi^{-1}\!\left(\frac{r_{t,s}}{N_{\text{yr}} + 1}\right)
$$

where $r_{t,s}$ is the rank of observation $t$ among all observations on day $d$ at site $s$, and $\Phi^{-1}$ is the standard normal quantile function.

**Mean-center transform** (`transform='mean_center'`, default). The global site mean is subtracted:

$$
y_{t,s} = Q_{t,s} - \bar{Q}_s
$$

matching the PRSim reference implementation. The kappa marginal fitting always operates on the raw observed flows $Q_{t,s}$ regardless of which transform is selected.

### Synthesis Procedure

1. Fit kappa distribution parameters $(\xi_d^s, \alpha_d^s, k_d^s, h_d^s)$ for each day $d$ and site $s$ using L-moments over the $\pm w$-day window.

2. Apply the normal score transform to $Q_{t,s}$ per day of year, producing $y_{t,s}$ for each site.

3. Compute the CWT of each normal-score series: $W_s(a, t) = \text{CWT}(y_{\cdot,s},\, a,\, \psi)$.

4. For each realization:

   a. Draw white noise $\varepsilon \sim \mathcal{N}(0, 1)$ of length $N$.

   b. Compute the shared phase field:
   $$\phi_\varepsilon(a, t) = \arg\!\left[\text{CWT}(\varepsilon,\, a,\, \psi)\right]$$

   c. For each site $s$, form synthetic CWT coefficients:
   $$\hat{W}_s(a, t) = |W_s(a, t)| \cdot e^{i\phi_\varepsilon(a, t)}$$

   d. Recover the synthetic normal-score series via the approximate inverse CWT:
   $$\hat{y}_{t,s} = C \, \Delta_j \sum_{a} \frac{\operatorname{Re}\!\left[\hat{W}_s(a, t)\right]}{\sqrt{a}}$$
   where $C$ is a normalization constant chosen so that $\hat{y}_{t,s}$ has unit variance.

   e. Back-transform to original units. For each day $d$ and site $s$, rank $\{\hat{y}_{t,s} : \text{doy}(t) = d\}$, draw $N_{\text{yr}}$ kappa quantiles, and map via rank-order to produce $\hat{Q}_{t,s}$. Enforce non-negativity.

## Statistical Properties

The method preserves, per site, the full spectral envelope (CWT amplitude spectrum) of the normal-score series, which encodes both short-range seasonality and long-range persistence. The back-transformation via the kappa distribution reproduces the daily marginal distributions accurately, including heavy tails, because L-moments are resistant to outliers and the kappa family subsumes GEV, logistic, exponential, and Pareto distributions as special cases.

Spatial correlation is preserved because all sites receive identical phase perturbations. In the wavelet domain, phase at each scale and time jointly determines whether a cross-site event (e.g., a flood pulse) occurs; sharing $\phi_\varepsilon$ keeps these co-occurrence patterns intact. The approach does not impose a parametric spatial model and is therefore robust to non-Gaussian and nonlinear dependence structures.

Higher-order cross-site statistics (e.g., cross-site lag correlations) are not explicitly targeted and will depend on how well the single shared phase field represents the observed joint phase structure across sites.

## Limitations

- The approximate inverse CWT does not guarantee exact reconstruction of the original signal; the round-trip error is acceptable for generating synthetic variability but not for signal decomposition.
- Spatial correlation is preserved on average over realizations, but individual realizations may deviate from the observed correlation matrix, particularly for short records.
- The CWT of a white-noise series does not have a flat spectrum, so the shared phase field is not strictly uniform over scale; this introduces some residual scale-dependence in the inter-site phase coherence.
- The method assumes stationarity; non-stationary trends or shifts in the observed record are embedded in the fitted amplitudes and marginals but not explicitly modeled.
- Computational cost scales as $O(S \cdot n_{\text{scales}} \cdot N)$ per scale (direct-convolution CWT), and grows with scale because the Morlet wavelet support widens proportionally. At daily resolution with 100 scales and records of ~12,000 days, the CWT step dominates fit() runtime (~15 s per site).

## Implementation Notes

The default `transform='mean_center'` matches the PRSim reference implementation (Brunner and Gilleland, 2020). The `transform='normal_score'` option applies the Van der Waerden transform per day of year before the CWT, producing a more Gaussian-distributed spectral input; this approach is the basis of the single-site PRSim formulation (Brunner et al., 2019). In both cases the kappa back-transformation uses rank-order mapping, so the absolute scale of the pre-CWT series does not affect the fitted marginals.

## References

**Primary:**
Brunner, M.I., and Gilleland, E. (2020). Stochastic simulation of streamflow and spatial extremes: a continuous, wavelet-based approach. *Hydrology and Earth System Sciences*, 24, 3967-3982. https://doi.org/10.5194/hess-24-3967-2020

**See also:**
- Brunner, M.I., Bardossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019
- Hosking, J.R.M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38(3), 251-258. https://doi.org/10.1147/rd.383.0251
- Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society, Series B*, 52(1), 105-124.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94. https://doi.org/10.1016/0167-2789(92)90043-8

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/multisite_phase_randomization.py`
