# Phase Randomization (Brunner et al., 2019)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Daily |
| **Sites** | Univariate |

## Overview

Phase randomization generates synthetic daily streamflow by replacing the Fourier phase spectrum with uniform random phases while preserving the amplitude (power) spectrum. Because the power spectrum encodes all second-order temporal structure, this approach maintains both short-range autocorrelation and long-range dependence (the Hurst phenomenon). A four-parameter kappa distribution, fitted per day of year via L-moments, enables the back-transformation to reproduce or extrapolate beyond the observed marginal distributions.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed daily streamflow at time $t$ |
| $\hat{Q}_t$ | Synthetic daily streamflow at time $t$ |
| $X_t$ | Normal score-transformed series |
| $\mathcal{F}\{X\}_k$ | Discrete Fourier Transform of $X$ at frequency index $k$ |
| $A_k$ | Amplitude (modulus) at frequency $k$, $A_k = |\mathcal{F}\{X\}_k|$ |
| $\varphi_k$ | Original phase at frequency $k$ |
| $\varphi_k'$ | Randomized phase, $\varphi_k' \sim \text{Uniform}(-\pi, \pi)$ |
| $N$ | Length of the observed series (in days) |
| $d$ | Day-of-year index, $d \in \{1, \ldots, 365\}$ |
| $\xi, \alpha, \kappa, h$ | Kappa distribution parameters (location, scale, shape 1, shape 2) |
| $\lambda_1, \lambda_2$ | First and second L-moments |
| $\tau_3, \tau_4$ | L-skewness and L-kurtosis ratios |

## Formulation

### Normal Score Transform

For each day of year $d$, the observed values across all years are ranked and mapped to standard normal quantiles:

$$
X_t = \Phi^{-1}\!\left(\frac{r_d(Q_t) - 0.5}{N_d}\right)
$$

where $r_d(\cdot)$ is the rank among the $N_d$ observations for day $d$ and $\Phi^{-1}$ is the standard normal inverse CDF. This produces a series $\{X_t\}$ with approximately $\mathcal{N}(0,1)$ marginals at each calendar day.

### Fourier Decomposition

The Discrete Fourier Transform of the normalized series yields:

$$
\mathcal{F}\{X\}_k = A_k\,e^{i\varphi_k}, \qquad k = 0, 1, \ldots, N-1
$$

The power spectrum $|A_k|^2$ encodes all second-order temporal dependence, including the spectral slope that characterizes long-range persistence.

### Phase Randomization

New phases are drawn independently from a uniform distribution:

$$
\varphi_k' \sim \text{Uniform}(-\pi, \pi), \qquad k = 1, \ldots, \lfloor N/2 \rfloor
$$

The surrogate spectrum is constructed by combining the original amplitudes with the random phases:

$$
\mathcal{F}\{\hat{X}\}_k = A_k\,e^{i\varphi_k'}
$$

The DC component ($k = 0$) is preserved unchanged. Conjugate symmetry is imposed for the negative frequencies to ensure a real-valued inverse:

$$
\mathcal{F}\{\hat{X}\}_{N-k} = \overline{\mathcal{F}\{\hat{X}\}_k}
$$

For even-length signals, the Nyquist component ($k = N/2$) is set to its real modulus. The phase-randomized series in normal space is recovered via the inverse FFT:

$$
\hat{X}_t = \text{Re}\left(\mathcal{F}^{-1}\{\mathcal{F}\{\hat{X}\}\}_t\right)
$$

### Kappa Distribution and Back-Transformation

For each day of year $d$, a four-parameter kappa distribution (Hosking, 1994) is fitted via L-moment matching. The quantile function is:

$$
F^{-1}(u) = \xi + \frac{\alpha}{\kappa}\left[1 - \left(\frac{1 - u^h}{h}\right)^{\!\kappa}\right]
$$

with special cases: when $h = 0$ it reduces to the generalized extreme value (GEV) distribution; when $\kappa = 0$ it further reduces to the Gumbel distribution.

The L-moments are computed from probability weighted moments $b_0, b_1, b_2, b_3$:

$$
\lambda_1 = b_0, \quad \lambda_2 = 2b_1 - b_0, \quad \tau_3 = \frac{2(3b_2 - b_0)}{2b_1 - b_0} - 3, \quad \tau_4 = \frac{5(2(2b_3 - 3b_2) + b_0)}{2b_1 - b_0} + 6
$$

The shape parameters $(\kappa, h)$ are determined by minimizing the squared difference between the sample $(\tau_3, \tau_4)$ and the theoretical L-moment ratios of the kappa distribution. The location and scale parameters $(\xi, \alpha)$ are then derived analytically from $\lambda_1$ and $\lambda_2$.

The back-transformation maps the phase-randomized normal scores to the fitted kappa distribution by rank matching: for each day $d$, the ranks of $\hat{X}_t$ among all values for that day are computed, a kappa sample of the same size is generated and sorted, and the ranks are used to select the corresponding kappa quantiles. If the empirical marginal option is used instead, observed values replace the kappa sample, preventing extrapolation beyond the historical range.

### Synthesis Procedure

1. Remove leap days and construct the day-of-year index.
2. Fit the kappa distribution for each day $d$ (using a moving window of $\pm h$ days).
3. Apply the normal score transform to obtain $\{X_t\}$.
4. Compute the FFT and extract amplitudes $\{A_k\}$.
5. Draw random phases $\{\varphi_k'\}$ and construct the surrogate spectrum.
6. Apply the inverse FFT to obtain $\{\hat{X}_t\}$.
7. Back-transform via rank matching against kappa (or empirical) samples.
8. Enforce non-negativity.

## Statistical Properties

The method preserves the full power spectrum of the normalized series, which in expectation reproduces all temporal autocorrelations including long-range dependence characterized by the Hurst exponent. The seasonal marginal distributions are preserved (or extrapolated via the kappa fit) through the day-of-year back-transformation.

Phase coherence is destroyed by design, meaning the temporal sequencing of events (e.g., the specific ordering of flood peaks within a year) is randomized. Individual autocorrelation values are preserved only in expectation across the ensemble, not in each realization. The method is univariate and does not model spatial dependence.

## Limitations

- Univariate only; no spatial correlation between sites.
- Synthetic series length equals the observed series length (no temporal extrapolation).
- February 29 dates are excluded.
- Kappa distribution fitting may fail for some days of year, requiring fallback to adjacent-day parameters or the empirical distribution.
- Minimum of 2 complete years required; 10+ recommended for stable kappa fits.

## References

**Primary:**
Brunner, M.I., Bardossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019

**See also:**
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94.
- Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society Series B*, 52, 105-124.
- Hosking, J.R.M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38, 251-258.

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/phase_randomization.py`
