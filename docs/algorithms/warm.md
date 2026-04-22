# WARM -- Wavelet Auto-Regressive Method (Nowak et al., 2011)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Univariate |

## Overview

WARM combines continuous wavelet transforms with autoregressive modeling to generate synthetic annual streamflow that preserves non-stationary low-frequency variability. The observed series is decomposed into time-frequency components via the continuous wavelet transform (CWT), and the time-varying energy is captured by the Scale Averaged Wavelet Power (SAWP). After normalizing by SAWP to produce stationary components, independent AR models are fitted at each wavelet scale. Synthetic generation reverses the process: AR-generated coefficients are rescaled by bootstrapped SAWP values and reconstructed via the inverse CWT.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed annual streamflow at year $t$ |
| $\hat{Q}_t$ | Synthetic annual streamflow at year $t$ |
| $W(a_j, t)$ | CWT coefficient at scale $a_j$ and time $t$ |
| $a_j$ | Wavelet scale, $j = 1, \ldots, J$ |
| $J$ | Total number of scales |
| $P(t)$ | Scale Averaged Wavelet Power at time $t$ |
| $\tilde{W}(a_j, t)$ | SAWP-normalized wavelet coefficient |
| $C_\delta$ | Wavelet-specific reconstruction factor (T&C 1998, Table 2) |
| $\delta_j$ | Mean logarithmic (base-2) spacing of scales |
| $\delta_t$ | Sampling period (one year for annual input) |
| $\phi_k^{(a_j)}$ | AR coefficient at lag $k$ for scale $a_j$ |
| $p$ | AR model order |
| $\sigma_{a_j}$ | Innovation standard deviation for scale $a_j$ |
| $\varepsilon_t$ | Independent standard normal innovation |
| $N$ | Number of years in the historical record |

## Formulation

### Continuous Wavelet Transform

The observed annual flow series $\{Q_t\}$ is decomposed using a mother wavelet $\psi$ (default: Morlet) at $J$ discrete scales:

$$
W(a_j, t) = \text{CWT}(Q, a_j), \qquad j = 1, \ldots, J
$$

yielding a coefficient matrix of dimension $J \times N$.

### Scale Averaged Wavelet Power

Following Nowak et al. (2011, Eq. 5) and Torrence and Compo (1998, Eq. 24), the SAWP captures the time-varying energy across all frequency scales as a scale-weighted sum with wavelet-specific normalization:

$$
P(t) = \frac{\delta_j \, \delta_t}{C_\delta} \sum_{j=1}^{J} \frac{|W(a_j, t)|^2}{a_j}
$$

The $1/a_j$ weighting is the scale-density correction from the continuous wavelet transform; $\delta_j$ is the mean logarithmic (base-2) spacing of the scales, $\delta_t$ is the sampling period (one year for annual input), and $C_\delta$ is a wavelet-specific reconstruction factor. For the Morlet wavelet, $C_\delta = 0.776$ (Torrence and Compo 1998, Table 2).

High values of $P(t)$ indicate periods of elevated variability (e.g., regime shifts, climate oscillations).

### Normalization

The wavelet coefficients are normalized by the SAWP to remove time-varying power and produce approximately stationary components:

$$
\tilde{W}(a_j, t) = \frac{W(a_j, t)}{\sqrt{P(t) + \epsilon}}
$$

where $\epsilon$ is a small constant to prevent division by zero.

### Autoregressive Fitting

For each scale $a_j$, the normalized coefficient series $\{\tilde{W}(a_j, t)\}_{t=1}^{N}$ is modeled as an AR($p$) process. Let $\mu_{a_j} = \mathbb{E}[\tilde{W}(a_j, \cdot)]$ denote the mean and $\gamma_k^{(a_j)}$ the autocovariance at lag $k$. The AR coefficients are estimated via the Yule-Walker equations:

$$
\begin{pmatrix} \gamma_0^{(a_j)} & \gamma_1^{(a_j)} & \cdots & \gamma_{p-1}^{(a_j)} \\ \gamma_1^{(a_j)} & \gamma_0^{(a_j)} & \cdots & \gamma_{p-2}^{(a_j)} \\ \vdots & & \ddots & \vdots \\ \gamma_{p-1}^{(a_j)} & \gamma_{p-2}^{(a_j)} & \cdots & \gamma_0^{(a_j)} \end{pmatrix} \begin{pmatrix} \phi_1^{(a_j)} \\ \phi_2^{(a_j)} \\ \vdots \\ \phi_p^{(a_j)} \end{pmatrix} = \begin{pmatrix} \gamma_1^{(a_j)} \\ \gamma_2^{(a_j)} \\ \vdots \\ \gamma_p^{(a_j)} \end{pmatrix}
$$

The innovation variance is:

$$
\sigma_{a_j}^2 = \gamma_0^{(a_j)} \left(1 - \sum_{k=1}^{p} \phi_k^{(a_j)} \cdot \frac{\gamma_k^{(a_j)}}{\gamma_0^{(a_j)}}\right)
$$

### Synthesis Procedure

1. For each scale $a_j$, generate a synthetic normalized coefficient series via the AR($p$) recursion:

$$
\hat{\tilde{W}}(a_j, t) = \mu_{a_j} + \sum_{k=1}^{p} \phi_k^{(a_j)} \left[\hat{\tilde{W}}(a_j, t-k) - \mu_{a_j}\right] + \sigma_{a_j}\,\varepsilon_t
$$

2. Bootstrap SAWP values from the historical record with replacement to obtain $\{\hat{P}(t)\}_{t=1}^{T}$.

3. Rescale the synthetic coefficients by the bootstrapped SAWP:

$$
\hat{W}(a_j, t) = \hat{\tilde{W}}(a_j, t) \cdot \sqrt{\hat{P}(t) + \epsilon}
$$

4. Reconstruct the synthetic flow via inverse CWT:

$$
\hat{Q}_t = c \sum_{j=1}^{J} \frac{1}{\sqrt{a_j}} \,\text{Re}\!\left[\hat{W}(a_j, t)\right]
$$

where $c$ is a normalization constant. The reconstructed series is then rescaled to match the observed mean and standard deviation.

5. Enforce non-negativity: $\hat{Q}_t \leftarrow \max(\hat{Q}_t, 0)$.

## Statistical Properties

The method preserves the mean and variance of the annual flow series (exactly, through post-reconstruction adjustment) and the lag-1 autocorrelation (approximately, through the AR models at each scale). The wavelet decomposition and SAWP bootstrapping capture multi-year persistence and non-stationary variance structure, making WARM well suited for flows influenced by climate oscillations or multi-decadal drought regimes.

The power spectrum is approximately preserved through the amplitude structure of the wavelet coefficients. However, the inverse CWT is approximate and requires a mean/variance correction step. Gaussian AR innovations may underrepresent extreme annual flows, and higher moments (skewness, kurtosis) are not explicitly modeled. The method is univariate and does not model spatial dependence.

## Limitations

- Annual frequency only; for finer resolution, must be coupled with a disaggregation method.
- Univariate; no native multi-site support.
- Inverse CWT is approximate; corrected by matching the first two moments.
- Edge effects in the CWT can degrade quality for short records (fewer than 20 years).
- Gaussian AR innovations may not adequately represent tail behavior.

## References

**Primary:**
Nowak, K., Rajagopalan, B., and Zagona, E. (2011). A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary trends. *Journal of Hydrology*, 410(1-2), 1-12. https://doi.org/10.1016/j.jhydrol.2011.08.049

**See also:**
- Erkyihun, S.T., Rajagopalan, B., Zagona, E., Lall, U., and Nowak, K. (2016). Wavelet-based time series bootstrap model for multidecadal streamflow simulation using climate indicators. *Water Resources Research*, 52(5), 4061-4077. https://doi.org/10.1002/2016WR018696
- Kwon, H.-H., Lall, U., and Khalil, A.F. (2007). Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition. *Water Resources Research*, 43(5).
- Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78.

---

**Implementation:** `src/synhydro/methods/generation/parametric/warm.py`
