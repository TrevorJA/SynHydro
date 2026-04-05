# ARFIMA -- Autoregressive Fractionally Integrated Moving Average (Hosking, 1984)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate |

## Overview

The ARFIMA(p, d, q) model extends classical ARMA by allowing the differencing parameter $d$ to take fractional values in $(0, 0.5)$. This enables the model to reproduce the long-range dependence (Hurst phenomenon) observed in many hydrologic time series, where the autocorrelation function decays hyperbolically rather than exponentially. The short-memory ARMA(p, q) component captures local temporal structure, while the fractional integration parameter $d$ governs the rate of low-frequency spectral divergence. The relationship $H = d + 0.5$ links the model directly to the Hurst exponent.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed streamflow at time $t$ |
| $\hat{Q}_t$ | Synthetic streamflow at time $t$ |
| $X_t$ | Deseasonalized (stationary) streamflow at time $t$ |
| $W_t$ | Fractionally differenced series |
| $d$ | Fractional differencing parameter, $d \in (0, 0.5)$ |
| $H$ | Hurst exponent, $H = d + 0.5$ |
| $p, q$ | Orders of the AR and MA components |
| $\phi_k$ | AR coefficients, $k = 1, \ldots, p$ |
| $\theta_k$ | MA coefficients, $k = 1, \ldots, q$ |
| $\varepsilon_t$ | White noise innovation, $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$ |
| $\pi_k$ | Fractional differencing coefficients (forward) |
| $\psi_k$ | Inverse fractional differencing coefficients |
| $B$ | Backshift operator, $B X_t = X_{t-1}$ |
| $K$ | Truncation lag for the infinite coefficient series |
| $\mu_m, \sigma_m$ | Monthly mean and standard deviation (for deseasonalization) |
| $N$ | Length of the observed record |

## Formulation

### Model Structure

The ARFIMA(p, d, q) process is defined by:

$$
\Phi(B)\,(1 - B)^d\,X_t = \Theta(B)\,\varepsilon_t
$$

where $\Phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ is the AR polynomial, $\Theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ is the MA polynomial, and $(1 - B)^d$ is the fractional differencing operator. The process is stationary and invertible when $0 < d < 0.5$ and the roots of $\Phi$ and $\Theta$ lie outside the unit circle.

The fractional differencing operator is expanded as an infinite-order filter:

$$
(1 - B)^d = \sum_{k=0}^{\infty} \pi_k\,B^k
$$

with coefficients defined recursively:

$$
\pi_0 = 1, \qquad \pi_k = \pi_{k-1} \cdot \frac{k - 1 - d}{k}, \quad k \geq 1
$$

In practice, the sum is truncated at lag $K$ (default 100). Application of this filter yields the fractionally differenced series:

$$
W_t = \sum_{k=0}^{K} \pi_k\,X_{t-k}
$$

which, under the ARFIMA model, follows a stationary ARMA(p, q) process.

### Deseasonalization

For monthly data, seasonal nonstationarity is removed before fitting. Let $\mu_m$ and $\sigma_m$ denote the sample mean and standard deviation for month $m$. The deseasonalized series is:

$$
X_t = \frac{Q_t - \mu_{m(t)}}{\sigma_{m(t)}}
$$

### Parameter Estimation

#### Estimation of $d$

Three frequency-domain estimators are available:

**Whittle estimator** (Fox and Taqqu, 1986). The spectral density of the fractional differencing component is:

$$
f(\omega;\,d) \propto \left[2(1 - \cos \omega)\right]^{-d}
$$

The Whittle likelihood is minimized over $d \in [0.01,\,0.49]$:

$$
\hat{d} = \arg\min_d \sum_{j=1}^{n/2} \left[\ln f(\omega_j;\,d) + \frac{I(\omega_j)}{f(\omega_j;\,d)}\right]
$$

where $I(\omega_j)$ is the periodogram at Fourier frequency $\omega_j = 2\pi j / n$.

**GPH log-periodogram regression** (Geweke and Porter-Hudak, 1983). Using the $m = \lfloor\sqrt{n}\rfloor$ lowest Fourier frequencies:

$$
\ln I(\omega_j) = c - d\,\ln\!\left[2(1 - \cos \omega_j)\right] + u_j
$$

The OLS slope estimate of the regressor yields $\hat{d}$.

**Rescaled range (R/S) analysis.** The Hurst exponent $H$ is estimated via log-linear regression of the R/S statistic against subsample size, and then $\hat{d} = \hat{H} - 0.5$.

#### ARMA(p, q) Fitting

After fractional differencing, the series $\{W_t\}$ is modeled as ARMA(p, q).

For pure AR models ($q = 0$), the Yule-Walker equations are solved. Let $\gamma_k = \text{Cov}(W_t, W_{t-k})$ denote the autocovariance at lag $k$. The AR coefficients satisfy:

$$
\begin{pmatrix} \gamma_0 & \gamma_1 & \cdots & \gamma_{p-1} \\ \gamma_1 & \gamma_0 & \cdots & \gamma_{p-2} \\ \vdots & & \ddots & \vdots \\ \gamma_{p-1} & \gamma_{p-2} & \cdots & \gamma_0 \end{pmatrix} \begin{pmatrix} \phi_1 \\ \phi_2 \\ \vdots \\ \phi_p \end{pmatrix} = \begin{pmatrix} \gamma_1 \\ \gamma_2 \\ \vdots \\ \gamma_p \end{pmatrix}
$$

For mixed ARMA ($q > 0$), the conditional sum of squares (CSS) method is used. The one-step-ahead prediction errors are:

$$
e_t = W_t - \sum_{k=1}^{p} \phi_k W_{t-k} - \sum_{j=1}^{q} \theta_j e_{t-j}
$$

and the parameters are estimated by minimizing $\sum_t e_t^2$ via L-BFGS-B with coefficient bounds $(-0.99, 0.99)$ to enforce stationarity and invertibility.

When automatic order selection is enabled, a grid search over $p \in \{0, 1, 2\}$ and $q \in \{0, 1, 2\}$ is performed, selecting the combination that minimizes the Bayesian Information Criterion:

$$
\text{BIC} = n_{\text{eff}} \ln(\hat{\sigma}_\varepsilon^2) + (p + q) \ln(n_{\text{eff}})
$$

### Synthesis Procedure

1. Estimate $\hat{d}$ and compute the fractional differencing coefficients $\{\pi_k\}$.
2. Apply fractional differencing to obtain $\{W_t\}$ and fit the ARMA(p, q) model.
3. Generate synthetic ARMA innovations and the differenced series:

$$
\hat{W}_t = \sum_{k=1}^{p} \hat{\phi}_k \hat{W}_{t-k} + \sum_{j=1}^{q} \hat{\theta}_j \varepsilon_{t-j} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \hat{\sigma}_\varepsilon^2)
$$

4. Invert the fractional differencing via a finite impulse response (FIR) convolution:

$$
\hat{X}_t = \sum_{k=0}^{K} \psi_k\,\hat{W}_{t-k}
$$

where the inverse coefficients are:

$$
\psi_0 = 1, \qquad \psi_k = \psi_{k-1} \cdot \frac{k - 1 + d}{k}, \quad k \geq 1
$$

5. Re-seasonalize: $\hat{Q}_t = \hat{X}_t \cdot \sigma_{m(t)} + \mu_{m(t)}$.
6. Enforce non-negativity.

## Statistical Properties

The ARFIMA model directly parameterizes long-range dependence through $d$, reproducing the hyperbolic decay of the autocorrelation function $\rho(k) \sim C k^{2d-1}$ as $k \to \infty$ and the spectral divergence $f(\omega) \sim C' \omega^{-2d}$ near the origin. The short-memory ARMA component captures structure at lags $1$ through $p$ and the moving-average smoothing at lags $1$ through $q$.

Monthly means and standard deviations are preserved through deseasonalization and re-seasonalization. However, the model assumes Gaussian innovations, which may underrepresent extreme events or heavy-tailed behavior. Spatial dependence is not modeled. Truncation of the infinite coefficient series at lag $K$ introduces approximation error that grows as $d$ approaches $0.5$.

## Limitations

- Univariate only; must be combined with spatial methods for multisite applications.
- Reliable estimation of $d$ requires long records (50+ years recommended).
- Gaussian innovation assumption may underrepresent tail behavior.
- Truncation of the fractional differencing series at finite $K$ is an approximation.
- CSS estimation for the MA component exhibits known small-sample bias.

## References

**Primary:**
Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using fractional differencing. *Water Resources Research*, 20(12), 1898-1908. https://doi.org/10.1029/WR020i012p01898

**See also:**
- Granger, C.W.J., and Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15-29. https://doi.org/10.1111/j.1467-9892.1980.tb00297.x
- Geweke, J., and Porter-Hudak, S. (1983). The estimation and application of long memory time series models. *Journal of Time Series Analysis*, 4(4), 221-238. https://doi.org/10.1111/j.1467-9892.1983.tb00371.x
- Fox, R., and Taqqu, M.S. (1986). Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series. *The Annals of Statistics*, 14(2), 517-532.
- Montanari, A., Rosso, R., and Taqqu, M.S. (1997). Fractionally differenced ARIMA models applied to hydrologic time series. *Water Resources Research*, 33(5), 1035-1044. https://doi.org/10.1029/97WR00043
- Koutsoyiannis, D. (2002). The Hurst phenomenon and fractional Gaussian noise made easy. *Hydrological Sciences Journal*, 47(4), 573-595.

---

**Implementation:** `src/synhydro/methods/generation/parametric/arfima.py`
