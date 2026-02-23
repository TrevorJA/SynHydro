# Wavelet Auto-Regressive Method (WARM)

## Technical Specifications

| Property | Value |
|----------|-------|
| **Method Name** | WARM (Wavelet Auto-Regressive Method) |
| **Implementation** | `WARMGenerator` |
| **Type** | Parametric, Stochastic |
| **Frequency** | Annual |
| **Sites** | Univariate (single site) |
| **Reference** | Nowak et al. (2011) |
| **Full Citation** | Nowak, K., Rajagopalan, B., & Zagona, E. (2011). A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary trends. *Journal of Hydrology*, 410(1-2), 1-12. |

## Overview

The Wavelet Auto-Regressive Method (WARM) is an advanced stochastic generator designed for non-stationary streamflow simulation. WARM combines continuous wavelet transforms with autoregressive modeling to preserve both time-varying spectral characteristics and temporal persistence in synthetic streamflow sequences.

**Key Innovation**: The Scale Averaged Wavelet Power (SAWP) enables WARM to capture and reproduce time-varying power spectral characteristics, making it particularly suitable for streamflows with non-stationary trends, regime shifts, or low-frequency variability (e.g., climate oscillations, long-term droughts).

## Algorithm Description

WARM operates through a 4-step process that decomposes, models, and reconstructs streamflow signals using wavelet analysis and autoregressive modeling.

### 1. Preprocessing

**Objective**: Prepare annual streamflow data for wavelet analysis.

**Steps**:
1. Validate input data (univariate time series)
2. Resample to annual frequency if needed (monthly → annual via sum, daily → annual via sum)
3. Store observed annual flows: $Q_{\text{obs}}(t)$, where $t = 1, 2, \ldots, T$ years

**Validation**:
- Minimum recommended: 20 years of data
- WARM designed specifically for annual streamflow
- Single-site only (univariate)

### 2. Fitting (Training)

WARM fitting consists of 4 sub-steps:

#### Step 1: Continuous Wavelet Transform (CWT)

**Objective**: Decompose time series into time-frequency components.

Apply continuous wavelet transform using mother wavelet $\psi$:

$$
W(s, t) = \int_{-\infty}^{\infty} Q(\tau) \cdot \frac{1}{\sqrt{s}} \psi^*\left(\frac{\tau - t}{s}\right) d\tau
$$

where:
- $W(s, t)$ = wavelet coefficients at scale $s$ and time $t$
- $\psi$ = mother wavelet function (e.g., Morlet, Mexican Hat)
- $s$ = scale parameter (related to frequency: larger scales = lower frequencies)
- $*$ denotes complex conjugate

**Implementation**:
```python
scales = np.arange(1, n_scales + 1)
coefficients, frequencies = pywt.cwt(Q_obs, scales, wavelet='morl')
# Result: coefficients with shape (n_scales, n_years)
```

**Recommended Wavelets**:
- **Morlet** (`'morl'`): Best for hydrologic applications, good time-frequency localization
- **Mexican Hat** (`'mexh'`): Symmetric, good for detecting peaks
- **Gaussian** (`'gaus1'`-`'gaus8'`): Smooth, varying derivatives

#### Step 2: Scale Averaged Wavelet Power (SAWP)

**Objective**: Compute time-varying power across all frequency scales.

SAWP is the **key innovation** of WARM (Nowak et al. 2011) compared to previous methods (Kwon et al. 2007). It captures temporal variations in spectral power.

Calculate wavelet power at each scale and time:

$$
P(s, t) = |W(s, t)|^2
$$

Average across all scales for each time point:

$$
\text{SAWP}(t) = \frac{1}{S} \sum_{s=1}^{S} P(s, t) = \frac{1}{S} \sum_{s=1}^{S} |W(s, t)|^2
$$

where $S$ = total number of scales.

**Physical Meaning**:
- High SAWP(t) → High energy/variability at time $t$
- Low SAWP(t) → Low energy/variability at time $t$
- Captures non-stationary variance structure

**Implementation**:
```python
power = np.abs(coefficients) ** 2
sawp = np.mean(power, axis=0)  # Average across scales
```

#### Step 3: Normalize by SAWP

**Objective**: Remove time-varying power to create stationary components.

For each scale $s$ and time $t$:

$$
W_{\text{norm}}(s, t) = \frac{W(s, t)}{\sqrt{\text{SAWP}(t) + \epsilon}}
$$

where $\epsilon$ is a small constant ($10^{-10}$) to prevent division by zero.

**Result**: Normalized coefficients $W_{\text{norm}}(s, t)$ are approximately stationary and suitable for AR modeling.

#### Step 4: Fit AR Models to Each Scale

**Objective**: Model temporal persistence at each frequency scale.

For each scale $s$, fit an AR(p) model to the normalized coefficients $W_{\text{norm}}(s, \cdot)$:

$$
W_{\text{norm}}(s, t) = \mu_s + \sum_{i=1}^{p} \phi_{s,i} \left[W_{\text{norm}}(s, t-i) - \mu_s\right] + \epsilon_s(t)
$$

where:
- $\mu_s$ = mean of normalized coefficients at scale $s$
- $\phi_{s,i}$ = AR coefficient for lag $i$ at scale $s$
- $\epsilon_s(t) \sim N(0, \sigma_s^2)$ = white noise innovation
- $p$ = AR order (typically 1 or 2)

**Parameter Estimation**: Yule-Walker equations

Given autocorrelation function $\rho(k)$, solve:

$$
\begin{bmatrix}
1 & \rho(1) & \cdots & \rho(p-1) \\
\rho(1) & 1 & \cdots & \rho(p-2) \\
\vdots & \vdots & \ddots & \vdots \\
\rho(p-1) & \rho(p-2) & \cdots & 1
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\
\phi_2 \\
\vdots \\
\phi_p
\end{bmatrix}
=
\begin{bmatrix}
\rho(1) \\
\rho(2) \\
\vdots \\
\rho(p)
\end{bmatrix}
$$

Innovation variance:

$$
\sigma^2 = \gamma(0) \left(1 - \sum_{i=1}^{p} \phi_i \rho(i)\right)
$$

where $\gamma(0)$ is the variance of the normalized coefficients.

**Stored Parameters**:
- For each scale $s$: $\{\mu_s, \phi_{s,1}, \ldots, \phi_{s,p}, \sigma_s\}$
- SAWP time series: $\text{SAWP}(1), \ldots, \text{SAWP}(T)$
- Wavelet parameters: wavelet type, scales array

### 3. Generation (Synthesis)

**Objective**: Generate synthetic annual streamflow sequences.

#### Step 1: Generate Normalized Coefficients

For each scale $s$ and each synthetic year $t$:

1. Initialize: $W_{\text{syn,norm}}(s, 0) = \mu_s$ (or use historical values)

2. Generate AR process:
   $$
   W_{\text{syn,norm}}(s, t) = \mu_s + \sum_{i=1}^{p} \phi_{s,i} \left[W_{\text{syn,norm}}(s, t-i) - \mu_s\right] + \epsilon_s(t)
   $$

   where $\epsilon_s(t) \sim N(0, \sigma_s^2)$ are independent random innovations.

**Result**: Normalized synthetic coefficients for all scales and times.

#### Step 2: Resample SAWP

**Objective**: Create time-varying power for synthetic sequence.

Resample SAWP from historical record with replacement:

$$
\text{SAWP}_{\text{syn}}(t) = \text{SAWP}_{\text{obs}}(k_t)
$$

where $k_t$ is a randomly selected time index from $\{1, 2, \ldots, T\}$.

**Purpose**: Preserve the distribution and range of power variations while allowing different temporal patterns.

#### Step 3: Rescale Coefficients

**Objective**: Restore time-varying power to synthetic coefficients.

For each scale $s$ and time $t$:

$$
W_{\text{syn}}(s, t) = W_{\text{syn,norm}}(s, t) \cdot \sqrt{\text{SAWP}_{\text{syn}}(t) + \epsilon}
$$

**Result**: Synthetic wavelet coefficients with time-varying power.

#### Step 4: Inverse Wavelet Transform

**Objective**: Reconstruct synthetic streamflow time series.

The inverse CWT is approximated by weighted summation across scales:

$$
Q_{\text{syn}}(t) = C \sum_{s=1}^{S} \frac{\Re[W_{\text{syn}}(s, t)]}{\sqrt{s}} \cdot w_s
$$

where:
- $\Re[\cdot]$ = real part
- $w_s$ = scale-dependent weights (typically $w_s = 1/\sqrt{s}$)
- $C$ = normalization constant

**Adjustment**:
After reconstruction, adjust mean and variance to match observed:

$$
Q_{\text{syn,adjusted}}(t) = \sigma_{\text{obs}} \cdot \frac{Q_{\text{syn}}(t) - \bar{Q}_{\text{syn}}}{\sigma_{\text{syn}}} + \bar{Q}_{\text{obs}}
$$

**Non-negativity**: Ensure $Q_{\text{syn}}(t) \geq 0$ (set negative values to 0).

## Mathematical Formulation Summary

### Input
- Historical annual flows: $Q_{\text{obs}}(t)$, $t = 1, \ldots, T$

### Parameters
- Wavelet type: $\psi$ (e.g., Morlet)
- Number of scales: $S$ (e.g., 64)
- AR order: $p$ (e.g., 1)

### Fitted Quantities
- Wavelet coefficients: $W(s, t)$ for $s = 1, \ldots, S$ and $t = 1, \ldots, T$
- SAWP: $\text{SAWP}(t)$ for $t = 1, \ldots, T$
- AR parameters for each scale: $\{\mu_s, \phi_{s,1}, \ldots, \phi_{s,p}, \sigma_s\}$ for $s = 1, \ldots, S$

### Output
- Synthetic annual flows: $Q_{\text{syn}}(t)$, $t = 1, \ldots, T_{\text{syn}}$

## Key Parameters and Tuning

### Wavelet Selection

| Wavelet | Code | Characteristics | Use Case |
|---------|------|-----------------|----------|
| **Morlet** | `'morl'` | Complex, good time-frequency localization | **Recommended** for streamflow (default) |
| **Mexican Hat** | `'mexh'` | Symmetric, second derivative of Gaussian | Detecting peaks/oscillations |
| **Gaussian (1st-8th)** | `'gaus1'`-`'gaus8'` | Smooth, varying smoothness | Experimental/smoothing |

**Recommendation**: Use **Morlet** wavelet (`'morl'`) as it provides the best balance between time and frequency localization for hydrologic signals.

### Number of Scales

**Parameter**: `scales` (default: 64)

**Guidelines**:
- **Low (8-16)**: Faster computation, captures only major frequencies
- **Medium (32-64)**: **Recommended**, good frequency resolution
- **High (128+)**: Very detailed, computationally expensive, may overfit

**Rule of thumb**: Use $S \approx T/2$ to $T$, where $T$ = number of years.

**Example**:
- 50 years of data → 32-64 scales
- 100 years of data → 64-128 scales

### AR Order

**Parameter**: `ar_order` (default: 1)

**Guidelines**:
- **AR(1)**: Captures first-order temporal persistence, **recommended for most cases**
- **AR(2)**: Captures additional lag structure, use if data shows strong 2-year memory
- **AR(3+)**: Rarely needed, risk of overfitting

**Selection**: Use information criteria (AIC/BIC) or cross-validation if uncertain. Default AR(1) works well for annual streamflow.

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| **CWT** | $O(S \cdot T)$ | $S$ = scales, $T$ = years |
| **SAWP Calculation** | $O(S \cdot T)$ | Simple averaging |
| **AR Fitting** | $O(S \cdot T \cdot p^2)$ | $p$ = AR order, Yule-Walker solution |
| **Generation** | $O(S \cdot T_{\text{syn}})$ | Per realization |
| **Inverse CWT** | $O(S \cdot T_{\text{syn}})$ | Weighted summation |

**Total Fitting**: $O(S \cdot T \cdot p^2)$

**Total Generation** (per realization): $O(S \cdot T_{\text{syn}})$

**Memory**: $O(S \cdot T)$ for storing wavelet coefficients.

**Typical Performance** (64 scales, 50 years, AR(1)):
- Fitting: < 1 second
- Generation (100 realizations × 100 years): ~ 2-5 seconds

## Limitations and Special Handling

### 1. Annual Frequency Only

**Limitation**: WARM is designed for annual streamflow.

**Reason**: Wavelet decomposition for capturing multi-year oscillations (e.g., ENSO, PDO) works best at annual scale.

**Workaround**: For monthly/daily generation, consider:
- Use WARM for annual totals
- Disaggregate to finer resolution using separate method (e.g., K-NN, HMM)

### 2. Univariate (Single Site)

**Current Implementation**: Single site only.

**Extension**: Nowak et al. (2011) describes multi-site extension:
- Generate multiple sites using WARM independently
- Apply spatial disaggregation method to impose spatial correlation
- Preserve historical spatial correlation structure

**Future Work**: Multi-site version could use:
- Coupled AR models at each scale (VAR instead of AR)
- Joint wavelet analysis with cross-wavelet transforms

### 3. Edge Effects

**Issue**: CWT has edge effects (cone of influence) at start/end of time series.

**Handling**:
- PyWavelets pads signal automatically
- WARM resamples SAWP, which mitigates edge effect impact
- For very short series (< 20 years), edge effects may be noticeable

**Recommendation**: Use at least 30 years of data for robust results.

### 4. Non-stationary Mean Trends

**Handling**: WARM preserves non-stationary spectral characteristics (via SAWP).

**Strong Linear Trends**: If data has strong deterministic trend:
- Option 1: Detrend before WARM, re-trend after generation
- Option 2: Use WARM as-is if trend is part of low-frequency variability

**Recommendation**: Analyze whether trend is deterministic or stochastic before deciding.

### 5. Zero/Near-Zero Flows

**Handling**: WARM can generate negative values (since AR model has Gaussian innovations).

**Correction**: Post-processing in `_generate()`:
```python
Q_syn = np.maximum(Q_syn, 0)  # Truncate to non-negative
```

**Limitation**: For intermittent streams with frequent zeros, consider:
- Two-stage model (occurrence + magnitude)
- Non-parametric methods (K-NN, phase randomization)

### 6. Reconstruction Accuracy

**Approximation**: Inverse CWT is approximate (PyWavelets doesn't provide exact inverse for CWT).

**Adjustment**: WARM applies mean/variance adjustment after reconstruction:
- Preserves observed mean and standard deviation
- Small reconstruction error is acceptable for stochastic generation

## Hydrologic Properties Preserved

### Temporal Properties

| Property | Preserved? | Method |
|----------|------------|--------|
| **Mean** | ✅ Exactly | Post-reconstruction adjustment |
| **Variance** | ✅ Exactly | Post-reconstruction adjustment |
| **Lag-1 autocorrelation** | ✅ Approximately | AR models at each scale |
| **Multi-year persistence** | ✅ Strong | Low-frequency wavelet components + AR |
| **Non-stationary variance** | ✅ Strong | SAWP resampling |

### Frequency-Domain Properties

| Property | Preserved? | Method |
|----------|------------|--------|
| **Power spectrum** | ✅ Strong | Wavelet decomposition |
| **Time-varying spectrum** | ✅ Strong | **SAWP innovation** |
| **Low-frequency oscillations** | ✅ Strong | Large-scale wavelet components |
| **High-frequency variability** | ✅ Moderate | Small-scale wavelet components + AR noise |

### Distributional Properties

| Property | Preserved? | Notes |
|----------|------------|-------|
| **Marginal distribution** | ⚠️ Approximate | Mean/variance preserved, but shape may differ |
| **Skewness** | ⚠️ Moderate | Not explicitly preserved |
| **Extremes** | ⚠️ Moderate | AR innovations are Gaussian |

**Note**: For better preservation of marginal distribution, consider:
- Log-transformation before WARM
- Post-processing to match observed distribution (e.g., quantile mapping)

## Comparison with Other Methods

### WARM vs. Traditional AR Models

| Aspect | WARM | AR(1) / ARMA |
|--------|------|--------------|
| **Non-stationarity** | ✅ Handles via SAWP | ❌ Assumes stationarity |
| **Low-frequency persistence** | ✅ Strong (wavelet scales) | ⚠️ Weak (limited memory) |
| **Computation** | Moderate (CWT) | Fast |
| **Interpretability** | Moderate | High |

### WARM vs. Kwon et al. (2007) Wavelet-AR

**Key Difference**: SAWP normalization

| Feature | WARM (Nowak 2011) | Kwon et al. (2007) |
|---------|-------------------|---------------------|
| **Time-varying power** | ✅ SAWP captures | ❌ Not captured |
| **Non-stationary variance** | ✅ Strong | ⚠️ Moderate |
| **Complexity** | Moderate | Moderate |

**Advantage**: WARM better preserves non-stationary spectral characteristics.

### WARM vs. HMM

| Aspect | WARM | HMM |
|--------|------|-----|
| **Regime changes** | ⚠️ Implicit (via SAWP) | ✅ Explicit (states) |
| **Multi-site** | ❌ Univariate (current) | ✅ Native multi-site |
| **Frequency** | Annual | Annual or Monthly |
| **Low-frequency persistence** | ✅ Strong | ⚠️ Depends on state design |

## Usage Examples

### Basic Usage

```python
import pandas as pd
from sglib.methods.generation.parametric.warm import WARMGenerator

# Load annual streamflow data
Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)

# Initialize WARM generator with default parameters
warm = WARMGenerator(Q_annual.iloc[:, 0])

# Preprocess
warm.preprocessing()

# Fit model to historical data
warm.fit()

# Generate 100 realizations of 100 years each
ensemble = warm.generate(n_years=100, n_realizations=100, seed=42)

# Access realizations
Q_syn_r0 = ensemble.data_by_realization[0]  # First realization
```

### Custom Parameters

```python
# Use Mexican Hat wavelet, 32 scales, AR(2) model
warm = WARMGenerator(
    Q_annual,
    wavelet='mexh',    # Mexican Hat wavelet
    scales=32,          # 32 frequency scales
    ar_order=2,         # AR(2) model for temporal persistence
    debug=True          # Enable debug logging
)

warm.preprocessing()
warm.fit()
ensemble = warm.generate(n_years=50, n_realizations=50, seed=123)
```

### Analyzing SAWP

```python
import matplotlib.pyplot as plt

# Fit model
warm = WARMGenerator(Q_annual, scales=64)
warm.preprocessing()
warm.fit()

# Plot SAWP time series
plt.figure(figsize=(12, 4))
plt.plot(warm.Q_obs_annual.index, warm.sawp_)
plt.xlabel('Year')
plt.ylabel('SAWP')
plt.title('Scale Averaged Wavelet Power - Captures Time-Varying Energy')
plt.grid(True)
plt.show()

# SAWP reveals periods of high/low variability
```

### Comparing Wavelets

```python
wavelets = ['morl', 'mexh', 'gaus4']

for wavelet in wavelets:
    warm = WARMGenerator(Q_annual, wavelet=wavelet, scales=32)
    warm.preprocessing()
    warm.fit()
    ensemble = warm.generate(n_years=100, n_realizations=10, seed=42)

    print(f"\n{wavelet} wavelet:")
    print(f"  Generated mean: {ensemble.data_by_realization[0].mean().values[0]:.2f}")
    print(f"  Generated std:  {ensemble.data_by_realization[0].std().values[0]:.2f}")
```

## References

### Primary Reference

**Nowak, K., Rajagopalan, B., & Zagona, E. (2011).** A Wavelet Auto-Regressive Method (WARM) for multi-site streamflow simulation of data with non-stationary trends. *Journal of Hydrology*, 410(1-2), 1-12.
- [https://doi.org/10.1016/j.jhydrol.2011.08.049](https://doi.org/10.1016/j.jhydrol.2011.08.049)

### Related Work

**Kwon, H.-H., Lall, U., & Khalil, A. F. (2007).** Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition: Applications to rainfall and temperature. *Water Resources Research*, 43(5).
- Precursor to WARM, introduced wavelet-AR methodology
- WARM extends this with SAWP for time-varying power

**Torrence, C., & Compo, G. P. (1998).** A Practical Guide to Wavelet Analysis. *Bulletin of the American Meteorological Society*, 79(1), 61-78.
- Comprehensive guide to wavelet analysis
- Foundational reference for continuous wavelet transform

**Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Yule-Walker equations for AR parameter estimation
- Theoretical foundation for autoregressive models

### Software

**PyWavelets**: Wavelet transforms in Python
- [https://pywavelets.readthedocs.io/](https://pywavelets.readthedocs.io/)
- Used for continuous wavelet transform (CWT)

## Version History

- **v0.0.1** (2024): Initial implementation
  - Univariate (single-site) WARM
  - Morlet, Mexican Hat, Gaussian wavelets supported
  - AR(1) through AR(p) models
  - SAWP-based time-varying power preservation
  - Full test coverage (39 tests)

## Future Enhancements

1. **Multi-Site Extension**
   - Implement VAR (Vector AR) models at each scale
   - Spatial correlation preservation via coupled wavelets

2. **Distribution Preservation**
   - Optional log-transformation for skewed flows
   - Quantile mapping post-processing

3. **Automatic Parameter Selection**
   - AIC/BIC for AR order selection
   - Cross-validation for scale selection

4. **Monthly/Daily Disaggregation**
   - Couple WARM (annual) with K-NN or HMM (monthly/daily)
   - Preserve both annual structure (WARM) and seasonal patterns

5. **Computational Optimization**
   - Parallel AR fitting across scales
   - GPU-accelerated wavelet transforms for large datasets
