# Phase Randomization (Brunner et al. 2019)

**Classification:** Nonparametric
**Temporal Resolution:** Daily
**Site Compatibility:** Univariate

---

## Technical Specifications

| Property | Specification |
|----------|---------------|
| Input data | Daily streamflow, minimum 730 days (2 years), must be complete (no missing days) |
| Output frequency | Daily (D) |
| Distributional assumption | Four-parameter kappa distribution (or empirical for no extrapolation) |
| Correlation structure | Full power spectrum preserved via Fourier transform |
| Temporal dependence | Both short-range (daily autocorrelation) and long-range (Hurst phenomenon) |

---

## Algorithm Description

Phase randomization (PR) is a nonparametric method that generates synthetic streamflow time series by randomizing the Fourier phase spectrum while preserving the amplitude (power) spectrum. This approach maintains both short- and long-range temporal dependence structures present in observed data.

### Preprocessing

1. **Leap day removal**
   - February 29 observations are removed to ensure consistent 365-day years
   - Facilitates day-of-year indexing and distribution fitting

2. **Day-of-year indexing**
   - Create index mapping each observation to day-of-year (1-365)
   - Accounts for leap year adjustments in original calendar

3. **Data validation**
   - Minimum 730 days (2 full years) required
   - Total length must be multiple of 365 after removing leap days
   - No missing observations allowed

### Fitting/Calibration

1. **Marginal distribution fitting (if `marginal='kappa'`)**
   - For each day d ∈ {1, ..., 365}:
     - Define moving window: days within ±`win_h_length` around day d (circular)
     - Extract all observations for window days across all years
     - Compute L-moments from window data
     - Fit four-parameter kappa distribution using L-moment matching
   - Parameters: ξ (location), α (scale), k (shape 1), h (shape 2)

2. **Normal score transformation**
   - For each day d ∈ {1, ..., 365}:
     - Rank all observations for day d across years
     - Generate standard normal sample of same size
     - Map ranked observations to ranked normal values
   - Creates normalized series with standard normal marginals per day

3. **Fourier transform**
   - Compute FFT of normalized series: `FT = FFT(norm)`
   - Extract modulus (amplitude spectrum): `M = |FT|`
   - Extract phases (argument): `φ = arg(FT)`
   - Store first half indices (positive frequencies) and mirror indices

### Generation

1. **Phase randomization**
   - Keep DC component (index 0, mean): `FT_new[0] = FT[0]`
   - For first half (positive frequencies):
     - Generate random phases from Uniform(-π, π)
     - Construct: `FT_new[k] = M[k] * exp(i * φ_random[k])`
   - For second half (negative frequencies):
     - Apply conjugate symmetry: `FT_new[-k] = conj(FT_new[k])`
   - For Nyquist frequency (if n even): keep real-valued

2. **Inverse Fourier transform**
   - Apply inverse FFT: `norm_new = real(IFFT(FT_new))`
   - Result: phase-randomized series in normalized domain

3. **Back-transformation to original distribution**
   - For each day d ∈ {1, ..., 365}:
     - **If kappa marginal:**
       - Generate kappa sample using fitted parameters for day d
       - Rank both `norm_new` values and kappa sample for day d
       - Map normalized ranks to kappa-distributed values via rank matching
     - **If empirical marginal:**
       - Rank `norm_new` values for day d
       - Map ranks directly to observed data for day d (no extrapolation)

4. **Negative value handling**
   - If any values < 0:
     - For day d with negative values:
       - Find minimum observed value for day d: `min_obs`
       - Replace negatives with `Uniform(0, min_obs)` samples
   - Ensures physical validity (non-negative flows)

---

## Key Parameters

- **`marginal`**: Marginal distribution type for back-transformation
  - Type: str
  - Options: `'kappa'` (default), `'empirical'`
  - Notes: Kappa allows extrapolation beyond observed range; empirical does not

- **`win_h_length`**: Half-window length for daily distribution fitting
  - Type: int
  - Default: 15
  - Total window: `2 * win_h_length + 1` days (e.g., 31 days for default)
  - Notes: Larger windows smooth seasonal transitions; smaller windows capture sharper variability

---

## Algorithmic Details

### L-Moments Estimation

L-moments are linear combinations of order statistics used for robust distribution fitting:

```
Given sorted sample x[1] ≤ x[2] ≤ ... ≤ x[n]:

Probability weighted moments:
  b₀ = mean(x)
  b₁ = mean(p₁ * x)  where p₁ = i/(n-1)
  b₂ = mean(p₂ * x)  where p₂ = p₁ * (i-1)/(n-1)
  b₃ = mean(p₃ * x)  where p₃ = p₂ * (i-2)/(n-1)

L-moments:
  λ₁ = b₀                           (L-mean, equals sample mean)
  λ₂ = 2b₁ - b₀                     (L-scale)
  τ₃ = 2(3b₂ - b₀)/(2b₁ - b₀) - 3  (L-skewness)
  τ₄ = 5(2(2b₃ - 3b₂) + b₀)/(2b₁ - b₀) + 6  (L-kurtosis)
```

### Kappa Distribution

Four-parameter kappa distribution CDF:
```
For h ≠ 0:
  F(x) = {1 - h[1 - k(x-ξ)/α]^(1/k)}^(1/h)

For h = 0 (GEV case):
  F(x) = exp{-[1 - k(x-ξ)/α]^(1/k)}
```

Inverse CDF (quantile function):
```
For h ≠ 0:
  x(F) = ξ + (α/k)[1 - ((1-F^h)/h)^k]

For h = 0 (GEV case):
  x(F) = ξ + (α/k)[1 - (-log F)^k]
```

Parameters fitted by minimizing:
```
minimize (τ₃_observed - τ₃_theoretical(k,h))² + (τ₄_observed - τ₄_theoretical(k,h))²
```

### Fourier Phase Spectrum

The phase spectrum φ(f) determines the temporal alignment of frequency components. Randomizing phases while preserving amplitudes maintains:

- **Power spectrum**: `S(f) = |FT(x)|²` (unchanged)
- **Autocorrelation**: R(τ) = IFFT(S(f)) (preserved in expectation)
- **Long-range dependence**: Power-law decay in S(f) maintained

Conjugate symmetry ensures real-valued output:
```
FT[-k] = conj(FT[k]) for k = 1, ..., floor(n/2)
```

---

## Algorithm Variations

- **Wavelet-based PR**: Replace Fourier with wavelet transform (better for non-stationary signals)
- **Amplitude-Adjusted PR**: Also perturb amplitudes slightly for increased ensemble spread
- **Multivariate PR**: Extend to preserve cross-site correlations

---

## Implementation Notes

### Computational complexity

- **Time**: O(n log n) dominated by FFT operations
- **Space**: O(365 × 4) for kappa parameters + O(n) for time series storage
- Efficient for typical hydrologic records (10-100 years of daily data)

### Limitations

- **Univariate only**: Cannot preserve spatial correlations between sites
- **Stationarity assumption**: Assumes temporal structure is time-invariant
- **Length constraint**: Generated series has same length as observed (no extrapolation in time)
- **Leap day handling**: Output excludes Feb 29 dates
- **Data requirements**: Minimum 2 years; works best with 10+ years
- **Seasonality**: Daily fitting can be noisy with limited data; window smooths but reduces flexibility

### Special handling

- **Negative values**: Replaced via uniform sampling between 0 and daily minimum
- **Kappa fitting failures**: If optimization fails for day d, use parameters from day d-1 (or d+1 if d=1)
- **Nyquist frequency**: Forced to be real-valued for even-length series
- **Circular window**: Days wrap around year boundaries (e.g., day 1 window includes days 351-365 from previous year)

### Hydrologic properties preserved

**Explicitly preserved:**
- Marginal distributions (via kappa or empirical)
- Power spectrum (all temporal autocorrelations in expectation)
- Long-range dependence (Hurst coefficient)
- Seasonal patterns (day-of-year distributions)

**Implicitly preserved (approximately):**
- Mean and standard deviation
- Skewness and kurtosis (via kappa parameters)
- Drought duration distributions (via autocorrelation structure)

**Not preserved:**
- Exact observed autocorrelations (only in expectation)
- Phase coherence (by design - randomized)
- Specific drought event timing

---

## References

**Primary:**
Brunner, M.I., Bárdossy, A., and Furrer, R. (2019). Technical note: Stochastic simulation of streamflow time series using phase randomization. *Hydrology and Earth System Sciences*, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019

**Methodological foundations:**
- **Phase randomization**: Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., and Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77-94.
- **L-moments**: Hosking, J. R. M. (1990). L-moments: Analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society Series B*, 52, 105-124.
- **Kappa distribution**: Hosking, J. R. M. (1994). The four-parameter kappa distribution. *IBM Journal of Research and Development*, 38, 251-258.

**Original R implementation:**
Brunner, M.I. (2017). PRSim: Stochastic Simulation of Streamflow Time Series using Phase Randomization. R package. https://cran.r-project.org/package=PRSim

---

**SGLib Implementation:** [`src/sglib/methods/generation/nonparametric/phase_randomization.py`](https://github.com/Pywr-DRB/SGLib/blob/main/src/sglib/methods/generation/nonparametric/phase_randomization.py)

**Tests:** [`tests/test_phase_randomization_generator.py`](https://github.com/Pywr-DRB/SGLib/blob/main/tests/test_phase_randomization_generator.py)

---

## Usage Example

```python
import pandas as pd
from sglib.methods.generation.nonparametric import PhaseRandomizationGenerator

# Load daily streamflow data
Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)

# Initialize generator with kappa marginal
gen = PhaseRandomizationGenerator(
    Q_daily,
    marginal='kappa',      # Use kappa distribution for extrapolation
    win_h_length=15        # 31-day window for daily fitting
)

# Preprocessing: remove leap days, create day index
gen.preprocessing()

# Fit: estimate kappa parameters, compute FFT
gen.fit()

# Generate 100 realizations
ensemble = gen.generate(n_realizations=100, seed=42)

# Access results
first_realization = ensemble.get_realization(0)
print(f"Generated {ensemble.n_realizations} realizations")
print(f"Each has {len(first_realization)} days")
```

---

## Verification Checklist

When implementing or validating phase randomization:

- [ ] Marginal distributions match kappa fit (or empirical) per day
- [ ] Power spectrum of synthetic matches observed
- [ ] Autocorrelation function preserved (ensemble mean)
- [ ] Long-range dependence (Hurst coefficient) maintained
- [ ] No negative values in output
- [ ] Seasonal patterns realistic
- [ ] Drought duration distributions similar to observed
- [ ] Ensemble spread is reasonable (not too narrow/wide)
