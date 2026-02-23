# Thomas-Fiering (1962) with Stedinger-Taylor Normalization

**Classification:** Parametric
**Temporal Resolution:** Monthly
**Site Compatibility:** Univariate

---

## Technical Specifications

| Property | Specification |
|----------|---------------|
| Input data | Monthly streamflow, minimum 2 complete years (24 months) |
| Output frequency | MS (Month Start) |
| Distributional assumption | Normal after Stedinger-Taylor transformation |
| Correlation structure | Lag-1 serial correlation (AR(1) model) |
| Temporal dependence | Monthly autoregressive with seasonal parameters |

---

## Algorithm Description

The Thomas-Fiering method generates synthetic monthly streamflow using a first-order autoregressive (AR(1)) model with season-specific (monthly) parameters. The Stedinger-Taylor (1982) normalization improves the method by applying a lower-bound adjustment before log transformation, reducing skewness and improving normality.

This method preserves monthly means, standard deviations, and lag-1 serial correlations, making it suitable for operational hydrology, reservoir simulation, and water supply planning at monthly resolution.

### Preprocessing

1. **Frequency validation**
   - Validate input has DatetimeIndex
   - If not monthly frequency ('MS'), resample:
     - Daily → Monthly: sum flows within each month
     - Weekly → Monthly: sum flows within each month

2. **Univariate enforcement**
   - Thomas-Fiering is univariate (single site only)
   - If DataFrame with multiple columns, use first column only
   - Warn user about multi-site input

3. **Stedinger-Taylor normalization**
   - **Fit lower bound parameters (τ)**:
     - For each month m ∈ {1, ..., 12}:
       - Extract all observations for month m across years
       - Compute: `τₘ = (Qₘₐₓ·Qₘᵢₙ - Q²ₘₑ

dian) / (Qₘₐₓ + Qₘᵢₙ - 2·Qₘₑdian)`
       - Clip: `τₘ = max(τₘ, 0)` (ensure non-negative)
   - **Apply transformation**:
     - For each observation in month m:
       - Subtract lower bound and log-transform: `Xₘ = log(Qₘ - τₘ)`
       - This reduces skewness and improves normality

### Fitting/Calibration

1. **Compute monthly statistics**
   - For each month m ∈ {1, ..., 12}, compute from normalized flows X:
     - Mean: `μₘ = E[Xₘ]`
     - Standard deviation: `σₘ = std(Xₘ)`

2. **Compute lag-1 serial correlations**
   - For each month transition m → m+1 (with Dec → Jan wraparound):
     - Extract all pairs of consecutive months across years:
       - `(X₁ₘ, X₁,ₘ₊₁), (X₂ₘ, X₂,ₘ₊₁), ..., (Xₙₘ, Xₙ,ₘ₊₁)`
     - Filter out NaN and Inf values from transformation
     - Compute Pearson correlation: `ρₘ = corr(Xₘ, Xₘ₊₁)`
     - If insufficient data or zero variance, default to `ρₘ = 0`

3. **Store fitted parameters**
   - Monthly means: `μ = {μ₁, μ₂, ..., μ₁₂}`
   - Monthly standard deviations: `σ = {σ₁, σ₂, ..., σ₁₂}`
   - Monthly lag-1 correlations: `ρ = {ρ₁, ρ₂, ..., ρ₁₂}`
   - Stedinger transformation parameters: `τ = {τ₁, τ₂, ..., τ₁₂}`
   - Total: 48 parameters (12 months × 4 parameters)

### Generation

1. **Initialize first month**
   - For first month (January) of first year:
     ```
     X₁,₁ = μ₁ + ε₁,₁ · σ₁
     ```
     where `ε₁,₁ ~ N(0, 1)` is standard normal random variate

2. **Generate subsequent months (AR(1) recursion)**
   - For each subsequent month i = 2, 3, ..., n_years × 12:
     - Determine current month m (1-12) and previous month m_prev
     - Generate standard normal random variate: `ε ~ N(0, 1)`
     - Apply AR(1) formula:
       ```
       Xᵢ,ₘ = μₘ + ρₘ · (σₘ/σₘ_prev) · (Xᵢ₋₁,ₘ_prev - μₘ_prev) + √(1 - ρₘ²) · σₘ · ε
       ```
     - This preserves:
       - Mean: `E[Xᵢ,ₘ] = μₘ`
       - Variance: `Var[Xᵢ,ₘ] = σₘ²`
       - Lag-1 correlation: `corr(Xᵢ₋₁, Xᵢ) = ρₘ`

3. **Inverse Stedinger transformation**
   - For each generated normalized flow Xᵢ,ₘ:
     ```
     Qᵢ,ₘ = exp(Xᵢ,ₘ) + τₘ
     ```
   - Back-transforms from normal space to original flow space

4. **Non-negativity enforcement**
   - Replace negative values (rare, from transformation):
     ```
     Qᵢ,ₘ = max(Qᵢ,ₘ, Qₘᵢₙ,observed)
     ```
   - Fill NaN values with observed minimum for that month

5. **Time index creation**
   - Generate monthly DatetimeIndex starting from observed data start year
   - Frequency: 'MS' (month start)

---

## Key Parameters

- **`Q_obs`**: Observed streamflow data
  - Type: pd.Series or pd.DataFrame (first column used if DataFrame)
  - Notes: Must have DatetimeIndex. If not monthly, will be resampled during preprocessing.

- **`debug`**: Enable debug logging
  - Type: bool
  - Default: False

---

## Algorithmic Details

### AR(1) Model Formulation

**General AR(1) model**:
```
Xₜ = φ·Xₜ₋₁ + εₜ
```

**Thomas-Fiering seasonal AR(1)**:
```
Xᵢ,ₘ - μₘ = ρₘ · (σₘ/σₘ₋₁) · (Xᵢ₋₁,ₘ₋₁ - μₘ₋₁) + √(1 - ρₘ²) · σₘ · εᵢ,ₘ
```

where:
- `Xᵢ,ₘ`: Normalized flow for year i, month m
- `μₘ`: Mean for month m
- `σₘ`: Standard deviation for month m
- `ρₘ`: Lag-1 correlation from month m-1 to month m
- `εᵢ,ₘ ~ N(0, 1)`: Independent standard normal innovation

**Key properties**:
1. **Mean preservation**: `E[Xᵢ,ₘ] = μₘ`
2. **Variance preservation**: `Var[Xᵢ,ₘ] = σₘ²`
3. **Correlation preservation**: `Cov[Xᵢ₋₁,ₘ₋₁, Xᵢ,ₘ] = ρₘ · σₘ₋₁ · σₘ`

### Stedinger-Taylor Normalization

**Purpose**: Reduce skewness in streamflow distributions to improve normality assumption.

**Lower bound estimation**:
```
τₘ = (Qₘₐₓ · Qₘᵢₙ - Qₘₑdian²) / (Qₘₐₓ + Qₘᵢₙ - 2 · Qₘₑdian)
```

**Forward transformation**:
```
X = log(Q - τ)
```

**Inverse transformation**:
```
Q = exp(X) + τ
```

**Advantages over simple log**:
- Better handles skewness
- Reduces bias in mean and variance
- More robust to low flows

### Statistical Properties Preserved

**Explicitly preserved (in normalized space)**:
- Monthly means: `μₘ`
- Monthly standard deviations: `σₘ`
- Lag-1 serial correlations: `ρₘ`

**Approximately preserved (in original space)**:
- Monthly mean flows
- Monthly flow variability
- Sequential dependence structure

**Not preserved**:
- Higher-order (lag > 1) correlations (first-order model only)
- Spatial correlations (univariate method)
- Exact marginal distributions (transformed to normal)

---

## Algorithm Variations

- **Disaggregation**: Extend to daily timestep using fragmentat ion methods (e.g., Valencia-Schaake)
- **Multi-site extension**: Multivariate AR models (MVAR) preserve spatial correlations
- **Higher-order AR**: AR(p) models capture longer memory
- **Nonlinear transformations**: Box-Cox instead of log
- **Alternative normalization**: NPLN (Normal, Log-Normal) instead of Stedinger-Taylor

---

## Implementation Notes

### Computational complexity

- **Time**: O(n) where n = number of observations
  - Single pass to compute monthly statistics
  - Generation is O(n_years × 12) per realization
- **Space**: O(12) for monthly parameter storage
- Very efficient for typical hydrologic records (10-100 years of monthly data)

### Limitations

- **Univariate only**: Cannot model spatial correlations between sites
- **First-order only**: Only captures lag-1 dependence (month-to-month)
  - Misses multi-month persistence and seasonality in correlation structure
- **Normality assumption**: Assumes flows are normal after transformation
  - May not capture extreme events well (heavy tails)
- **Monthly timestep**: Not suitable for daily analysis
  - Use disaggregation methods if daily data needed
- **Stationarity**: Assumes parameters are time-invariant
  - Not suitable for trending or non-stationary data
- **Data requirements**: Needs at least 2 complete years (better with 10+)
  - More data improves parameter estimation reliability

### Special handling

- **NaN/Inf values**: From Stedinger transform when `Q - τ ≤ 0`
  - Filtered before correlation calculation
  - Replaced with minimum observed flow in generation
- **Negative generated flows**: Rare but possible from transformation
  - Replaced with observed minimum for that month
- **First month initialization**: Sampled from `N(μ₁, σ₁²)` without dependence
  - Allows generation to start from any month
- **December → January transition**: Handled via modular arithmetic
  - Month index wraps: `(m % 12) + 1`

### Hydrologic properties preserved

**Explicitly preserved (by design)**:
- Monthly mean flows
- Monthly standard deviations
- Lag-1 autocorrelation

**Implicitly preserved (approximately)**:
- Seasonal flow patterns
- Year-to-year variability
- Dry/wet spell persistence (via lag-1 correlation)

**Not preserved**:
- Multi-month droughts (>1 month memory)
- Spatial correlations (univariate)
- Exact flow distributions (normality imposed)
- Trends or non-stationarity

---

## References

**Primary:**
Thomas, H.A., and Fiering, M.B. (1962). Mathematical synthesis of streamflow sequences for the analysis of river basins by simulation. In *Design of Water Resource Systems* (eds. Maass et al.), pp. 459-493. Harvard University Press.

**Stedinger-Taylor Normalization:**
Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation: 1. Model verification and validation. *Water Resources Research*, 18(4), 909-918. https://doi.org/10.1029/WR018i004p00909

**Methodological foundations:**
- **AR(1) models**: Box, G.E.P., and Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- **Streamflow synthesis**: Fiering, M.B. (1967). *Streamflow Synthesis*. Harvard University Press.
- **Operational hydrology**: Loucks, D.P., and van Beek, E. (2017). *Water Resource Systems Planning and Management*. Springer.

**Extensions:**
- **Multivariate AR**: Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980). *Applied Modeling of Hydrologic Time Series*. Water Resources Publications.
- **Disaggregation**: Valencia, D., and Schaake, J.C. (1973). Disaggregation processes in stochastic hydrology. *Water Resources Research*, 9(3), 580-585.

---

**SGLib Implementation:** [`src/sglib/methods/generation/parametric/thomas_fiering.py`](https://github.com/Pywr-DRB/SGLib/blob/main/src/sglib/methods/generation/parametric/thomas_fiering.py)

**Tests:** [`tests/test_thomas_fiering_generator.py`](https://github.com/Pywr-DRB/SGLib/blob/main/tests/test_thomas_fiering_generator.py)

---

## Usage Example

```python
import pandas as pd
from sglib.methods.generation.parametric import ThomasFieringGenerator

# Load monthly streamflow data
Q_monthly = pd.read_csv('monthly_flows.csv', index_col=0, parse_dates=True)

# Initialize generator
gen = ThomasFieringGenerator(Q_monthly)

# Preprocess: apply Stedinger-Taylor normalization
gen.preprocessing()

# Fit: estimate monthly parameters
gen.fit()

# Examine fitted parameters
print(f"Monthly means (normalized):\n{gen.mu_monthly}")
print(f"Monthly std devs (normalized):\n{gen.sigma_monthly}")
print(f"Monthly lag-1 correlations:\n{gen.rho_monthly}")

# Generate 100 realizations of 50 years each
ensemble = gen.generate(n_years=50, n_realizations=100, seed=42)

# Access results
print(f"Generated {ensemble.metadata.n_realizations} realizations")
print(f"Each has {len(ensemble.data_by_realization[0])} months")

# Get first realization
first_real = ensemble.data_by_realization[0]
print(first_real.head())

# Get monthly statistics from ensemble
import numpy as np
all_data = pd.concat([ensemble.data_by_realization[r] for r in range(100)], axis=0)
monthly_stats = all_data.groupby(all_data.index.month).agg(['mean', 'std'])
print(monthly_stats)
```

---

## Advanced Usage

### Generate from daily data (automatic resampling)

```python
# Daily data will be resampled to monthly
Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)

gen = ThomasFieringGenerator(Q_daily)
gen.preprocessing()  # Automatically resamples to monthly
gen.fit()
ensemble = gen.generate(n_years=20, n_realizations=50, seed=42)
```

### Generate specific number of timesteps

```python
# Generate exactly 37 months (3 years + 1 month)
ensemble = gen.generate(n_timesteps=37, n_realizations=10, seed=42)
```

### Access fitted parameters

```python
gen.preprocessing()
gen.fit()

# FittedParams object
params = gen.fitted_params_

print(f"Number of parameters: {params.n_parameters_}")  # 48 (12 months × 4)
print(f"Training period: {params.training_period_}")
print(f"Sample size: {params.sample_size_}")

# Access transformation parameters
tau_values = params.transformations_['stedinger_transform']['tau_monthly']
print(f"Lower bounds by month:\n{tau_values}")
```

---

## Verification Checklist

When implementing or validating Thomas-Fiering:

- [ ] **Monthly parameters correct**: 12 values for μ, σ, ρ, τ
- [ ] **Lag-1 correlations valid**: All ρₘ ∈ [-1, 1]
- [ ] **Non-negative flows**: No negative values in output
- [ ] **Reproducible**: Same seed gives same results
- [ ] **AR(1) formula**: Correctly implements variance-adjusted correlation
- [ ] **Stedinger transform**: Lower bounds computed correctly
- [ ] **Monthly frequency**: Output has 'MS' frequency
- [ ] **Ensemble structure**: Returns Ensemble object with realizations
- [ ] **Statistical preservation**: Monthly means/stds approximately match (ensemble average)
- [ ] **No NaN/Inf propagation**: Invalid values handled appropriately

---

## Comparison to Other Methods

| Aspect | Thomas-Fiering | Multi-Site HMM | Kirsch-Nowak |
|--------|----------------|----------------|--------------|
| **Temporal resolution** | Monthly | Annual | Monthly |
| **Spatial** | Univariate | Multisite | Multisite |
| **Correlation** | Lag-1 (AR1) | Full temporal + spatial | Full temporal + spatial |
| **Distributional** | Normal (Stedinger) | Log-normal mixture | Normal (transformed) |
| **Seasonality** | Monthly parameters | Implicit in states | Monthly parameters |
| **Computational** | Very fast | Moderate (EM) | Fast |
| **Sample size** | 2+ years | 20+ years | 10+ years |
| **Parameters** | 48 (12 × 4) | O(n_states · n_sites²) | O(12 · n_sites²) |
| **Best for** | Single-site monthly | Multi-site regimes | Multi-site monthly |

---

## Historical Context

The Thomas-Fiering method (1962) was one of the first widely-adopted stochastic streamflow generators. It revolutionized reservoir simulation and water resources planning by:

1. **Enabling Monte Carlo analysis**: Generate many equiprobable sequences
2. **Preserving key statistics**: Monthly means, std devs, lag-1 correlation
3. **Computational simplicity**: Fast generation, easy to implement
4. **Operational applicability**: Suitable for planning horizons (months-years)

The Stedinger-Taylor improvement (1982) addressed the main limitation: skewness in streamflow distributions. By estimating a lower bound before log transformation, it better approximates normality and reduces bias.

**Legacy**: Thomas-Fiering remains widely used in:
- Reservoir operation studies
- Water supply reliability analysis
- Hydropower planning
- Educational demonstrations of stochastic hydrology

**Modern alternatives**: For more complex applications (multi-site, extreme events, non-stationarity), consider:
- Multivariate AR models (spatial correlations)
- Hidden Markov Models (regime-dependent distributions)
- Nonparametric methods (fewer assumptions)
- Machine learning approaches (complex patterns)
