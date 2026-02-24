# Multi-Site Hidden Markov Model (Gold et al. 2025)

**Classification:** Parametric
**Temporal Resolution:** Annual
**Site Compatibility:** Multisite

---

## Technical Specifications

| Property | Specification |
|----------|---------------|
| Input data | Annual streamflow for multiple sites, minimum 2 years recommended |
| Output frequency | AS (Annual Start) or user-specified annual frequency |
| Distributional assumption | Multivariate Gaussian emissions per hidden state (after log transformation) |
| Correlation structure | Full covariance matrices per state (preserves spatial correlations) + temporal structure via state transitions |
| Temporal dependence | Markov chain (first-order) with hidden states representing hydrologic regimes |

---

## Algorithm Description

The Multi-Site Hidden Markov Model (HMM) generator uses a Gaussian Mixture Model HMM to generate synthetic streamflow across multiple sites simultaneously. The method models temporal dependencies through hidden states (e.g., dry/wet regimes) and spatial correlations through state-specific multivariate Gaussian emissions with full covariance matrices.

This approach is particularly effective for capturing drought dynamics and spatially compounding water scarcity across multiple basins, as demonstrated in Gold et al. (2025) for Colorado's West Slope river basins.

### Preprocessing

1. **Data validation**
   - Validate input as pandas DataFrame with multiple sites (columns)
   - Check for minimum sample size (recommended: 10+ years)
   - Allow site subset selection

2. **Offset application**
   - Add small offset to handle zero flows: `Q_adj = Q + offset`
   - Default offset: 1.0 (in flow units)
   - Prevents log(0) issues

3. **Log transformation**
   - Apply natural logarithm: `Q_log = log(Q_adj)`
   - Transform to log-space for Gaussian modeling
   - Store transformed data for fitting

### Fitting/Calibration

1. **GMMHMM initialization**
   - Use `hmmlearn.hmm.GMMHMM` with:
     - `n_components = n_states` (default: 2 for dry/wet)
     - `covariance_type = 'full'` (default, captures all spatial correlations)
     - Alternative: 'diag' (site-independent) or 'spherical' (single variance)
   - Set maximum iterations for Expectation-Maximization convergence

2. **Model fitting**
   - Fit GMMHMM to log-transformed flows: `model.fit(Q_log)`
   - Estimate via Baum-Welch algorithm (EM):
     - **E-step**: Compute state posteriors given parameters
     - **M-step**: Update parameters given state posteriors
   - Iterate until convergence or max iterations

3. **Parameter extraction**
   - Extract state means: μₛ ∈ ℝⁿˢⁱᵗᵉˢ for each state s
   - Extract covariance matrices: Σₛ ∈ ℝⁿˢⁱᵗᵉˢ ˣ ⁿˢⁱᵗᵉˢ for each state s
   - Extract transition matrix: P ∈ ℝⁿˢᵗᵃᵗᵉˢ ˣ ⁿˢᵗᵃᵗᵉˢ

4. **State ordering**
   - Order states by mean of first site (ascending)
   - State 0 = driest, State (n_states-1) = wettest
   - Ensures consistent state interpretation

5. **Stationary distribution computation**
   - Solve for stationary distribution π:
     - π is left eigenvector of P with eigenvalue 1
     - π · P = π
     - Σπᵢ = 1
   - Used for initial state sampling in generation

### Generation

1. **State trajectory generation**
   - Sample initial state from stationary distribution:
     ```
     s₀ ~ Categorical(π)
     ```
   - For each subsequent timestep t = 1, ..., T:
     ```
     sₜ ~ Categorical(P[sₜ₋₁, :])
     ```
   - Creates temporally coherent state sequence

2. **Emission sampling**
   - For each timestep t with state sₜ:
     ```
     Q_log[t, :] ~ MultivariateNormal(μ_sₜ, Σ_sₜ)
     ```
   - Samples preserve spatial correlations (via Σₛ)

3. **Back-transformation**
   - Inverse log transform:
     ```
     Q_syn = exp(Q_log) - offset
     ```
   - Ensures positive flows

4. **Non-negativity enforcement**
   - Apply floor: `Q_syn = max(Q_syn, 0)`
   - Handles rare numerical issues from back-transformation

5. **Time index creation**
   - Generate dates starting from observed data start
   - Use inferred frequency (typically 'AS' or 'YS')
   - Return as pandas DataFrame with proper DatetimeIndex

---

## Key Parameters

- **`n_states`**: Number of hidden states
  - Type: int
  - Default: 2
  - Notes: 2 states (dry/wet) is typical; 3+ states can model intermediate regimes. More states require more data for reliable estimation.

- **`offset`**: Value added before log transformation
  - Type: float
  - Default: 1.0
  - Notes: Should be small relative to typical flows. Too small risks numerical issues with zeros; too large biases the distribution.

- **`max_iterations`**: Maximum EM iterations for fitting
  - Type: int
  - Default: 1000
  - Notes: Increase if convergence warnings occur. Typical convergence: 50-200 iterations.

- **`covariance_type`**: Structure of state covariance matrices
  - Type: str
  - Default: 'full'
  - Options:
    - 'full': Full covariance (all correlations preserved) - **recommended**
    - 'diag': Diagonal (sites independent within states)
    - 'spherical': Single variance (spherical clusters)
  - Notes: 'full' is essential for preserving spatial correlations but requires n_sites² parameters per state.

---

## Algorithmic Details

### Gaussian Mixture Model HMM

The GMMHMM combines:
- **Hidden Markov Model**: Temporal structure via state transitions
- **Gaussian Mixture**: Multivariate emissions from each state

**Joint probability**:
```
P(Q, S) = π(s₀) · ∏ₜ P(sₜ|sₜ₋₁) · P(Qₜ|sₜ)
```

Where:
- S = {s₀, s₁, ..., sₜ}: Hidden state sequence
- Q = {Q₀, Q₁, ..., Qₜ}: Observed flows (log-space)
- π: Stationary distribution
- P(sₜ|sₜ₋₁): Transition probabilities
- P(Qₜ|sₜ) = 𝒩(Qₜ; μₛₜ, Σₛₜ): Multivariate normal emissions

### Multivariate Gaussian Emissions

For state s, emissions follow:
```
Q_log ~ 𝒩(μₛ, Σₛ)
```

**Mean vector**: μₛ ∈ ℝⁿ where n = number of sites

**Covariance matrix** (for 'full' type):
```
Σₛ = [
  σ₁₁  σ₁₂  ...  σ₁ₙ
  σ₂₁  σ₂₂  ...  σ₂ₙ
  ...
  σₙ₁  σₙ₂  ...  σₙₙ
]
```

Where σᵢⱼ captures correlation between sites i and j in state s.

**Probability density**:
```
p(Q) = (2π)^(-n/2) |Σₛ|^(-1/2) exp[-0.5(Q - μₛ)ᵀ Σₛ⁻¹ (Q - μₛ)]
```

### Stationary Distribution Computation

Solve for eigenvector of transition matrix transpose:
```
Pᵀ · π = 1 · π
```

Algorithm:
1. Compute eigendecomposition: `Pᵀ = V · Λ · V⁻¹`
2. Find eigenvector vᵢ corresponding to λᵢ ≈ 1
3. Normalize: `π = vᵢ / Σvᵢ`

### Baum-Welch (EM) Algorithm

**Expectation (E) step**: Compute forward-backward probabilities
```
Forward: α(sₜ) = P(Q₁:ₜ, sₜ)
Backward: β(sₜ) = P(Qₜ₊₁:ₜ | sₜ)
Posterior: γ(sₜ) = P(sₜ | Q) = α(sₜ)β(sₜ) / P(Q)
```

**Maximization (M) step**: Update parameters
```
πₛ = γ(s₀)
Pᵢⱼ = Σₜ ξ(sₜ=i, sₜ₊₁=j) / Σₜ γ(sₜ=i)
μₛ = Σₜ γ(sₜ=s) Qₜ / Σₜ γ(sₜ=s)
Σₛ = Σₜ γ(sₜ=s) (Qₜ - μₛ)(Qₜ - μₛ)ᵀ / Σₜ γ(sₜ=s)
```

---

## Algorithm Variations

- **Variable number of states**: n_states > 2 can model additional regimes (e.g., dry/normal/wet)
- **Diagonal covariance**: Faster fitting but ignores spatial correlations
- **Tied covariance**: Single Σ shared across states (more parsimonious)
- **Non-parametric emissions**: Replace Gaussian with histograms or kernel density estimates
- **Seasonal HMM**: Separate models per season or use time-varying transition matrices
- **Continuous-time HMM**: Model sub-annual dynamics with continuous-time Markov processes

---

## Implementation Notes

### Computational complexity

- **Time complexity**: O(n_states² · n_timesteps · n_sites³) per EM iteration
  - Dominated by covariance matrix operations
  - Typical: 50-200 EM iterations for convergence
- **Space complexity**: O(n_states · n_sites²) for covariance storage
- **Scalability**:
  - Efficient for typical applications: 2-10 sites, 10-100 years, 2-3 states
  - Full covariance becomes expensive for n_sites > 20
  - Consider diagonal covariance for large n_sites

### Limitations

- **Sample size**: Requires sufficient data for reliable parameter estimation
  - Minimum: ~10 years
  - Recommended: 20+ years for 2-3 states, 50+ years for more states
- **First-order Markov**: Only captures one-timestep memory
  - May miss multi-year persistence (e.g., droughts spanning 3+ years)
  - Consider higher-order HMMs or AR-HMM hybrids for longer memory
- **Stationarity assumption**: Assumes transition probabilities are time-invariant
  - Not suitable for data with trends or regime shifts
  - Consider time-varying or non-stationary HMM variants
- **Gaussian assumption**: Emissions assumed Gaussian in log-space
  - May not capture extreme tail behavior perfectly
  - Consider mixture models or Student-t emissions for heavy tails
- **Annual timestep**: Designed for annual data
  - Can handle monthly/daily but computationally expensive
  - State interpretation may differ at finer resolutions

### Special handling

- **Zeros and negative flows**: Offset prevents log(0); negatives from back-transform are floored to 0
- **Covariance matrix**: May be non-positive-definite if sample size is small
  - hmmlearn handles regularization internally
  - Increase sample size or reduce n_states if fitting fails
- **State label switching**: HMM likelihood is invariant to state label permutation
  - State ordering by mean ensures consistent interpretation
  - Still, different random seeds may converge to different local optima
- **EM convergence**: May converge to local optimum
  - Use multiple random initializations and select best by likelihood
  - Check convergence warnings; increase max_iterations if needed
- **Degenerate states**: One state may "absorb" most observations
  - Indicates n_states too large for data
  - Reduce n_states or increase sample size

### Hydrologic properties preserved

**Explicitly preserved:**
- Spatial correlations (via full covariance matrices Σₛ)
- Temporal persistence (via transition matrix P)
- Regime-dependent distributions (dry vs wet states have different μₛ, Σₛ)
- Marginal distributions in log-space (Gaussian mixture)

**Implicitly preserved (approximately):**
- Drought frequency (via state persistence)
- Drought spatial extent (via spatial correlations in dry state)
- Flow magnitudes per regime (via state-specific means)

**Not preserved:**
- Exact observed autocorrelation at lags > 1 (first-order Markov)
- Non-Gaussian marginal distributions (log-transformation imposes log-normality)
- Trends or non-stationarity
- Sub-annual seasonality (annual timestep)

---

## References

**Primary:**
Gold, D.F., Reed, P.M., & Gupta, R.S. (In Revision). Exploring the Spatially Compounding Multi-sectoral Drought Vulnerabilities in Colorado's West Slope River Basins. *Earth's Future*.

**Methodological foundations:**
- **Hidden Markov Models**: Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- **Gaussian Mixture Models**: Reynolds, D.A., & Rose, R.C. (1995). Robust text-independent speaker identification using Gaussian mixture speaker models. *IEEE Transactions on Speech and Audio Processing*, 3(1), 72-83.
- **GMMHMM for hydrology**: Akintug, B., & Rasmussen, P.F. (2005). A Markov switching model for annual hydrologic time series. *Water Resources Research*, 41(9).
- **Baum-Welch algorithm**: Bilmes, J.A. (1998). A gentle tutorial of the EM algorithm and its application to parameter estimation for Gaussian mixture and hidden Markov models. *International Computer Science Institute*, 4(510), 126.

**Python implementation:**
- hmmlearn: https://hmmlearn.readthedocs.io/

**Original application:**
- Gold et al. GitHub repository: https://github.com/TrevorJA/Multisite_GMMHMM

---

**SynHydro Implementation:** [`src/synhydro/methods/generation/parametric/multisite_hmm.py`](https://github.com/TrevorJA/SynHydro/blob/main/src/synhydro/methods/generation/parametric/multisite_hmm.py)

**Tests:** [`tests/test_multisite_hmm_generator.py`](https://github.com/TrevorJA/SynHydro/blob/main/tests/test_multisite_hmm_generator.py)

---

## Usage Example

```python
import pandas as pd
from synhydro.methods.generation.parametric import MultiSiteHMMGenerator

# Load annual streamflow data for multiple sites
Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
# Expected format: DatetimeIndex with annual frequency, columns = sites

# Initialize generator with 2 states (dry/wet)
gen = MultiSiteHMMGenerator(
    Q_annual,
    n_states=2,              # Dry and wet states
    offset=1.0,              # Offset for log transform
    covariance_type='full'   # Preserve spatial correlations
)

# Preprocess: apply log transformation
gen.preprocessing()

# Fit: estimate HMM parameters
gen.fit(random_state=42)

# Examine fitted parameters
print(f"State means (log-space):\n{gen.means_}")
print(f"Transition matrix:\n{gen.transition_matrix_}")
print(f"Stationary distribution: {gen.stationary_distribution_}")

# Generate 100 realizations of 50 years each
ensemble = gen.generate(n_realizations=100, n_years=50, seed=42)

# Access results
print(f"Generated {ensemble.metadata.n_realizations} realizations")
print(f"Each has {len(ensemble.data_by_realization[0])} years")

# Get first realization
first_real = ensemble.data_by_realization[0]
print(first_real.head())

# Get all realizations for a specific site
site_data = ensemble.data_by_site['site_1']
print(f"Site 1 data shape: {site_data.shape}")  # (n_years, n_realizations)
```

---

## Advanced Usage

### Multi-state model for finer regime resolution

```python
# 3-state model: dry, normal, wet
gen = MultiSiteHMMGenerator(Q_annual, n_states=3)
gen.preprocessing()
gen.fit(random_state=42)

# Interpret states (ordered by mean)
print("State 0 (Dry):", gen.means_[0])
print("State 1 (Normal):", gen.means_[1])
print("State 2 (Wet):", gen.means_[2])
```

### Diagonal covariance for faster fitting

```python
# When spatial correlations are less important
gen = MultiSiteHMMGenerator(
    Q_annual,
    n_states=2,
    covariance_type='diag'  # Faster, but ignores correlations
)
gen.preprocessing()
gen.fit()
```

### Site subset selection

```python
# Generate only for subset of sites
gen = MultiSiteHMMGenerator(Q_annual)
gen.preprocessing(sites=['site_A', 'site_B', 'site_C'])
gen.fit()
ensemble = gen.generate(n_realizations=50, n_years=30)
```

---

## Verification Checklist

When implementing or validating Multi-Site HMM:

- [ ] **Parameter shapes correct**: means (n_states, n_sites), covariances (n_states, n_sites, n_sites)
- [ ] **Transition matrix valid**: rows sum to 1, all entries in [0, 1]
- [ ] **Stationary distribution valid**: sums to 1, is left eigenvector of P
- [ ] **State ordering**: states ordered by mean (dry to wet)
- [ ] **Covariance matrices positive semi-definite**: all eigenvalues ≥ 0
- [ ] **Generation reproducible**: same seed gives same results
- [ ] **Non-negative flows**: no negative values in synthetic output
- [ ] **Spatial correlations preserved**: synthetic corr ≈ observed corr (ensemble average)
- [ ] **Temporal persistence**: state autocorrelation matches transition probabilities
- [ ] **Ensemble structure**: proper Ensemble object with by-site and by-realization views
- [ ] **Drought characteristics**: frequency and spatial extent similar to observed (in dry state)

---

## Comparison to Other Methods

| Aspect | Multi-Site HMM | Kirsch-Nowak | Phase Randomization |
|--------|---------------|--------------|---------------------|
| **Temporal resolution** | Annual | Monthly | Daily |
| **Spatial** | Multisite | Multisite | Univariate |
| **Correlation** | Full spatial + temporal | Full spatial + temporal | Temporal only (full spectrum) |
| **Distributional** | Log-normal mixture | Normal (transformed) | Kappa or empirical |
| **Regime modeling** | Explicit (hidden states) | Implicit (correlations) | None |
| **Drought dynamics** | Excellent (state-based) | Good (via correlations) | Good (via persistence) |
| **Computational** | Moderate (EM iterations) | Fast | Fast (FFT) |
| **Sample size** | 20+ years recommended | 10+ years | 10+ years |
| **Extrapolation** | Yes (beyond observed) | Yes (via distributions) | Depends on marginal choice |
