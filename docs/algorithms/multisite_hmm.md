# Multi-Site Hidden Markov Model (Gold et al. 2024)

| | |
|---|---|
| **Type** | Parametric |
| **Resolution** | Annual |
| **Sites** | Multisite |
| **Class** | `MultiSiteHMMGenerator` |

## Overview

The Multi-Site HMM uses a Gaussian Mixture Model HMM to generate synthetic streamflow across multiple sites simultaneously. Hidden states represent hydrologic regimes (e.g., dry/wet), with state-specific multivariate Gaussian emissions capturing spatial correlations via full covariance matrices. Temporal dependence arises from the Markov state transition structure. This approach is particularly effective for modeling drought dynamics and spatially compounding water scarcity.

## Algorithm

### Preprocessing

1. Validate input as multi-site DataFrame; optionally select site subset.
2. Add offset to handle zeros: `Q_adj = Q + offset` (default: 1.0).
3. Log-transform: `Q_log = log(Q_adj)`.

### Fitting

1. **Initialize GMMHMM** via `hmmlearn.hmm.GMMHMM`:
   - `n_components = n_states` (default 2: dry/wet)
   - `covariance_type = 'full'` (preserves spatial correlations)
2. **Fit** via Baum-Welch (EM) algorithm on log-transformed flows:
   - E-step: compute state posteriors via forward-backward
   - M-step: update state means, covariances, and transition probabilities
3. **Order states** by mean of first site (ascending: driest → wettest).
4. **Compute stationary distribution** — solve for left eigenvector of transition matrix with eigenvalue 1.

### Generation

1. **State trajectory** — sample initial state from stationary distribution, then at each timestep sample next state from transition matrix row.
2. **Emission sampling** — for each timestep with state s:
   ```
   Q_log[t, :] ~ MultivariateNormal(mu_s, Sigma_s)
   ```
3. **Back-transform**: `Q_syn = exp(Q_log) - offset`, then clip negatives to 0.
4. **Build output** — create DataFrame with DatetimeIndex at inferred frequency.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | `int` | `2` | Number of hidden states (2 = dry/wet) |
| `offset` | `float` | `1.0` | Additive offset before log transform |
| `max_iterations` | `int` | `1000` | Maximum EM iterations |
| `covariance_type` | `str` | `'full'` | Covariance structure: `'full'`, `'diag'`, or `'spherical'` |

## Properties Preserved

- Spatial correlations (via full covariance matrices per state)
- Temporal persistence (via Markov state transitions)
- Regime-dependent distributions (distinct mean/covariance per state)
- Drought frequency and spatial extent (via dry-state persistence)

**Not preserved:**
- Autocorrelation at lags > 1 (first-order Markov)
- Non-Gaussian marginal distributions (log-normality imposed)
- Trends or non-stationarity

## Limitations

- Requires 20+ years for 2 states; 50+ for more states
- First-order Markov — may miss multi-year drought persistence
- Full covariance becomes expensive for n_sites > 20 (consider `'diag'`)
- EM may converge to local optima; multiple initializations recommended
- State label switching: ordering by mean ensures consistency but different seeds may find different optima

## References

**Primary:**
Gold, D.F., Reed, P.M., and Gupta, R.S. (2024). Exploring the spatially compounding multi-sectoral drought vulnerabilities in Colorado's West Slope river basins. *Earth's Future*. https://doi.org/10.1029/2023EF004126

**See also:**
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Akintug, B., and Rasmussen, P.F. (2005). A Markov switching model for annual hydrologic time series. *Water Resources Research*, 41(9).

---

**Implementation:** `src/synhydro/methods/generation/parametric/multisite_hmm.py`
**Tests:** `tests/test_multisite_hmm_generator.py`
