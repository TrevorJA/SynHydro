# KNN Bootstrap Generator (Lall and Sharma 1996)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate / Multisite |
| **Class** | `KNNBootstrapGenerator` |

## Overview

The K-Nearest Neighbor (KNN) bootstrap generates synthetic streamflow by conditionally resampling from the historical record. At each timestep, the current flow value determines a neighborhood of K similar historical states, and the next value is drawn from the successors of those neighbors using kernel-weighted probabilities. This nonparametric approach preserves the empirical marginal distribution exactly and captures nonlinear dependence structures that parametric models may miss.

For multisite applications, all sites are resampled jointly using the same selected neighbor index, preserving spatial correlation by construction.

## Algorithm

### Preprocessing

1. **Validate input** as univariate or multisite DataFrame with DatetimeIndex.
2. **Construct state vectors**: for each timestep t, define the feature vector used for neighbor search. Default: the flow value(s) at time t.
   - For monthly data with lag-1 conditioning: feature = Q(t)
   - For monthly data with seasonal conditioning: feature = [Q(t), month(t)]
3. **Build successor pairs**: for each historical timestep t, store (feature_t, Q_{t+1}) so that neighbors of the current state yield candidate next values.

### Fitting

1. **Determine K** (number of neighbors). Default heuristic:
   ```
   K = ceil(sqrt(n))
   ```
   where n is the number of historical timesteps. Can also be set manually.

2. **Fit KNN model** using `sklearn.NearestNeighbors` on the historical feature vectors.

3. **Compute Lall-Sharma kernel weights** for neighbor selection:
   ```
   K(i) = (1/i) / sum_{j=1}^{K} (1/j)
   ```
   where i is the rank of the neighbor (i=1 is closest). This harmonic kernel gives the closest neighbor approximately twice the weight of the second-closest.

4. **Store** the fitted KNN model, the historical feature-successor pairs, and the kernel weights.

### Generation

1. **Initialize** by randomly selecting a historical timestep as the starting state.
2. **For each subsequent timestep**:
   a. Query the KNN model for the K nearest neighbors of the current state.
   b. Select one neighbor with probability K(i) (Lall-Sharma kernel).
   c. The generated value is the **successor** of the selected neighbor in the historical record:
      ```
      Q_syn(t+1) = Q_obs(neighbor_t + 1)
      ```
   d. Update the current state to Q_syn(t+1) for the next iteration.
3. **Multisite**: use the index site (or multivariate distance) for neighbor search, then take the successor vector across all sites jointly.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | `int` | `None` | K; if None, uses ceil(sqrt(n)) |
| `feature_cols` | `list` | `None` | Columns to use as features for KNN search. If None, uses all site columns |
| `index_site` | `str` | `None` | Site to use for distance computation in multisite mode. If None, uses multivariate distance |
| `block_size` | `int` | `1` | Number of consecutive timesteps to resample as a block (1 = standard KNN) |

## Properties Preserved

- Empirical marginal distribution (resampled values are historical observations)
- Nonlinear dependence structure (via conditional resampling)
- Lag-1 autocorrelation (approximately, via nearest-neighbor conditioning)
- Spatial cross-correlations (via joint resampling in multisite mode)

**Not preserved:**
- Values outside the historical range (bootstrap limitation)
- Long-range persistence beyond the conditioning lag
- Trends or non-stationarity

## Limitations

- Cannot generate values outside the range of the historical record
- Sensitive to the choice of K: too small causes repetitive sequences, too large destroys temporal structure
- Curse of dimensionality for high-dimensional feature spaces (many sites)
- Successor-based resampling can create discontinuities at December-January boundaries if not handled explicitly
- Requires at least 20 years for monthly data to avoid excessive repetition of analogs

## References

**Primary:**
Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693. https://doi.org/10.1029/95WR02966

**See also:**
- Rajagopalan, B., and Lall, U. (1999). A k-nearest-neighbor simulator for daily precipitation and other weather variables. *Water Resources Research*, 35(10), 3089-3101. https://doi.org/10.1029/1999WR900028
- Lall, U. (1995). Recent advances in nonparametric function estimation: Hydrologic applications. *Reviews of Geophysics*, 33(S2), 1093-1102.

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/knn_bootstrap.py`
**Tests:** `tests/test_knn_bootstrap_generator.py`
