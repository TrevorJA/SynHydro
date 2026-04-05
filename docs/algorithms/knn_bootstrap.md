# K-Nearest Neighbor Bootstrap (Lall and Sharma, 1996)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly / Annual |
| **Sites** | Univariate / Multisite |

## Overview

The K-Nearest Neighbor (KNN) bootstrap generates synthetic streamflow by conditionally resampling from the historical record. At each time step, the most recent generated value defines a query point in feature space; the $K$ closest historical states are identified, and one is selected with probability inversely proportional to its rank (Lall-Sharma kernel). The generated value is the historical successor of the selected neighbor. This nonparametric approach preserves the empirical marginal distribution exactly and can capture nonlinear dependence structures that parametric models may miss.

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_t \in \mathbb{R}^S$ | Observed flow vector at time $t$ across $S$ sites |
| $\hat{Q}_t$ | Synthetic flow vector at time $t$ |
| $K$ | Number of nearest neighbors |
| $N$ | Number of historical time steps |
| $w_i$ | Lall-Sharma kernel weight for the $i$-th closest neighbor |
| $\mathbf{x}_t$ | Feature vector at time $t$ (flow values at selected sites) |
| $d(\cdot, \cdot)$ | Euclidean distance in feature space |

## Formulation

### Neighbor Selection and Kernel Weights

The number of neighbors defaults to $K = \lceil \sqrt{N} \rceil$, where $N$ is the length of the historical record. The Lall-Sharma kernel assigns probability to the $i$-th closest neighbor ($i = 1, 2, \ldots, K$) as:

$$
w_i = \frac{1/i}{\displaystyle\sum_{j=1}^{K} 1/j}
$$

This harmonic weighting gives the closest neighbor approximately twice the selection probability of the second-closest, encouraging fidelity to the local neighborhood while maintaining stochastic diversity.

### Feature-Successor Structure

For each historical time step $t$, a feature-successor pair is stored:

$$
(\mathbf{x}_t,\; Q_{t+1})
$$

where $\mathbf{x}_t$ is the feature vector (flow values at time $t$ for the selected sites) and $Q_{t+1}$ is the observed flow at the next time step. For monthly data, the pairs are partitioned by calendar month to respect seasonality: at generation time, only neighbors from the same calendar month are considered.

### Multisite Extension

In multisite mode, the neighbor search is performed on a single index site or on the full multivariate feature vector using Euclidean distance:

$$
d(\mathbf{x}, \mathbf{x}') = \left\|\mathbf{x} - \mathbf{x}'\right\|_2
$$

Once a neighbor is selected, the successor vector across all $S$ sites is taken jointly, preserving spatial correlation by construction.

### Synthesis Procedure

1. Select a random historical time step as the initial state and set $\hat{Q}_1$ equal to the successor of that state.
2. For each subsequent time step $t = 2, 3, \ldots, T$:
   - Form the query feature vector $\hat{\mathbf{x}}_{t-1}$ from the most recently generated flow values.
   - Find the $K$ nearest neighbors of $\hat{\mathbf{x}}_{t-1}$ among the historical feature vectors (within the same calendar month if monthly).
   - Select one neighbor $j^*$ with probability $w_i$ based on its rank $i$.
   - Set the generated value to the historical successor:

$$
\hat{Q}_t = Q_{j^* + 1}
$$

3. For multisite data, the entire successor vector is assigned jointly.

## Statistical Properties

The empirical marginal distribution is preserved exactly, since every generated value is drawn directly from the historical record. Nonlinear dependence is captured implicitly through the conditional neighborhood structure. Lag-1 autocorrelation is approximately preserved because the successor of a similar state will tend to exhibit similar temporal dynamics. Spatial cross-correlations are maintained through joint resampling.

However, generated values cannot exceed the historical range (a fundamental bootstrap limitation). Long-range persistence beyond the conditioning lag is not explicitly modeled, and the method does not capture trends or nonstationarity. The curse of dimensionality can degrade neighbor selection quality when many sites are used simultaneously as features.

## Limitations

- Cannot generate values outside the range of the historical record.
- Sensitive to $K$: too small leads to repetitive cycling through a few neighbors; too large destroys local temporal structure.
- Curse of dimensionality for high-dimensional feature spaces (many sites).
- Requires sufficient record length (roughly 20+ years for monthly data) to avoid excessive repetition of analogs.

## References

**Primary:**
Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693. https://doi.org/10.1029/95WR02966

**See also:**
- Rajagopalan, B., and Lall, U. (1999). A k-nearest-neighbor simulator for daily precipitation and other weather variables. *Water Resources Research*, 35(10), 3089-3101. https://doi.org/10.1029/1999WR900028
- Lall, U. (1995). Recent advances in nonparametric function estimation: Hydrologic applications. *Reviews of Geophysics*, 33(S2), 1093-1102.

---

**Implementation:** `src/synhydro/methods/generation/nonparametric/knn_bootstrap.py`
