# Nowak KNN Temporal Disaggregation (Nowak et al., 2010)

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly to Daily |
| **Sites** | Univariate / Multisite |

## Overview

The Nowak disaggregator converts synthetic monthly flows to daily flows by borrowing within-month daily patterns from the closest historical analogs. For each synthetic month, the $K$ nearest historical months (by total flow magnitude) are identified, one is selected stochastically using either inverse-distance weighting (default) or Lall-Sharma kernel weights, and its daily flow proportions are applied to the synthetic monthly total. The method preserves monthly totals exactly by construction and maintains realistic daily flow dynamics drawn directly from the observed record.

### Relation to Nowak et al. (2010)

The original Nowak et al. (2010) paper presents the method at annual-to-daily resolution: KNN selects one donor year, and its 365-day proportions are applied to the synthetic annual total at every site. The algorithm is **timestep-agnostic**: the same logic applies to annual-to-monthly, monthly-to-daily, monthly-to-weekly, and other aggregate-to-sub-period pairings. This implementation operates at monthly-to-daily resolution by default. Operating at the monthly level yields a substantially larger pool of historical analogs (12 x N months vs. N years) and therefore better representation of within-pool sampling uncertainty. Annual-to-daily disaggregation can be approximated by chaining a monthly generator (e.g., `KirschGenerator`, `ThomasFieringGenerator`) with this disaggregator (see `KirschNowakPipeline`).

## Notation

| Symbol | Description |
|--------|-------------|
| $Q_m^{\text{syn}}$ | Synthetic total flow for month $m$ |
| $q_d^{\text{syn}}$ | Synthetic daily flow on day $d$ |
| $q_d^*$ | Observed daily flow on day $d$ of the selected analog month |
| $K$ | Number of nearest neighbors |
| $w_i$ | Selection weight for the $i$-th closest neighbor |
| $d_i$ | Euclidean distance from the synthetic monthly flow to the $i$-th neighbor |
| $b$ | Number of blending days at month boundaries |
| $N_m$ | Number of historical months in the candidate pool for month $m$ |

## Formulation

### Analog Pool Construction

For each calendar month $m$, a pool of candidate historical months is assembled. A tolerance of $\pm b_{\text{shift}}$ calendar days around the center of month $m$ is allowed, which provides flexibility in matching months near the calendar boundary. The candidate pool stores the total monthly flow and the corresponding daily flow time series for each historical month.

### Neighbor Selection

For each synthetic monthly flow $Q_m^{\text{syn}}$, the $K$ nearest historical months are found by Euclidean distance on total monthly flow. One neighbor is then drawn stochastically from the $K$ candidates using one of two weighting schemes.

**Inverse-distance weighting** (default). The selection probability for the $i$-th neighbor is proportional to the inverse of its distance:

$$
w_i = \frac{1/d_i}{\displaystyle\sum_{j=1}^{K} 1/d_j}
$$

where $d_i$ is the Euclidean distance between $Q_m^{\text{syn}}$ and the $i$-th neighbor's monthly total. This gives stronger preference to closer analogs when the distance differences are large, but approaches uniform selection when all neighbors are similarly distant.

**Lall-Sharma kernel** (Lall and Sharma, 1996). The selection probability depends only on rank, not distance:

$$
w_i = \frac{1/i}{\displaystyle\sum_{j=1}^{K} 1/j}, \qquad i = 1, \ldots, K
$$

This harmonic weighting gives the closest neighbor approximately twice the probability of the second-closest, regardless of the actual distance magnitudes.

### Proportional Disaggregation

The daily flows of the selected analog month are used as a template. Let $\{q_d^*\}_{d=1}^{D}$ denote the observed daily flows in the selected analog month (with $D$ days). The synthetic daily flows are computed by proportional scaling:

$$
q_d^{\text{syn}} = Q_m^{\text{syn}} \cdot \frac{q_d^*}{\displaystyle\sum_{d'=1}^{D} q_{d'}^*}
$$

This ensures that the synthetic daily flows sum exactly to the synthetic monthly total. For multisite data, each site is disaggregated independently using the same selected analog month, preserving inter-site consistency within each month.

### Month Boundary Smoothing

To reduce discontinuities at month transitions, an optional blending step applies a weighted average across $b$ days on each side of the boundary. After smoothing, each month is rescaled to restore the original monthly total:

$$
q_d^{\text{smoothed}} \leftarrow q_d^{\text{smoothed}} \cdot \frac{Q_m^{\text{syn}}}{\displaystyle\sum_{d'} q_{d'}^{\text{smoothed}}}
$$

### Synthesis Procedure

1. Fit a KNN model on the historical monthly flow totals for each calendar month.
2. For each synthetic monthly flow $Q_m^{\text{syn}}$:
   - Query the $K$ nearest neighbors by total flow.
   - Select one analog month using inverse-distance or Lall-Sharma kernel weights.
   - Disaggregate by applying the analog's daily proportions to $Q_m^{\text{syn}}$.
3. Optionally smooth month boundaries and rescale to preserve monthly totals.
4. Enforce non-negativity.

## Statistical Properties

Monthly totals are preserved exactly by construction. Daily flow patterns within each month are drawn from the historical record, maintaining realistic intra-monthly dynamics including storm hydrographs and recession curves. Multisite consistency within each month is preserved through joint analog selection.

Month-to-month daily transitions are not explicitly modeled, though the optional boundary blending partially addresses this. The method cannot produce daily patterns not observed in the historical record, limiting its ability to represent unprecedented extremes. Daily autocorrelation across month boundaries depends on the quality of analog matching.

## Limitations

- Cannot produce daily flow patterns outside the historical range.
- Leap year handling requires proportional adjustment when the analog and target months differ in length.
- Quality depends on having a sufficiently long historical record to find good analogs across the range of synthetic monthly totals.
- Month-to-month daily transitions may exhibit discontinuities despite blending.

## References

**Primary:**
Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46(8). https://doi.org/10.1029/2009WR008530

**See also:**
- Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/nowak.py`
