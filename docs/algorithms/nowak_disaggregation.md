# Nowak (2010) KNN Temporal Disaggregation

| | |
|---|---|
| **Type** | Nonparametric |
| **Resolution** | Monthly to Daily |
| **Sites** | Univariate / Multisite |
| **Class** | `NowakDisaggregator` |

## Overview

The Nowak disaggregator converts synthetic monthly flows to daily flows by borrowing within-month daily patterns from the closest historical analogs. For each synthetic month, it identifies the K nearest historic months by total flow, randomly selects one using Lall-Sharma kernel weights, and applies its daily proportions to the synthetic total.

## Algorithm

### Preprocessing

1. Validate and store daily observed flows.
2. Build historic monthly totals at the index gauge (sum across all sites for multi-site disaggregation).

### Fitting

For each calendar month m, fit a `sklearn.NearestNeighbors` model on the scalar historic monthly totals. Only historic months within +/- `max_month_shift` calendar days of month m's center are included in the pool.

### Disaggregation

For each synthetic monthly flow `Q_syn_m`:

1. **Find neighbors** — query the KNN model for the K nearest historic months by Euclidean distance on total flow.
2. **Select one neighbor** — draw with probability proportional to Lall-Sharma kernel weights (Lall and Sharma 1996). Neighbors are ranked by distance (rank i = 1 is closest), then selected with probability:
   ```
   K(i) = (1/i) / sum_{j=1}^{K} (1/j)
   ```
   This harmonic weighting favors closer analogs while maintaining stochastic diversity.
3. **Disaggregate** — apply the selected month's daily proportions:
   ```
   q_d = Q_syn_m * (q*_d / sum(q*_d'))
   ```
   For multi-site data, each site is disaggregated independently using the same selected analog month.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | `int` | `5` | K — number of candidate neighbors |
| `max_month_shift` | `int` | `7` | Days of calendar flexibility around each month |

## Properties Preserved

- Daily flow patterns within each month (borrowed from historical record)
- Monthly totals (exact, by construction)
- Multi-site consistency (same analog month applied to all sites)

**Not preserved:**
- Month-to-month daily transitions (each month disaggregated independently)
- Daily patterns outside historical range

## Limitations

- Cannot produce daily patterns not seen in the historical record
- Leap year handling requires proportional adjustment when analog and target months differ
- Quality depends on having a sufficiently long historical record to find good analogs

## References

**Primary:**
Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A nonparametric stochastic approach for multisite disaggregation of annual to daily streamflow. *Water Resources Research*, 46(8). https://doi.org/10.1029/2009WR008530

**See also:**
- Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693.

---

**Implementation:** `src/synhydro/methods/disaggregation/temporal/nowak.py`
**Tests:** `tests/test_nowak_disaggregator.py`
