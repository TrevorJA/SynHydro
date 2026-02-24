# Nowak (2010) KNN Temporal Disaggregation

## Technical Specifications

| Property | Value |
|----------|-------|
| **Class** | `NowakDisaggregator` |
| **Type** | Nonparametric |
| **Input Frequency** | Monthly |
| **Output Frequency** | Daily |
| **Sites** | Single or Multi |
| **Reference** | Nowak et al. (2010) |

## Overview

The Nowak disaggregator converts synthetic monthly flows to daily flows by borrowing
within-month daily patterns from the closest historic analogs. For each synthetic month,
it identifies the $K$ nearest historic months (by total flow), randomly selects one, and
applies its daily proportions to the synthetic total.

## Algorithm Description

### Preprocessing

1. Validate and store daily observed flows $\{q^*_d\}$.
2. Build historic monthly totals $Q^*_m = \sum_{d \in m} q^*_d$ at the index gauge
   (sum across all sites for multi-site disaggregation).

### Fitting

For each calendar month $m \in \{1, \ldots, 12\}$, fit a `sklearn` `NearestNeighbors`
model on the scalar historic monthly totals. Only historic months within
$\pm$`max_month_shift` calendar days of month $m$'s center are included in the pool.

### Disaggregation

For each synthetic monthly flow $Q_{\text{syn},m}$:

1. **Find neighbors.** Query the fitted KNN model to retrieve the $K$ nearest historic
   months $\{m^*_1, \ldots, m^*_K\}$ by Euclidean distance on total flow:

   $$d_k = \left| Q_{\text{syn},m} - Q^*_{m^*_k} \right|$$

2. **Select one neighbor.** Draw index $k^*$ with probability proportional to
   $1/d_k$ (Lall-Sharma kernel):

   $$P(k^*= k) \propto \frac{1}{k}$$

   (ranks are used if distances are tied).

3. **Disaggregate.** Apply the selected month's daily proportions to the synthetic total:

   $$q_d = Q_{\text{syn},m} \cdot \frac{q^*_d}{\displaystyle\sum_{d' \in m^*_{k^*}} q^*_{d'}}$$

   For multi-site data, each site is disaggregated independently using the same selected
   analog month.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | `5` | $K$ — number of candidate neighbors |
| `max_month_shift` | `7` | Days of calendar flexibility around each month |

## Notes

- Leap year handling: if the synthetic month spans a leap day and the analog month does
  not (or vice versa), flow is proportionally adjusted.
- For multi-site data, the **index gauge** (sum of all sites) is used for KNN search;
  the same analog is then applied to all sites.

## References

Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).
A nonparametric stochastic approach for multisite disaggregation of annual to daily
streamflow. *Water Resources Research*, 46(8).

**SynHydro Implementation:** [`src/synhydro/methods/disaggregation/temporal/nowak.py`](https://github.com/TrevorJA/SynHydro/blob/main/src/synhydro/methods/disaggregation/temporal/nowak.py)
