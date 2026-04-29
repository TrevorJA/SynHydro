# HMM-KNN Generator

| | |
|---|---|
| **Type** | Parametric (HMM regime layer) + Nonparametric (KNN bootstrap layer) |
| **Resolution** | Annual |
| **Sites** | Multisite |

## Overview

The HMM-KNN generator produces synthetic annual multisite streamflow by coupling a discrete Hidden Markov Model (HMM) state sequencer with K-Nearest Neighbor (KNN) bootstrapped resampling. The HMM learns K hidden hydrologic regimes (e.g., dry and wet states) from log-annual flows and provides a probabilistic sequence of regime transitions. Within each regime-transition category, a KNN search conditioned on the previous year's log-flows selects a historical analog year, and the full multisite flow vector for that year is directly resampled. This design preserves empirical marginal distributions, multisite spatial dependence, and low-frequency regime persistence without imposing parametric emission assumptions.

### Attribution

This generator is not attributable to a single primary publication. It combines two independent developments from the literature:

* The non-parametric category (NPC) framework of Prairie et al. (2008), which conditions a KNN bootstrap on a discrete regime-transition category. Prairie (2008) used a non-homogeneous Markov chain (NHM) with kernel-smoothed transition probabilities, not a hidden Markov model.
* The hidden-Markov regime-state representation introduced for hydrologic applications by Akintug and Rasmussen (2005) and applied at multi-site annual scale by Gold et al. (2024).

The HMM substitutes for Prairie's NHM as the upper-layer regime sequencer; the lower-layer KNN bootstrap follows Prairie (2008) directly, with the rank-based kernel weights of Lall and Sharma (1996).

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathbf{Q}_t \in \mathbb{R}^S$ | Observed annual flow vector at time $t$ across $S$ sites |
| $\hat{\mathbf{Q}}_t$ | Synthetic annual flow vector at time $t$ |
| $\mathbf{Y}_t \in \mathbb{R}^S$ | Log-transformed flow vector, $Y_{t,s} = \ln(Q_{t,s} + \delta)$ |
| $K$ | Number of hidden states |
| $s_t \in \{0, \ldots, K-1\}$ | Hidden state at time $t$ |
| $\mathbf{A} \in \mathbb{R}^{K \times K}$ | Transition probability matrix, $A_{ij} = P(s_{t+1} = j \mid s_t = i)$ |
| $\boldsymbol{\pi} \in \mathbb{R}^K$ | Stationary distribution of the Markov chain |
| $c_t = (s_{t-1}, s_t)$ | Regime-transition category at time $t$ |
| $\mathcal{P}(c)$ | Pool of historical years whose (previous state, current state) pair equals $c$ |
| $n_c$ | Number of years in pool $\mathcal{P}(c)$ |
| $k_c$ | Number of nearest neighbors, $k_c = \lceil\sqrt{n_c}\rceil$ |
| $d_i$ | Euclidean distance from the query to historical pool member $i$ in normalized log-flow space |
| $w_i$ | Lall-Sharma weight for rank-$i$ neighbor |
| $\delta$ | Additive offset before log transform (default 1.0) |
| $N$ | Number of years in the historical record |

## Formulation

### Model Structure

The generator defines a two-layer structure. The upper layer is a first-order discrete Markov chain over $K$ hidden states governing low-frequency regime behavior:

$$
P(s_t = j \mid s_{t-1} = i) = A_{ij}, \qquad \sum_{j=0}^{K-1} A_{ij} = 1 \quad \forall\, i
$$

States are indexed so that state 0 corresponds to the driest regime (lowest mean log-flow at the first site) and higher indices correspond to progressively wetter regimes.

The lower layer is a nonparametric KNN resampler conditioned on the regime-transition category $c_t = (s_{t-1}, s_t)$ and the previous synthetic log-flow vector $\mathbf{Y}_{t-1}$. The synthetic flow at time $t$ is drawn directly from the historical record rather than from a parametric emission distribution:

$$
\hat{\mathbf{Q}}_t = \mathbf{Q}_{\text{obs}}[j^*], \qquad j^* \text{ sampled from the KNN-weighted pool } \mathcal{P}(c_t)
$$

### Parameter Estimation

**HMM fitting.** The log-transformed observations $\mathbf{Y}_t = \ln(\mathbf{Q}_t + \delta)$ are used as input to a Gaussian HMM (GaussianHMM from hmmlearn). All parameters -- emission means, emission covariances, transition matrix, and initial state distribution -- are estimated jointly via the Baum-Welch (Expectation-Maximization) algorithm. Multiple random initializations (controlled by n_init) are used to mitigate convergence to poor local optima, retaining the fit with the highest log-likelihood.

**Viterbi decoding.** After fitting, the most likely state sequence $\{s_1, \ldots, s_N\}$ for the historical record is obtained via the Viterbi algorithm. States are reordered by ascending mean log-flow at the first site so that state 0 is always the driest regime.

**Stationary distribution.** The stationary distribution $\boldsymbol{\pi}$ satisfies $\boldsymbol{\pi}^\top \mathbf{A} = \boldsymbol{\pi}^\top$ and is computed as the normalized left eigenvector of $\mathbf{A}$ corresponding to eigenvalue 1.

**Feature normalization.** For KNN distance computation, log-flow vectors are normalized by the historical standard deviation of each site to give equal weight to all sites regardless of scale.

### Synthesis Procedure

1. Draw the initial hidden state from the stationary distribution: $s_1 \sim \text{Categorical}(\boldsymbol{\pi})$.

2. For $t = 1$ (first year), form the pool $\mathcal{P}(s_1)$ of all historical years assigned to state $s_1$ by the Viterbi sequence. Sample one year uniformly at random from this pool (no previous synthetic value exists for distance conditioning). Set $\hat{\mathbf{Q}}_1 = \mathbf{Q}_{\text{obs}}[j]$.

3. For each subsequent year $t = 2, \ldots, T$:

   a. Draw the next state from the transition row:

   $$
   s_t \sim \text{Categorical}(\mathbf{A}_{s_{t-1}, \cdot})
   $$

   b. Form the regime-transition category $c_t = (s_{t-1}, s_t)$ and assemble the pool $\mathcal{P}(c_t)$ of all historical years $i \in \{2, \ldots, N\}$ such that $(s_{i-1}^{\text{hist}}, s_i^{\text{hist}}) = c_t$. If the pool is empty (the category was never observed historically), fall back to the state-only pool $\mathcal{P}(s_t)$.

   c. Compute the number of neighbors $k_c = \lceil\sqrt{n_c}\rceil$ where $n_c = |\mathcal{P}(c_t)|$.

   d. For each pool member $i$ with historical index $i_{\text{obs}}$, compute the normalized Euclidean distance between the previous synthetic log-flow and the log-flow in the year immediately preceding $i_{\text{obs}}$ in the historical record:

   $$
   d_i = \left\| \frac{\mathbf{Y}_{t-1}^{\text{syn}} - \mathbf{Y}_{i_{\text{obs}}-1}^{\text{hist}}}{\boldsymbol{\sigma}_{\text{hist}}} \right\|_2
   $$

   e. Identify the $k_c$ nearest neighbors (smallest $d_i$). Assign Lall-Sharma weights to the ranked neighbors ($r = 1$ = nearest):

   $$
   w_r = \frac{1/r}{\sum_{j=1}^{k_c} 1/j}
   $$

   f. Sample one neighbor $j^*$ according to $\{w_r\}$. Set $\hat{\mathbf{Q}}_t = \mathbf{Q}_{\text{obs}}[j^*]$.

4. Enforce non-negativity: $\hat{Q}_{t,s} = \max(\hat{Q}_{t,s},\, 0)$ for all sites.

## Statistical Properties

Because synthetic flows are drawn directly from the historical record, the generator exactly preserves empirical univariate marginal distributions at each site and all pairwise spatial correlations within each historical year. Regime persistence -- the tendency for dry or wet conditions to last multiple years -- is captured through the HMM transition structure rather than through parametric autocorrelation. The KNN conditioning on the previous year's flow further induces year-to-year lag-1 dependence consistent with the historical data within each regime.

The method can reproduce multi-year drought and pluvial sequences when the HMM transition matrix assigns high self-transition probability to dry or wet states. Low-frequency variability at scales longer than the calibration record cannot be reproduced beyond what the HMM transition structure implies; the generator does not extrapolate beyond observed flow magnitudes since all output values are historical observations.

With a small historical record (fewer than roughly 30 years), some regime-transition categories may contain very few pool members, reducing the effective conditioning power of the KNN step and increasing sampling variability.

## Limitations

- Output flows are constrained to observed magnitudes; the generator cannot produce flow values outside the historical range.
- First-order Markov assumption; persistence at lags beyond one year arises only from state self-transition probabilities.
- Regime-transition categories with no historical occurrences require a fallback to state-only pools, weakening the conditioning.
- HMM Baum-Welch fitting may converge to local optima; multiple random initializations (n_init) are recommended.
- With many sites (more than roughly 20), Gaussian HMM fitting requires large sample sizes to reliably estimate full covariance matrices used internally.
- Annual resolution only; the method does not disaggregate to sub-annual timesteps.

## References

**KNN bootstrap layer (lower):**
- Prairie, J., Rajagopalan, B., Lall, U., and Fulp, T. (2008). A stochastic nonparametric approach for streamflow generation combining observational and paleoreconstructed data. *Water Resources Research*, 44, W06423. https://doi.org/10.1029/2007WR006684
- Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic time series. *Water Resources Research*, 32(3), 679-693. https://doi.org/10.1029/95WR02966

**HMM regime layer (upper):**
- Akintug, B., and Rasmussen, P.F. (2005). A Markov switching model for annual hydrologic time series. *Water Resources Research*, 41(9). https://doi.org/10.1029/2004WR003605
- Gold, D.F., Reed, P.M., and Gupta, R.S. (2024). Exploring the spatially compounding multi-sectoral drought vulnerabilities in Colorado's West Slope river basins. *Earth's Future*. https://doi.org/10.1029/2024EF004841
- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286. https://doi.org/10.1109/5.18626

---

**Implementation:** `src/synhydro/methods/generation/parametric/hmm_knn.py`
