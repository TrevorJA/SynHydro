# [Method Name] ([Author et al., Year])

| | |
|---|---|
| **Type** | [Parametric / Nonparametric] |
| **Resolution** | [Monthly / Daily / Annual] |
| **Sites** | [Univariate / Multisite] |

## Overview

[2-4 sentences describing the method's core mechanism, its statistical basis, and what it is best suited for. Write for a reader with hydrology background who may be new to stochastic generation. Avoid implementation details.]

## Notation

[Table of mathematical symbols used in this document. Standardize where possible across all algorithm docs.]

| Symbol | Description |
|--------|-------------|
| $Q_t$ | Observed streamflow at time $t$ |
| $\hat{Q}_t$ | Synthetic (generated) streamflow at time $t$ |
| $N$ | Length of the historical record |
| $S$ | Number of sites |
| ... | ... |

## Formulation

### Model Structure

[Define the core mathematical model. State the key equation(s) that characterize the method. Use display math for primary equations:]

$$
\hat{Q}_t = f(\cdot)
$$

### Parameter Estimation

[Describe how model parameters are estimated from the observed data. Provide estimator equations. If multiple estimation approaches exist, present each with its assumptions.]

### Synthesis Procedure

[Step-by-step procedure for generating synthetic sequences from the fitted model. Mix numbered prose steps with display equations. This should read like a formal algorithm description, not pseudocode.]

1. [Initialize ...]
2. For each time step $t = 1, \ldots, T$:

$$
\hat{Q}_t = \ldots
$$

3. [Post-processing ...]

## Statistical Properties

[Discuss what statistical properties the model preserves and which it does not. Frame in terms of moments, correlation structure, spectral properties, or distributional characteristics. Use prose rather than checklists.]

## Limitations

- [Precise statement of known shortcoming or assumption]
- [...]

## References

**Primary:**
[Full APA citation with DOI]

**See also:**
- [Supporting citation]

---

**Implementation:** `src/synhydro/methods/[category]/[subcategory]/[file].py`
