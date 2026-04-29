# Algorithms

SynHydro implements the following stochastic generation and disaggregation methods.

## Generation Methods

| Algorithm | Type | Resolution | Sites |
|-----------|------|------------|-------|
| [Thomas-Fiering AR(1)](thomas_fiering.md) | Parametric | Monthly | Univariate |
| [Matalas MAR(1)](matalas.md) | Parametric | Monthly | Multisite |
| [ARFIMA](arfima.md) | Parametric | Monthly/Annual | Univariate |
| [SPARTA](sparta.md) | Parametric | Monthly | Multisite |
| [SMARTA](smarta.md) | Parametric | Annual | Multisite |
| [Multi-Site HMM](multisite_hmm.md) | Parametric | Annual | Multisite |
| [HMM-KNN](hmm_knn.md) | Parametric | Annual | Multisite |
| [WARM](warm.md) | Parametric | Annual | Univariate |
| [Kirsch Bootstrap](kirsch.md) | Nonparametric | Monthly | Multisite |
| [KNN Bootstrap](knn_bootstrap.md) | Nonparametric | Monthly/Annual | Univariate/Multisite |
| [Phase Randomization](phase_randomization.md) | Nonparametric | Daily | Univariate |
| [Multisite Phase Randomization](multisite_phase_randomization.md) | Nonparametric | Daily | Multisite |

## Disaggregation Methods

| Algorithm | Type | Resolution |
|-----------|------|------------|
| [Nowak KNN](nowak_disaggregation.md) | Nonparametric | Monthly to Daily |
| [Valencia-Schaake](valencia_schaake.md) | Parametric | Annual to Monthly |

## Key Properties Preserved

| Property | Thomas-Fiering | Kirsch | Matalas | Phase Random | WARM | HMM | ARFIMA | KNN Bootstrap | SMARTA | SPARTA |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | x | x | x | - | - | - | x | x | - | x |
| Temporal correlation | x | x | x | x | x | x | x | x | x | x |
| Spatial correlation | - | x | x | - | - | x | - | x | x | x |
| Long-range persistence | - | - | - | - | x | - | x | - | x | - |
| Non-stationarity | - | - | - | - | x | - | - | - | - | - |
| Drought states | - | - | - | - | - | x | - | - | - | - |
| Power spectrum | - | - | - | x | x | - | x | - | - | - |
| Arbitrary marginals | - | - | - | - | - | - | - | - | x | x |
| Empirical distribution | - | x | - | - | - | - | - | x | - | - |
