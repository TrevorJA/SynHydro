# Algorithms

SynHydro implements the following stochastic generation and disaggregation methods.

## Generation Methods

| Algorithm | Type | Resolution | Sites |
|-----------|------|------------|-------|
| [Thomas-Fiering AR(1)](thomas_fiering.md) | Parametric | Monthly | Univariate |
| [Kirsch Bootstrap](kirsch.md) | Nonparametric | Monthly | Multisite |
| [Matalas MAR(1)](matalas.md) | Parametric | Monthly | Multisite |
| [Phase Randomization](phase_randomization.md) | Nonparametric | Daily | Univariate |
| [WARM](warm.md) | Parametric | Annual | Univariate |
| [Multi-Site HMM](multisite_hmm.md) | Parametric | Annual | Multisite |
| [ARFIMA](arfima.md) | Parametric | Monthly/Annual | Univariate |
| [KNN Bootstrap](knn_bootstrap.md) | Nonparametric | Monthly/Annual | Univariate/Multisite |
| [Gaussian/t-Copula](gaussian_copula.md) | Parametric | Monthly | Multisite |
| [Vine Copula](vine_copula.md) | Parametric | Monthly | Multisite |

## Disaggregation Methods

| Algorithm | Type | Resolution |
|-----------|------|------------|
| [Nowak KNN](nowak_disaggregation.md) | Nonparametric | Monthly to Daily |
| [Valencia-Schaake](valencia_schaake.md) | Parametric | Annual to Monthly |
| [Grygier-Stedinger](grygier_stedinger.md) | Parametric | Annual to Monthly |

## Key Properties Preserved

| Property | Thomas-Fiering | Kirsch | Matalas | Phase Random | WARM | HMM | ARFIMA | KNN Bootstrap | Copula | Vine Copula |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | x | x | x | - | - | - | x | x | x | x |
| Temporal correlation | x | x | x | x | x | x | x | x | x | x |
| Spatial correlation | - | x | x | - | - | x | - | x | x | x |
| Long-range persistence | - | - | - | - | x | - | x | - | - | - |
| Non-stationarity | - | - | - | - | x | - | - | - | - | - |
| Drought states | - | - | - | - | - | x | - | - | - | - |
| Power spectrum | - | - | - | x | x | - | x | - | - | - |
| Empirical distribution | - | x | - | - | - | - | - | x | x | x |
| Tail dependence | - | - | - | - | - | - | - | - | x (t-copula) | x (asymmetric) |
