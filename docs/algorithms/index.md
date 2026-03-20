# Algorithms

SynHydro implements the following stochastic generation and disaggregation methods.

## Generation Methods

| Algorithm | Class | Type | Frequency | Sites |
|-----------|-------|------|-----------|-------|
| [Thomas-Fiering AR(1)](thomas_fiering.md) | `ThomasFieringGenerator` | Parametric | Monthly | Single |
| [Kirsch Bootstrap](kirsch.md) | `KirschGenerator` | Nonparametric | Monthly | Multi |
| [Matalas MAR(1)](matalas.md) | `MATALASGenerator` | Parametric | Monthly | Multi |
| [Phase Randomization](phase_randomization.md) | `PhaseRandomizationGenerator` | Nonparametric | Daily | Single |
| [WARM](warm.md) | `WARMGenerator` | Parametric | Annual | Single |
| [Multi-Site HMM](multisite_hmm.md) | `MultiSiteHMMGenerator` | Parametric | Annual | Multi |
| [ARFIMA](arfima.md) | `ARFIMAGenerator` | Parametric | Monthly/Annual | Single |
| [KNN Bootstrap](knn_bootstrap.md) | `KNNBootstrapGenerator` | Nonparametric | Monthly/Annual | Single/Multi |
| [Gaussian/t-Copula](gaussian_copula.md) | `GaussianCopulaGenerator` | Parametric | Monthly | Multi |

## Disaggregation Methods

| Algorithm | Class | Type | Frequency |
|-----------|-------|------|-----------|
| [Nowak KNN](nowak_disaggregation.md) | `NowakDisaggregator` | Nonparametric | Monthly-to-Daily |
| [Valencia-Schaake](valencia_schaake.md) | `ValenciaSchaakeDisaggregator` | Parametric | Annual-to-Monthly |
| [Grygier-Stedinger](grygier_stedinger.md) | `GrygierStedingerDisaggregator` | Parametric | Annual-to-Monthly |

## Key Properties Preserved

| Property | Thomas-Fiering | Kirsch | Matalas | Phase Random | WARM | HMM | ARFIMA | KNN Bootstrap | Copula |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | x | x | x | - | - | - | x | x | x |
| Temporal correlation | x | x | x | x | x | x | x | x | x |
| Spatial correlation | - | x | x | - | - | x | - | x | x |
| Long-range persistence | - | - | - | - | x | - | x | - | - |
| Non-stationarity | - | - | - | - | x | - | - | - | - |
| Drought states | - | - | - | - | - | x | - | - | - |
| Power spectrum | - | - | - | x | x | - | x | - | - |
| Empirical distribution | - | x | - | - | - | - | - | x | x |
| Tail dependence | - | - | - | - | - | - | - | - | x (t-copula) |
