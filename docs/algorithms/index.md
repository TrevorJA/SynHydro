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

## Disaggregation Methods

| Algorithm | Class | Type | Frequency |
|-----------|-------|------|-----------|
| [Nowak KNN](nowak_disaggregation.md) | `NowakDisaggregator` | Nonparametric | Monthly→Daily |
| [Valencia-Schaake](valencia_schaake.md) | `ValenciaSchaakeDisaggregator` | Parametric | Annual→Monthly |
| [Grygier-Stedinger](grygier_stedinger.md) | `GrygierStedingerDisaggregator` | Parametric | Annual→Monthly |

## Key Properties Preserved

| Property | Thomas-Fiering | Kirsch | Matalas | Phase Random | WARM | HMM | ARFIMA | KNN Bootstrap |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ |
| Temporal correlation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Spatial correlation | — | ✓ | ✓ | — | — | ✓ | — | ✓ |
| Long-range persistence | — | — | — | — | ✓ | — | ✓ | — |
| Non-stationarity | — | — | — | — | ✓ | — | — | — |
| Drought states | — | — | — | — | — | ✓ | — | — |
| Power spectrum | — | — | — | ✓ | ✓ | — | ✓ | — |
| Empirical distribution | — | ✓ | — | — | — | — | — | ✓ |
