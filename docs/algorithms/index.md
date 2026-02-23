# Algorithms

SGLib implements the following stochastic generation and disaggregation methods.

## Generation Methods

| Algorithm | Class | Type | Frequency | Sites |
|-----------|-------|------|-----------|-------|
| [Thomas-Fiering AR(1)](thomas_fiering.md) | `ThomasFieringGenerator` | Parametric | Monthly | Single |
| [Kirsch Bootstrap](kirsch.md) | `KirschGenerator` | Nonparametric | Monthly | Multi |
| [Matalas MAR(1)](matalas.md) | `MATALASGenerator` | Parametric | Monthly | Multi |
| [Phase Randomization](phase_randomization.md) | `PhaseRandomizationGenerator` | Nonparametric | Daily | Single |
| [WARM](warm.md) | `WARMGenerator` | Parametric | Annual | Single |
| [Multi-Site HMM](multisite_hmm.md) | `MultiSiteHMMGenerator` | Parametric | Annual | Multi |

## Disaggregation Methods

| Algorithm | Class | Type | Frequency |
|-----------|-------|------|-----------|
| [Nowak KNN](nowak_disaggregation.md) | `NowakDisaggregator` | Nonparametric | Monthly→Daily |

## Key Properties Preserved

| Property | Thomas-Fiering | Kirsch | Matalas | Phase Random | WARM | HMM |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Monthly means/stds | ✓ | ✓ | ✓ | — | — | — |
| Temporal correlation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Spatial correlation | — | ✓ | ✓ | — | — | ✓ |
| Non-stationarity | — | — | — | — | ✓ | — |
| Drought states | — | — | — | — | — | ✓ |
| Power spectrum | — | — | — | ✓ | ✓ | — |
