# SynHydro

[![Tests](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml)

SynHydro is a Python library for generating synthetic hydrologic timeseries using a unified, scikit-learn-style API. All generators share a common `fit() / generate()` workflow, and the library includes validation, drought analysis, plotting, and ensemble management tools.

## Installation

```bash
pip install git+https://github.com/TrevorJA/SynHydro.git
```

## Quick example

```python
import synhydro

Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").sum()

gen = synhydro.KirschGenerator()
gen.fit(Q_monthly)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

## Supported generators

| Generator | Type | Frequency | Sites | Reference |
|---|---|---|---|---|
| `ThomasFieringGenerator` | Parametric AR(1) | Monthly | Single | Thomas & Fiering (1962) |
| `MatalasGenerator` | Parametric MAR(1) | Monthly | Multi | Matalas (1967) |
| `ARFIMAGenerator` | Fractional ARIMA | Monthly/Annual | Single | Hosking (1984) |
| `KirschGenerator` | Nonparametric Bootstrap | Monthly | Multi | Kirsch et al. (2013) |
| `KNNBootstrapGenerator` | K-Nearest Neighbor | Daily/Monthly/Annual | Multi | Lall & Sharma (1996) |
| `PhaseRandomizationGenerator` | Spectral | Daily | Single | Brunner et al. (2019) |
| `MultiSiteHMMGenerator` | Hidden Markov Model | Annual | Multi | Gold et al. (2024) |
| `WARMGenerator` | Wavelet AR | Annual | Single | Nowak et al. (2011) |

## Supported disaggregators

| Disaggregator | Direction | Reference |
|---|---|---|
| `NowakDisaggregator` | Monthly to Daily | Nowak et al. (2010) |
| `ValenciaSchaakeDisaggregator` | Annual to Monthly | Valencia & Schaake (1973) |
| `GrygierStedingerDisaggregator` | Annual to Monthly | Grygier & Stedinger (1988) |

Pre-built pipelines (`KirschNowakPipeline`, `ThomasFieringNowakPipeline`) chain generation and disaggregation in a single interface.

## Documentation

Full documentation including tutorials, algorithm descriptions, and API reference is available at the [project website](https://trevorja.github.io/SynHydro/).
