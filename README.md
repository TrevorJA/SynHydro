# SynHydro

[![Tests](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml)

SynHydro is a Python library for generating synthetic hydrologic timeseries using a unified, scikit-learn-style API. All generators share a common `fit()` and `generate()` workflow, and the library includes validation, drought analysis, plotting, and ensemble data management tools.

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
| `GaussianCopulaGenerator` | Copula (Gaussian/t) | Monthly | Multi | Pereira et al. (2017) |
| `VineCopulaGenerator` | Vine copula | Monthly | Multi | Yu et al. (2025) |
| `SPARTAGenerator` | PAR-to-Anything | Monthly | Multi | Tsoukalas et al. (2018) |
| `SMARTAGenerator` | SMA-to-Anything | Annual | Multi | Tsoukalas et al. (2018) |
| `MultiSiteHMMGenerator` | Hidden Markov Model | Annual | Multi | Gold et al. (2024) |
| `HMMKNNGenerator` | HMM + KNN resampling | Annual | Multi | Prairie et al. (2008) |
| `WARMGenerator` | Wavelet AR | Annual | Single | Nowak et al. (2011) |
| `KirschGenerator` | Nonparametric Bootstrap | Monthly | Multi | Kirsch et al. (2013) |
| `KNNBootstrapGenerator` | K-Nearest Neighbor | Daily/Monthly/Annual | Multi | Lall & Sharma (1996) |
| `PhaseRandomizationGenerator` | Spectral | Daily | Single | Brunner et al. (2019) |
| `MultisitePhaseRandomizationGenerator` | Wavelet phase-random | Daily | Multi | Brunner & Gilleland (2020) |

## Supported disaggregators

| Disaggregator | Direction | Reference |
|---|---|---|
| `NowakDisaggregator` | Monthly to Daily | Nowak et al. (2010) |
| `ValenciaSchaakeDisaggregator` | Annual to Monthly | Valencia & Schaake (1973) |
| `GrygierStedingerDisaggregator` | Annual to Monthly | Grygier & Stedinger (1988) |

Pre-built pipelines (`KirschNowakPipeline`, `ThomasFieringNowakPipeline`) chain generation and disaggregation in a single interface.

## Contributing

SynHydro is under active development, and contributions are welcome. For bug reports, feature requests, or discussion of new methods, please open an issue or pull request on GitHub. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new generators and the project's development practices.

## Documentation

Full documentation including tutorials, algorithm descriptions, and API reference is available at the [project website](https://trevorja.github.io/SynHydro/).
