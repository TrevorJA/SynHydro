# SynHydro

[![Tests](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml)

SynHydro is a Python library for generating synthetic hydrologic timeseries using a unified, scikit-learn-style API. All generators share a common `fit()` and `generate()` workflow, and the library includes validation, drought analysis, plotting, and ensemble data management tools.

## Installation

```bash
pip install synhydro
```

To install a specific tagged release directly from GitHub instead:

```bash
pip install git+https://github.com/TrevorJA/SynHydro.git@v0.1.0
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

## Citing SynHydro

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

The DOI badge and the version DOI below are filled in after the first release is archived on Zenodo. If you use SynHydro in your research, please cite it as:

```
Amestoy, T. (2026). SynHydro: Synthetic hydrologic timeseries generation
in Python (Version 0.1.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.YYYYYYY
```

```bibtex
@software{amestoy_synhydro_2026,
  author    = {Amestoy, Trevor},
  title     = {{SynHydro}: Synthetic hydrologic timeseries generation in Python},
  version   = {0.1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.YYYYYYY},
  url       = {https://github.com/TrevorJA/SynHydro}
}
```

## Supported generators

Generators are classified into three bins by the mathematical character of their generative mechanism: **parametric** (sample from a fitted probability model), **hybrid** (combine a parametric structure with a non-parametric resampling step), and **non-parametric** (resample the historical record directly). See the [Algorithms page](https://trevorja.github.io/SynHydro/algorithms/) for full descriptions.

| Generator | Class | Frequency | Sites | Reference |
|---|---|---|---|---|
| `ThomasFieringGenerator` | Parametric | Monthly | Single | Thomas & Fiering (1962) |
| `MatalasGenerator` | Parametric | Monthly | Multi | Matalas (1967) |
| `ARFIMAGenerator` | Parametric | Monthly/Annual | Single | Hosking (1984) |
| `SPARTAGenerator` | Parametric | Monthly | Multi | Tsoukalas et al. (2018) |
| `SMARTAGenerator` | Parametric | Annual | Multi | Tsoukalas et al. (2018) |
| `MultiSiteHMMGenerator` | Parametric | Annual | Multi | Gold et al. (2024) |
| `KirschGenerator` | Hybrid | Weekly/Monthly | Multi | Kirsch et al. (2013) |
| `WARMGenerator` | Hybrid | Annual | Single | Nowak et al. (2011) |
| `PhaseRandomizationGenerator` | Hybrid | Daily | Single | Brunner et al. (2019) |
| `MultisitePhaseRandomizationGenerator` | Hybrid | Daily | Multi | Brunner & Gilleland (2020) |
| `KNNBootstrapGenerator` | Non-parametric | Monthly/Annual | Multi | Lall & Sharma (1996); Prairie et al. (2006, 2008) |

## Supported disaggregators

| Disaggregator | Direction | Reference |
|---|---|---|
| `NowakDisaggregator` | {Annual, Monthly, Weekly} to {Monthly, Weekly, Daily} | Nowak et al. (2010) |
| `ValenciaSchaakeDisaggregator` | Annual to Monthly | Valencia & Schaake (1973) |

Pre-built pipelines (`KirschNowakPipeline`, `ThomasFieringNowakPipeline`) chain generation and disaggregation in a single interface.

## Contributing

SynHydro is under active development, and contributions are welcome. For bug reports, feature requests, or discussion of new methods, please open an issue or pull request on GitHub. See [CONTRIBUTING.md](https://github.com/TrevorJA/SynHydro/blob/main/CONTRIBUTING.md) for guidelines on adding new generators and the project's development practices.

## Documentation

Full documentation including tutorials, algorithm descriptions, and API reference is available at the [project website](https://trevorja.github.io/SynHydro/).
