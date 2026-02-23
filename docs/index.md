# SGLib

**Synthetic Generation Library** — stochastic streamflow generation for hydrologic analysis.

[![Tests](https://github.com/TrevorJA/SGLib/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SGLib/actions/workflows/tests.yml)
[![Docs](https://github.com/TrevorJA/SGLib/actions/workflows/docs.yml/badge.svg)](https://github.com/TrevorJA/SGLib/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/TrevorJA/SGLib/blob/main/LICENSE)

SGLib provides parametric, nonparametric, and machine-learning stochastic generation methods under a unified API. All generators share the same `preprocessing() → fit() → generate()` workflow.

## Generators

| Class | Type | Frequency | Sites | Reference |
|-------|------|-----------|-------|-----------|
| [`KirschGenerator`][sglib.methods.generation.nonparametric.kirsch.KirschGenerator] | Nonparametric | Monthly | Multi | Kirsch et al. (2013) |
| [`PhaseRandomizationGenerator`][sglib.methods.generation.nonparametric.phase_randomization.PhaseRandomizationGenerator] | Nonparametric | Daily | Single | Brunner et al. (2019) |
| [`ThomasFieringGenerator`][sglib.methods.generation.parametric.thomas_fiering.ThomasFieringGenerator] | Parametric AR(1) | Monthly | Single | Thomas & Fiering (1962) |
| [`MATALASGenerator`][sglib.methods.generation.parametric.matalas.MATALASGenerator] | Parametric MAR(1) | Monthly | Multi | Matalas (1967) |
| [`MultiSiteHMMGenerator`][sglib.methods.generation.parametric.multisite_hmm.MultiSiteHMMGenerator] | Hidden Markov Model | Annual | Multi | Gold et al. (2025) |
| [`WARMGenerator`][sglib.methods.generation.parametric.warm.WARMGenerator] | Wavelet AR | Annual | Single | Nowak et al. (2011) |

## Quick Example

```python
import sglib

Q_obs = sglib.load_example_data()                       # daily DataFrame
Q_monthly = Q_obs.resample("MS").mean()                 # resample to monthly

gen = sglib.KirschGenerator(Q_monthly)
gen.preprocessing()
gen.fit()
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

## Installation

```bash
pip install sglib
```

See [Getting Started](getting-started.md) for full setup and data format details.
