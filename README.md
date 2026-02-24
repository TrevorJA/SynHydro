# SynHydro

[![Tests](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml/badge.svg)](https://github.com/TrevorJA/SynHydro/actions/workflows/tests.yml)

The Synthetic Generation Library (SynHydro) provides a suite of different methods for generating synthetic timeseries, with a focus on hydrologic applications.

This package seeks to support multiple different synthetic generation models using a standardized API. All of the models are built on a common Abstract Base Class (`synhydro.core.Generator`) which facilitates the standardized workflows.  Given the diversity of generation methods, the internal functionality of the different generators is customized, and each class (will eventually) require a corresponding `Option` argument. 

In addition to the basic generation capabilities, this package (will someday) provide:
- Diagnostic tools to verify the statistical properites of the generated series
- Drought identification and quantification tools
- Basic plotting resources
- Ensemble management (e.g., reformatting, transformations, and read/write to memory)

This package is a work in progress, and many of the methods are incomplete or unverified. 

## Supported generators

| Generator | Type | Frequency | Sites | Reference |
|---|---|---|---|---|
| `KirschGenerator` | Nonparametric | Monthly | Multi | Kirsch et al. (2013) |
| `PhaseRandomizationGenerator` | Nonparametric | Daily | Single | Brunner et al. (2019) |
| `ThomasFieringGenerator` | Parametric AR(1) | Monthly | Single | Thomas & Fiering (1962) |
| `MATALASGenerator` | Parametric MAR(1) | Monthly | Multi | Matalas (1967) |
| `MultiSiteHMMGenerator` | Hidden Markov Model | Annual | Multi | Gold et al. (2025) |
| `WARMGenerator` | Wavelet AR | Annual | Single | Nowak et al. (2011) |

## Planned additions

- ARFIMA (Hosking 1984) — long-memory / Hurst persistence
- Vine Copula (Fernandes et al. 2017) — modern flexible multivariate
- LSTM stochastic generator (2020+) — deep learning
