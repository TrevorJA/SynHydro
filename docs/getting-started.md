# Getting Started

## Installation

SGLib is not yet published on PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/TrevorJA/SGLib.git
```

For development (editable install with dev extras):

```bash
git clone https://github.com/TrevorJA/SGLib.git
cd SGLib
pip install -e ".[dev]"
```

## Data Format

All generators expect a `pd.DataFrame` with a `DatetimeIndex`:

| Property | Requirement |
|----------|------------|
| Index | `DatetimeIndex` |
| Monthly data | `freq='MS'` (month-start) |
| Daily data | `freq='D'` |
| Columns | Site names (one column per gauge) |
| Units | cfs, MGD, or cms (consistent) |

```python
import pandas as pd

# Minimal example: single site, monthly
dates = pd.date_range("1980-01", periods=480, freq="MS")
Q_obs = pd.DataFrame({"site_A": [...]}, index=dates)
```

Use `sglib.load_example_data()` to get a ready-to-use daily dataset:

```python
import sglib

Q_daily = sglib.load_example_data()                 # USGS daily streamflow (cms)
Q_monthly = Q_daily.resample("MS").mean()           # aggregate to monthly
```

## Choosing a Generator

| Need | Generator |
|------|-----------|
| Monthly, single-site, parametric | `ThomasFieringGenerator` |
| Monthly, multi-site, parametric | `MATALASGenerator` |
| Monthly, multi-site, nonparametric | `KirschGenerator` |
| Daily, single-site | `PhaseRandomizationGenerator` |
| Annual, single-site | `WARMGenerator` |
| Annual, multi-site, drought-aware | `MultiSiteHMMGenerator` |
| Monthly→Daily disaggregation | `NowakDisaggregator` (or use a Pipeline) |

## Basic Workflow

Every generator follows the same three-step pattern:

```python
gen = sglib.ThomasFieringGenerator(Q_obs)
gen.preprocessing()                                 # validate and prepare data
gen.fit()                                           # estimate parameters
ensemble = gen.generate(n_realizations=10,          # synthetic flows
                        n_years=30,
                        seed=42)
```

See the [Tutorials](tutorials/index.md) for worked examples.
