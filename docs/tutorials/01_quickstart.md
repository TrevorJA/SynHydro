# Quickstart: Single-Site Monthly Generation

This tutorial demonstrates the core SynHydro workflow: load observed data,
fit a generator, generate synthetic realizations, and plot the results.

## Load example data

```python
import synhydro

Q_daily = synhydro.load_example_data()       # multi-site daily DataFrame
Q_monthly = Q_daily.resample("MS").mean()    # aggregate to monthly
site = Q_monthly.columns[0]                  # pick one site
Q_single = Q_monthly[[site]]                 # keep as DataFrame
```

## Fit and generate

Every generator follows two steps: **fit**, **generate**.
`ThomasFieringGenerator` fits a seasonal AR(1) model to single-site monthly data.

```python
gen = synhydro.ThomasFieringGenerator()
gen.fit(Q_single)
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

The returned `Ensemble` contains 50 synthetic 30-year monthly timeseries.

```python
Q_syn_0 = ensemble.data_by_realization[0]
print(Q_syn_0.shape)  # (360, 1) — 30 years × 12 months
```

!!! note "Seed reproducibility"
    Passing the same `seed` value always produces identical results.

## Visualize

```python
from synhydro.plotting import plot_timeseries, plot_flow_duration_curve

fig, ax = plot_timeseries(
    ensemble,
    observed=Q_monthly[site],
    show_members=3,
)

fig, ax = plot_flow_duration_curve(
    ensemble,
    observed=Q_monthly[site],
)
```

## Next steps

- **Multi-site generation** → [Tutorial 02](02_multisite.md)
- **Monthly-to-daily pipeline** → [Tutorial 03](03_pipeline.md)
- **Algorithm details** → [Thomas-Fiering AR(1)](../algorithms/thomas_fiering.md)
