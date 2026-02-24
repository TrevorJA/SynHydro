# Quickstart: Single-Site Monthly Generation

The `ThomasFieringGenerator` fits a seasonal AR(1) model to single-site monthly streamflow.

```python
import synhydro

# Load example data and extract a single site
Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").mean().iloc[:, [0]]   # single-site monthly

# Fit and generate
gen = synhydro.ThomasFieringGenerator(Q_monthly)
gen.preprocessing()
gen.fit()
ensemble = gen.generate(n_realizations=10, n_years=30, seed=42)
```

The returned `ensemble` object contains 10 synthetic 30-year monthly timeseries.

```python
# Access realizations
Q_syn_0 = ensemble.data_by_realization[0]    # first realization (DataFrame)
print(Q_syn_0.shape)                          # (360, 1) — 360 months × 1 site

# Summary of fitted parameters
gen.summary()
```

!!! note "Seed reproducibility"
    Passing the same `seed` value to `generate()` always produces identical results.

**Algorithm details:** [Thomas-Fiering AR(1)](../algorithms/thomas_fiering.md)
