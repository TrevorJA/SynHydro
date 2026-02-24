# Multi-Site Monthly Generation

`KirschGenerator` uses nonparametric bootstrap with Cholesky decomposition to preserve
cross-site correlations in monthly streamflow.

```python
import synhydro

# Multi-site monthly data
Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").mean()

gen = synhydro.KirschGenerator(Q_monthly)
gen.preprocessing()
gen.fit()
ensemble = gen.generate(n_realizations=50, n_years=30, seed=42)
```

```python
# Access by realization or by site
Q_syn_0 = ensemble.data_by_realization[0]      # shape: (360, n_sites)
Q_site_A = ensemble.data_by_site[Q_monthly.columns[0]]

# Cross-site correlations are preserved
print(Q_syn_0.corr())
```

**Algorithm details:** [Kirsch Bootstrap](../algorithms/kirsch.md)
