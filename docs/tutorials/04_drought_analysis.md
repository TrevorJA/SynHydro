# Drought Analysis

SGLib provides Standardized Streamflow Index (SSI) calculation and drought metric extraction
via `SSIDroughtMetrics`.

## SSI Calculation

```python
import sglib

Q_daily = sglib.load_example_data()
Q_monthly = Q_daily.resample("MS").mean()
site = Q_monthly.columns[0]

# Calculate 12-month SSI using gamma distribution
ssi_calc = sglib.SSIDroughtMetrics(timescale="M", window=12, dist="gamma")
ssi = ssi_calc.calculate_ssi(Q_monthly[site])
```

The SSI series has mean ≈ 0 and std ≈ 1. Values below −1 indicate moderate-to-severe drought.

## Drought Metrics

```python
metrics = sglib.get_drought_metrics(ssi)
print(metrics.head())
```

Returned columns include: `duration`, `magnitude`, `severity` (minimum SSI), and
`avg_severity` for each identified drought event.

## Comparing Observed vs. Synthetic

```python
gen = sglib.KirschGenerator(Q_monthly)
gen.preprocessing()
gen.fit()
ensemble = gen.generate(n_realizations=20, n_years=30, seed=42)

# Compute SSI for each realization
syn_ssi_list = []
for i, Q_syn in ensemble.data_by_realization.items():
    ssi_syn = ssi_calc.calculate_ssi(Q_syn.iloc[:, 0])
    syn_ssi_list.append(ssi_syn)
```

**Algorithm details:** [Thomas-Fiering AR(1)](../algorithms/thomas_fiering.md) · [Kirsch Bootstrap](../algorithms/kirsch.md)
