# Drought Analysis with SSI

SynHydro provides the **Standardized Streamflow Index (SSI)** for
characterizing drought from streamflow data. This tutorial covers the
SSI calculation and drought event extraction workflow.

## Calculate SSI

SSI transforms raw flows into standardized anomalies by fitting a probability
distribution within a rolling window. Values below -1 indicate drought;
below -2 is extreme.

```python
import synhydro

Q_daily = synhydro.load_example_data()
Q_monthly = Q_daily.resample("MS").mean()
site = Q_monthly.columns[0]

ssi_calc = synhydro.SSI(dist="gamma", timescale=12, fit_freq="ME")
ssi_calc.fit(Q_monthly[site])
ssi = ssi_calc.get_training_ssi()

print(f"SSI mean: {ssi.mean():.3f}")   # ≈ 0
print(f"SSI std:  {ssi.std():.3f}")     # ≈ 1
```

## Visualize

```python
from synhydro.plotting import plot_ssi_timeseries

fig, ax = plot_ssi_timeseries(ssi, title=f"SSI-12 - {site}")
```

Shaded zones mark moderate (-1 to -1.5), severe (-1.5 to -2), and
extreme (< -2) drought conditions.

## Extract drought events

`get_drought_metrics` identifies contiguous periods where SSI stays below -1:

```python
metrics = synhydro.get_drought_metrics(ssi)
print(metrics[["start", "end", "duration", "severity", "avg_severity"]].head())
```

Each row is a drought event with its duration, severity (minimum SSI),
and magnitude (cumulative deficit).

!!! tip "Choosing a distribution"
    Use `compare_distributions` to rank candidate distributions by AIC and
    Kolmogorov-Smirnov test:
    ```python
    results = synhydro.compare_distributions(Q_monthly[site].dropna().values)
    print(results)
    ```

## Next steps

- **Ensemble validation** - [Tutorial 05](05_validation.md)
- **Algorithm details** - [Kirsch Bootstrap](../algorithms/kirsch.md)
