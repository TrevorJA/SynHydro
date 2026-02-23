# Monthly→Daily Pipeline

`KirschNowakPipeline` combines the Kirsch monthly generator with the Nowak KNN
disaggregator. It accepts only daily observed data and handles internal aggregation.

```python
import sglib

Q_daily = sglib.load_example_data()                     # daily observed flows

pipeline = sglib.KirschNowakPipeline(Q_daily)
pipeline.preprocessing()
pipeline.fit()
daily_ensemble = pipeline.generate(n_realizations=10, n_years=30, seed=42)
```

The output ensemble contains daily synthetic flows, preserving both the monthly
statistical structure (from Kirsch) and within-month daily patterns (from Nowak).

```python
# First realization — daily DataFrame
Q_syn_daily = daily_ensemble.data_by_realization[0]
print(Q_syn_daily.shape)                                 # (~10957 days × n_sites)
```

!!! tip "Thomas-Fiering + Nowak"
    A single-site monthly→daily pipeline is also available:
    ```python
    pipeline = sglib.ThomasFieringNowakPipeline(Q_daily)
    ```

**Algorithm details:** [Kirsch Bootstrap](../algorithms/kirsch.md) · [Nowak Disaggregation](../algorithms/nowak_disaggregation.md)
