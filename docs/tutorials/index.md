# Tutorials

Concise examples covering the core SynHydro workflows.

| Tutorial | Generator | Description |
|----------|-----------|-------------|
| [Quickstart](01_quickstart.md) | `ThomasFieringGenerator` | Single-site monthly generation |
| [Multi-Site Monthly](02_multisite.md) | `KirschGenerator` | Multi-site nonparametric bootstrap |
| [Monthly→Daily Pipeline](03_pipeline.md) | `KirschNowakPipeline` | Full monthly-to-daily workflow |
| [Drought Analysis](04_drought_analysis.md) | `SSIDroughtMetrics` | SSI calculation and drought metrics |

All examples use `synhydro.load_example_data()`, which returns a multi-site daily streamflow `DataFrame`.
