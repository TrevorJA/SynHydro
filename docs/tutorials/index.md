# Tutorials

Step-by-step guides, each focused on a single SynHydro workflow.

| Tutorial | What you'll learn |
|----------|-------------------|
| [01 - Quickstart](01_quickstart.md) | Three-step generator workflow: preprocessing, fit, generate |
| [02 - Multi-Site](02_multisite.md) | Multi-site generation with spatial correlation preservation |
| [03 - Pipeline](03_pipeline.md) | Monthly-to-daily disaggregation via `KirschNowakPipeline` |
| [04 - Drought Analysis](04_drought_analysis.md) | SSI calculation and drought event extraction |
| [05 - Ensemble Validation](05_validation.md) | `validate_ensemble` metrics and `plot_validation_panel` |
| [06 - Plotting Walkthrough](06_plotting.md) | Default plots for ensemble visualization and validation |

All examples use `synhydro.load_example_data()`, which returns a multi-site
daily streamflow `DataFrame`.
