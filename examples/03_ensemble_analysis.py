"""
Example 3: Ensemble Analysis and Statistics

This example demonstrates working with Ensemble objects for analysis:
1. Generate a synthetic ensemble using Kirsch-Nowak pipeline
2. Access data by-site vs by-realization
3. Compute summary statistics and percentiles
4. Subset ensemble by time period and realizations
5. Resample from daily to monthly to annual frequency
6. Create validation plots comparing historical vs synthetic

Key parameters to explore:
- Different percentile levels for uncertainty quantification
- Subsetting strategies (time periods, sites, realizations)
- Resampling frequencies (daily, monthly, annual, seasonal)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sglib import load_example_data, Ensemble
from sglib.pipelines import KirschNowakPipeline
from sglib.plotting import plot_validation_panel

# ============================================================================
# Configuration
# ============================================================================
N_YEARS = 30              # Years of synthetic data
N_REALIZATIONS = 100      # Number of synthetic traces
SEED = 456                # Random seed
N_NEIGHBORS = 5           # KNN neighbors for disaggregation

# ============================================================================
# Generate ensemble (daily flows)
# ============================================================================
Q_daily = load_example_data('usgs_daily_streamflow_cms')
site = Q_daily.columns[1]

# Initialize Kirsch-Nowak pipeline (generates monthly, disaggregates to daily)
pipeline = KirschNowakPipeline(
    Q_daily[[site]],
    generate_using_log_flow=True,
    n_neighbors=N_NEIGHBORS
)
pipeline.preprocessing()
pipeline.fit()

ensemble_daily = Ensemble.from_generator(pipeline, n_years=N_YEARS, n_realizations=N_REALIZATIONS, seed=SEED)

# ============================================================================
# Access data in different ways
# ============================================================================
# By realization: All sites for a single realization
real_0 = ensemble_daily.data_by_realization[0]

# By site: All realizations for a single site
site_data = ensemble_daily.data_by_site[site]

print(f"Realization 0 shape: {real_0.shape}")
print(f"Site {site} data shape: {site_data.shape}")
print()

# ============================================================================
# Compute summary statistics
# ============================================================================
stats = ensemble_daily.summary(by='site')
print("Summary statistics by site:")
print(stats)
print()

# Compute percentiles
percentiles = ensemble_daily.percentile([10, 50, 90], by='site')
print(f"Percentile keys: {list(percentiles.keys())}")
print(f"Percentile columns: {percentiles[site].columns.tolist()}")
print()

# ============================================================================
# Subset ensemble
# ============================================================================
# Subset to first 10 years and first 20 realizations
subset = ensemble_daily.subset(
    realizations=list(range(20)),
    start_date=ensemble_daily.data_by_realization[0].index[0],
    end_date=ensemble_daily.data_by_realization[0].index[0] + pd.DateOffset(years=10)
)

print(f"Original: {len(ensemble_daily.realization_ids)} realizations")
print(f"Subset: {len(subset.realization_ids)} realizations")
print()

# ============================================================================
# Resample from daily to monthly to annual
# ============================================================================
ensemble_monthly = ensemble_daily.resample('MS')
ensemble_annual = ensemble_monthly.resample('YS')

print(f"Daily timesteps: {len(ensemble_daily.data_by_realization[0])}")
print(f"Monthly timesteps: {len(ensemble_monthly.data_by_realization[0])}")
print(f"Annual timesteps: {len(ensemble_annual.data_by_realization[0])}")
print()

# ============================================================================
# Visualization: Multi-panel validation plot
# ============================================================================
# Resample observed data to monthly for comparison
Q_daily_site = Q_daily[[site]]
Q_monthly_obs = Q_daily_site.resample('MS').sum()

fig = plot_validation_panel(
    ensemble_monthly,
    observed=Q_monthly_obs.iloc[:, 0],
    timestep='monthly',
    log_space=False
)

plt.savefig('examples/figures/03_ensemble_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Generated ensemble with {N_REALIZATIONS} realizations")
print(f"Demonstrated: data access, statistics, subsetting, resampling")
print(f"Validation plot saved to examples/figures/03_ensemble_analysis.png")
