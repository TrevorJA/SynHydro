"""
Retrieves USGS gauge data using hyriver suite.

Selects 3-4 gauges in the NE US with long, overlapping records.
"""

import numpy as np
import pandas as pd
import pygeohydro as gh
from pygeohydro import NWIS


# Use specific gauges in the NE US (Delaware River Basin and nearby) with long records
# These gauges are known to have good data quality and temporal overlap
stations = [
    '01434000',  # Delaware River at Port Jervis, NY (1904-present)
    '01438500',  # Delaware River at Montague, NJ (1940-present)
    '01463500',  # Delaware River at Trenton, NJ (1912-present)
    '01440000',  # Flat Brook near Flatbrookville, NJ (1923-present)
]

# Specify time period - use 1945-2025 to ensure good overlap
dates = ('1945-01-01', '2025-09-30')

# Retrieve data for the gauges
nwis = NWIS()
Q = nwis.get_streamflow(stations, dates)
Q.index = pd.to_datetime(Q.index.date)

print(f"Initial data shape: {Q.shape}")
print(f"Date range: {Q.index[0]} to {Q.index[-1]}")
print()

# Check data availability for each gauge
for gauge in Q.columns:
    valid_data = Q[gauge].dropna()
    if len(valid_data) > 0:
        print(f"{gauge}: {len(valid_data)} valid days, {valid_data.index[0]} to {valid_data.index[-1]}")
    else:
        print(f"{gauge}: No valid data")
print()

# Filter to continuous period where all gauges have data
# Find the date range where all columns have non-NaN values
valid_mask = Q.notna().all(axis=1) & (Q > 0).all(axis=1)
print(f"Rows where all gauges valid: {valid_mask.sum()} / {len(Q)}")

# Find longest continuous valid period
if valid_mask.sum() > 0:
    valid_groups = (valid_mask != valid_mask.shift()).cumsum()
    valid_groups_filtered = valid_groups[valid_mask]
    group_sizes = valid_groups_filtered.value_counts()
    longest_group_id = group_sizes.idxmax()
    longest_mask = valid_groups_filtered == longest_group_id
    valid_indices = longest_mask[longest_mask].index
    Q = Q.loc[valid_indices]
    print(f"Filtered to longest continuous period: {len(Q)} days")
    print(f"Final date range: {Q.index[0]} to {Q.index[-1]}")
else:
    print("WARNING: No overlapping valid data found, keeping all data")
print()

# Transform to monthly
Q_monthly = Q.resample('MS').sum()

# Export
Q.to_csv(f'./usgs_daily_streamflow_cms.csv', sep=',')
Q_monthly.to_csv(f'./usgs_monthly_streamflow_cms.csv', sep=',')

print(f"Data exported successfully!")
print(f"  - {Q.shape[1]} USGS streamflow gauges")
print(f"  - {len(Q)} daily observations")
print(f"  - {len(Q_monthly)} monthly observations")
print(f"  - Date range: {Q.index[0]} to {Q.index[-1]}")
print(f"  - Files: usgs_daily_streamflow_cms.csv, usgs_monthly_streamflow_cms.csv")

