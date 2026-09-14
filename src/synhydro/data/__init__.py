"""
Example datasets shipped with SynHydro.

This package holds the CSV files returned by
``synhydro.utils.data.load_example_data``. They are located with
``importlib.resources.files("synhydro.data")`` rather than by a path
relative to the repository, so the lookup works for editable installs,
regular ``pip`` installs, and built wheels alike. The files are regenerated
by ``examples/example_data/retrieve_example_data.py``.

Files
-----
usgs_daily_streamflow_cms.csv
    Daily streamflow (cubic meters per second) for four USGS gauges in the
    Delaware River Basin, 1945 to 2025.
usgs_monthly_streamflow_cms.csv
    The same record summed to calendar months.
"""
