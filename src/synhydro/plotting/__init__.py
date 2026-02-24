"""
SynHydro Plotting Module
======================

Comprehensive plotting functions for ensemble hydrologic data analysis.

This module provides standardized plotting functions with consistent APIs
for visualizing ensemble timeseries, distributions, correlations, drought
metrics, and validation statistics.

Quick Start
-----------
>>> from synhydro import Ensemble
>>> from synhydro.plotting import plot_timeseries, plot_flow_duration_curve

>>> # Create ensemble
>>> ensemble = Ensemble.from_hdf5('flows.h5')

>>> # Plot timeseries with uncertainty
>>> fig, ax = plot_timeseries(ensemble, observed=Q_obs, site='site_A')

>>> # Plot flow duration curve
>>> fig, ax = plot_flow_duration_curve(ensemble, observed=Q_obs)

Module Organization
-------------------
- **timeseries**: Timeseries and temporal flow ranges
- **distributions**: Histograms, FDCs, and monthly distributions
- **correlation**: Autocorrelation and spatial correlation
- **drought**: SSI timeseries and drought characteristics
- **validation**: Multi-panel statistical validation
- **config**: Centralized plotting configuration

Design Principles
-----------------
1. **Ensemble-First**: All functions natively accept Ensemble objects
2. **Consistent API**: Standardized arguments across all functions
3. **Composable**: Support for matplotlib axes for multi-panel layouts
4. **Auto-Save**: Optional filename parameter for convenient output
5. **Publication-Ready**: Default styling suitable for papers/presentations

Function Categories
-------------------

Timeseries Plots
~~~~~~~~~~~~~~~~
- plot_timeseries: Ensemble timeseries with uncertainty bands
- plot_flow_ranges: Min/max/median ranges by period (daily/weekly/monthly)

Distribution Plots
~~~~~~~~~~~~~~~~~~
- plot_flow_duration_curve: FDC with ensemble uncertainty
- plot_histogram: Flow value histograms with optional KDE
- plot_monthly_distributions: Monthly boxplots/violin plots

Correlation Plots
~~~~~~~~~~~~~~~~~
- plot_autocorrelation: Temporal autocorrelation with ensemble range
- plot_spatial_correlation: Multi-site correlation heatmaps

Drought Plots
~~~~~~~~~~~~~
- plot_drought_characteristics: Scatter plot of drought metrics
- plot_ssi_timeseries: SSI over time with drought period shading

Validation Plots
~~~~~~~~~~~~~~~~
- plot_validation_panel: Multi-panel statistical validation (5 panels)
"""

# Import configuration first
from .config import COLORS, STYLE, LAYOUT, LABELS, apply_plotting_style

# Import plotting functions
from .timeseries import (
    plot_timeseries,
    plot_flow_ranges,
)

from .distributions import (
    plot_flow_duration_curve,
    plot_histogram,
    plot_monthly_distributions,
)

from .correlation import (
    plot_autocorrelation,
    plot_spatial_correlation,
)

from .drought import (
    plot_drought_characteristics,
    plot_ssi_timeseries,
)

from .validation import (
    plot_validation_panel,
)

# Define public API
__all__ = [
    # Configuration
    'COLORS',
    'STYLE',
    'LAYOUT',
    'LABELS',
    'apply_plotting_style',

    # Timeseries plots
    'plot_timeseries',
    'plot_flow_ranges',

    # Distribution plots
    'plot_flow_duration_curve',
    'plot_histogram',
    'plot_monthly_distributions',

    # Correlation plots
    'plot_autocorrelation',
    'plot_spatial_correlation',

    # Drought plots
    'plot_drought_characteristics',
    'plot_ssi_timeseries',

    # Validation plots
    'plot_validation_panel',
]

# Version info
__version__ = '2.0.0'
__author__ = 'SynHydro Development Team'
