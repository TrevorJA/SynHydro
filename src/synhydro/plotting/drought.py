"""
Drought-specific plotting functions for SynHydro.

This module provides functions for visualizing drought characteristics,
SSI timeseries, and drought frequency analyses.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from synhydro.core.ensemble import Ensemble
from synhydro.droughts.ssi import SSIDroughtMetrics, get_drought_metrics
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes, apply_default_styling, save_figure, get_site_data,
    subset_date_range, compute_ensemble_percentiles, format_date_axis,
    validate_ensemble_input, validate_observed_input
)


def plot_drought_characteristics(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    x_metric: str = 'magnitude',
    y_metric: str = 'duration',
    color_metric: str = 'severity',
    threshold: float = -1.0,
    method: str = 'ssi',
    window: int = 12,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['square_figsize'],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    cmap: str = 'viridis_r',
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of drought characteristics.

    Shows drought events in 2D space (e.g., magnitude vs duration) with
    color indicating a third metric (e.g., severity).

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to analyze. If None, uses first site in ensemble
    x_metric : str, default 'magnitude'
        Metric for x-axis: 'magnitude', 'duration', 'severity'
    y_metric : str, default 'duration'
        Metric for y-axis: 'magnitude', 'duration', 'severity'
    color_metric : str, default 'severity'
        Metric for color mapping: 'magnitude', 'duration', 'severity'
    threshold : float, default -1.0
        SSI threshold for drought identification
    method : str, default 'ssi'
        Drought identification method (currently only 'ssi' supported)
    window : int, default 12
        Rolling window size for SSI calculation (months)
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses x_metric
    ylabel : str, optional
        Y-axis label. If None, uses y_metric
    legend : bool, default True
        Whether to display legend
    cmap : str, default 'viridis_r'
        Colormap for scatter points
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Additional arguments passed to matplotlib scatter

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_drought_characteristics(ensemble, observed=Q_obs)
    >>> plot_drought_characteristics(ensemble, x_metric='duration', y_metric='severity')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    if method != 'ssi':
        raise ValueError(f"Only 'ssi' method is currently supported, got '{method}'")

    valid_metrics = ['magnitude', 'duration', 'severity']
    for metric_name, metric_val in [('x_metric', x_metric), ('y_metric', y_metric),
                                     ('color_metric', color_metric)]:
        if metric_val not in valid_metrics:
            raise ValueError(f"{metric_name} must be one of {valid_metrics}, got '{metric_val}'")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Calculate SSI and drought metrics for ensemble
    # Aggregate ensemble to get median realization for drought analysis
    ensemble_median = site_data.median(axis=1)

    ssi_calc = SSIDroughtMetrics(timescale='M', window=window)
    ssi_ensemble = ssi_calc.calculate_ssi(ensemble_median)
    drought_metrics_ensemble = get_drought_metrics(ssi_ensemble)

    # Calculate for observed if provided
    drought_metrics_obs = None
    if observed is not None:
        ssi_obs = ssi_calc.calculate_ssi(observed)
        drought_metrics_obs = get_drought_metrics(ssi_obs)

    # Determine color scale range (use both datasets if available)
    max_color_val = drought_metrics_ensemble[color_metric].abs().max()
    if drought_metrics_obs is not None:
        max_color_val = max(max_color_val,
                           drought_metrics_obs[color_metric].abs().max())

    # Plot ensemble drought characteristics
    if len(drought_metrics_ensemble) > 0:
        scatter_ens = ax.scatter(
            drought_metrics_ensemble[x_metric],
            -drought_metrics_ensemble[y_metric],  # Negative for duration
            c=drought_metrics_ensemble[color_metric],
            cmap=cmap,
            s=100,
            alpha=0.5,
            edgecolor='none',
            vmin=0,
            vmax=max_color_val,
            label='Ensemble',
            **kwargs
        )

    # Plot observed drought characteristics
    if drought_metrics_obs is not None and len(drought_metrics_obs) > 0:
        scatter_obs = ax.scatter(
            drought_metrics_obs[x_metric],
            -drought_metrics_obs[y_metric],  # Negative for duration
            c=drought_metrics_obs[color_metric],
            cmap=cmap,
            s=100,
            alpha=1.0,
            edgecolor='k',
            linewidth=1.5,
            vmin=0,
            vmax=max_color_val,
            label='Observed',
            **kwargs
        )

    # Add colorbar
    if len(drought_metrics_ensemble) > 0:
        cbar = plt.colorbar(scatter_ens, ax=ax)
        cbar.set_label(color_metric.capitalize(), fontsize=LAYOUT['label_fontsize'])

    # Set labels
    if xlabel is None:
        xlabel = x_metric.capitalize()
    if ylabel is None:
        ylabel = f'-{y_metric.capitalize()}'  # Negative sign for duration
    if title is None:
        title = f'Drought Characteristics - {site_name}'

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid=True, log_scale=False)

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_ssi_timeseries(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    percentiles: Optional[list] = [10, 50, 90],
    window: int = 12,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['default_figsize'],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot SSI timeseries with drought period shading.

    Shows Standardized Streamflow Index over time with shaded regions
    indicating drought severity levels.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to analyze. If None, uses first site in ensemble
    percentiles : list, optional
        Percentiles for ensemble uncertainty bands. Default [10, 50, 90]
    window : int, default 12
        Rolling window size for SSI calculation (months)
    start_date : str, optional
        Start date for plot
    end_date : str, optional
        End date for plot
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses 'Date'
    ylabel : str, optional
        Y-axis label. If None, uses 'SSI'
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Additional arguments passed to matplotlib plot functions

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_ssi_timeseries(ensemble, observed=Q_obs, window=12)
    >>> plot_ssi_timeseries(ensemble, start_date='2000-01-01', end_date='2010-12-31')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Calculate SSI for each realization
    ssi_calc = SSIDroughtMetrics(timescale='M', window=window)
    ssi_realizations = {}

    for real_id in site_data.columns:
        try:
            ssi_realizations[real_id] = ssi_calc.calculate_ssi(site_data[real_id])
        except Exception as e:
            # Skip realizations that fail SSI calculation
            continue

    # Convert to DataFrame
    if len(ssi_realizations) == 0:
        raise ValueError("Failed to calculate SSI for any realizations")

    ssi_df = pd.DataFrame(ssi_realizations)

    # Subset by date range
    ssi_df = subset_date_range(ssi_df, start_date, end_date)

    # Add drought threshold shading
    # Moderate drought: SSI < -1
    ax.axhspan(-3, -2, alpha=0.1, color=COLORS['drought_extreme'],
              label='Extreme Drought (SSI < -2)')
    ax.axhspan(-2, -1, alpha=0.1, color=COLORS['drought_severe'],
              label='Severe Drought (-2 < SSI < -1)')
    ax.axhspan(-1, 0, alpha=0.1, color=COLORS['drought_moderate'],
              label='Moderate Drought (-1 < SSI < 0)')

    # Add zero line
    ax.axhline(0, color=COLORS['zero_line'], linewidth=1, linestyle='--')

    # Plot percentiles
    if percentiles is not None and len(percentiles) > 0:
        perc_data = compute_ensemble_percentiles(ssi_df, percentiles)

        sorted_percs = sorted(percentiles)

        # Fill between outer percentiles
        if len(sorted_percs) >= 2:
            ax.fill_between(perc_data.index,
                          perc_data[f'p{sorted_percs[0]}'],
                          perc_data[f'p{sorted_percs[-1]}'],
                          color=COLORS['ensemble_fill'],
                          alpha=STYLE['fill_alpha'],
                          label=f'Ensemble {sorted_percs[0]}-{sorted_percs[-1]}th %ile')

        # Plot median
        if 50 in percentiles:
            ax.plot(perc_data.index, perc_data['p50'],
                   color=COLORS['ensemble_median'],
                   linewidth=STYLE['ensemble_linewidth'],
                   label='Ensemble Median')

    # Plot observed SSI if provided
    if observed is not None:
        ssi_obs = ssi_calc.calculate_ssi(observed)
        ssi_obs = subset_date_range(ssi_obs.to_frame(), start_date, end_date).iloc[:, 0]

        ax.plot(ssi_obs.index, ssi_obs.values,
               color=COLORS['observed'],
               linewidth=STYLE['observed_linewidth'],
               label='Observed')

    # Set labels
    if xlabel is None:
        xlabel = 'Date'
    if ylabel is None:
        ylabel = 'Standardized Streamflow Index (SSI)'
    if title is None:
        title = f'SSI Timeseries ({window}-month window) - {site_name}'

    # Format date axis
    format_date_axis(ax, ssi_df)

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)

    # Set y-axis limits to show drought thresholds
    y_min = min(-3.5, ssi_df.min().min() - 0.5)
    y_max = max(2, ssi_df.max().max() + 0.5)
    ax.set_ylim(y_min, y_max)

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax
