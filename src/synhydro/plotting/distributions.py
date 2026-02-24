"""
Distribution plotting functions for SynHydro.

This module provides functions for plotting flow distributions, flow duration
curves, histograms, and monthly distribution comparisons.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from scipy.stats import gaussian_kde
from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes, apply_default_styling, save_figure, get_site_data,
    get_ylabel, filter_complete_years, resample_data,
    validate_ensemble_input, validate_observed_input
)


def plot_flow_duration_curve(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    show_annual_range: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['default_figsize'],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = True,
    units: str = 'cms',
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot flow duration curve with ensemble uncertainty.

    Shows flow exceedance probabilities with ensemble range and total period FDC.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    show_annual_range : bool, default True
        Show range of annual FDCs as shaded region
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses 'Non-Exceedance Probability'
    ylabel : str, optional
        Y-axis label. If None, uses units
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_scale : bool, default True
        Use logarithmic y-axis
    units : str, default 'cms'
        Flow units for y-axis label
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
    >>> fig, ax = plot_flow_duration_curve(ensemble, observed=Q_obs)
    >>> plot_flow_duration_curve(ensemble, show_annual_range=False, log_scale=False)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Filter complete years
    site_data = filter_complete_years(site_data)
    if observed is not None:
        observed = filter_complete_years(observed.to_frame()).iloc[:, 0]

    # Define non-exceedance probabilities
    nonexceedance = np.linspace(0.0001, 0.9999, 50)

    # Calculate total period FDC for ensemble
    s_total_fdc = np.nanquantile(site_data.values.flatten(), nonexceedance)

    # Calculate annual FDCs if requested
    if show_annual_range:
        s_annual_fdcs = site_data.groupby(site_data.index.year).quantile(nonexceedance).unstack(level=0)
        s_fdc_min = s_annual_fdcs.min(axis=1)
        s_fdc_max = s_annual_fdcs.max(axis=1)

        # Plot annual range
        ax.fill_between(nonexceedance, s_fdc_min, s_fdc_max,
                       color=COLORS['ensemble_fill'],
                       alpha=STYLE['fill_alpha'],
                       label='Ensemble Annual FDC Range',
                       **kwargs)

    # Plot total period FDC
    ax.plot(nonexceedance, s_total_fdc,
           color=COLORS['ensemble_median'],
           linewidth=STYLE['ensemble_linewidth'],
           label='Ensemble Total FDC',
           **kwargs)

    # Plot observed if provided
    if observed is not None:
        h_total_fdc = np.nanquantile(observed.values.flatten(), nonexceedance)

        if show_annual_range:
            h_annual_fdcs = observed.groupby(observed.index.year).quantile(nonexceedance).unstack(level=0)
            h_fdc_min = h_annual_fdcs.min(axis=1)
            h_fdc_max = h_annual_fdcs.max(axis=1)

            ax.fill_between(nonexceedance, h_fdc_min, h_fdc_max,
                          color=COLORS['observed'],
                          alpha=0.3,
                          label='Observed Annual FDC Range',
                          **kwargs)

        ax.plot(nonexceedance, h_total_fdc,
               color=COLORS['observed'],
               linewidth=STYLE['observed_linewidth'],
               label='Observed Total FDC',
               **kwargs)

    # Set labels
    if xlabel is None:
        xlabel = 'Non-Exceedance Probability'
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f'Flow Duration Curve - {site_name}'

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale)

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_histogram(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    bins: Union[int, str] = 'auto',
    density: bool = True,
    show_kde: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['default_figsize'],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = 'cms',
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of flow values with optional KDE.

    Shows distribution of all flow values across ensemble and optionally
    overlays kernel density estimate.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    bins : int or str, default 'auto'
        Number of bins or binning strategy ('auto', 'sturges', 'fd', etc.)
    density : bool, default True
        If True, plot probability density; else raw counts
    show_kde : bool, default True
        Overlay kernel density estimate
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses units
    ylabel : str, optional
        Y-axis label. If None, uses 'Density' or 'Count'
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_scale : bool, default False
        Use logarithmic x-axis
    units : str, default 'cms'
        Flow units for x-axis label
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Additional arguments passed to matplotlib hist

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_histogram(ensemble, observed=Q_obs, bins=50)
    >>> plot_histogram(ensemble, density=False, show_kde=False)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Flatten ensemble data
    ensemble_values = site_data.values.flatten()
    ensemble_values = ensemble_values[~np.isnan(ensemble_values)]

    # Plot ensemble histogram
    ax.hist(ensemble_values, bins=bins, density=density,
           color=COLORS['ensemble_fill'], alpha=0.6,
           edgecolor=COLORS['ensemble_median'],
           label='Ensemble', **kwargs)

    # Plot observed histogram if provided
    if observed is not None:
        obs_values = observed.dropna().values
        ax.hist(obs_values, bins=bins, density=density,
               color=COLORS['observed'], alpha=0.5,
               edgecolor='black', linewidth=1.5,
               label='Observed', **kwargs)

    # Add KDE if requested
    if show_kde:
        # Ensemble KDE
        kde_ens = gaussian_kde(ensemble_values)
        x_range = np.linspace(ensemble_values.min(), ensemble_values.max(), 200)
        ax.plot(x_range, kde_ens(x_range),
               color=COLORS['ensemble_median'],
               linewidth=STYLE['ensemble_linewidth'],
               label='Ensemble KDE')

        # Observed KDE
        if observed is not None:
            obs_values = observed.dropna().values
            kde_obs = gaussian_kde(obs_values)
            x_range_obs = np.linspace(obs_values.min(), obs_values.max(), 200)
            ax.plot(x_range_obs, kde_obs(x_range_obs),
                   color=COLORS['observed'],
                   linewidth=STYLE['observed_linewidth'],
                   label='Observed KDE')

    # Set labels
    if xlabel is None:
        xlabel = LABELS['flow_units'].get(units, f'Flow ({units})')
    if ylabel is None:
        ylabel = 'Density' if density else 'Count'
    if title is None:
        title = f'Flow Distribution - {site_name}'

    # Apply styling (but don't use log_scale for y-axis, apply to x-axis instead)
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)
    if log_scale:
        ax.set_xscale('log')

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_monthly_distributions(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    plot_type: str = 'box',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['wide_figsize'],
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = 'cms',
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of flows for each month.

    Shows monthly boxplots, violin plots, or strip plots comparing
    ensemble and observed distributions.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    plot_type : str, default 'box'
        Type of distribution plot: 'box', 'violin', or 'strip'
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    ylabel : str, optional
        Y-axis label. If None, uses units
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_scale : bool, default False
        Use logarithmic y-axis
    units : str, default 'cms'
        Flow units for y-axis label
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_monthly_distributions(ensemble, observed=Q_obs)
    >>> plot_monthly_distributions(ensemble, plot_type='violin')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    if plot_type not in ['box', 'violin', 'strip']:
        raise ValueError(f"plot_type must be 'box', 'violin', or 'strip', got '{plot_type}'")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample to monthly
    site_data_monthly = resample_data(site_data, 'monthly')

    # Prepare data by month
    ensemble_by_month = []
    for month in range(1, 13):
        month_mask = site_data_monthly.index.month == month
        month_data = site_data_monthly[month_mask].values.flatten()
        month_data = month_data[~np.isnan(month_data)]
        ensemble_by_month.append(month_data)

    # Prepare observed data by month if provided
    obs_by_month = None
    if observed is not None:
        obs_monthly = resample_data(observed.to_frame(), 'monthly').iloc[:, 0]
        obs_by_month = []
        for month in range(1, 13):
            month_mask = obs_monthly.index.month == month
            month_data = obs_monthly[month_mask].values
            month_data = month_data[~np.isnan(month_data)]
            obs_by_month.append(month_data)

    # Create positions for boxplots
    positions_ens = np.arange(1, 13) - 0.2
    positions_obs = np.arange(1, 13) + 0.2

    if plot_type == 'box':
        # Ensemble boxplots
        bp_ens = ax.boxplot(ensemble_by_month, positions=positions_ens,
                           widths=0.35, patch_artist=True,
                           boxprops=dict(facecolor=COLORS['ensemble_fill'], alpha=0.7),
                           medianprops=dict(color=COLORS['ensemble_median'], linewidth=2),
                           whiskerprops=dict(color=COLORS['ensemble_median']),
                           capprops=dict(color=COLORS['ensemble_median']),
                           **kwargs)

        # Observed boxplots
        if obs_by_month is not None:
            bp_obs = ax.boxplot(obs_by_month, positions=positions_obs,
                               widths=0.35, patch_artist=True,
                               boxprops=dict(facecolor=COLORS['observed'], alpha=0.5),
                               medianprops=dict(color='black', linewidth=2),
                               whiskerprops=dict(color=COLORS['observed']),
                               capprops=dict(color=COLORS['observed']),
                               **kwargs)

    elif plot_type == 'violin':
        # Violin plots for ensemble
        parts_ens = ax.violinplot(ensemble_by_month, positions=positions_ens,
                                 widths=0.35, showmeans=False, showmedians=True)
        for pc in parts_ens['bodies']:
            pc.set_facecolor(COLORS['ensemble_fill'])
            pc.set_alpha(0.7)

        # Violin plots for observed
        if obs_by_month is not None:
            parts_obs = ax.violinplot(obs_by_month, positions=positions_obs,
                                     widths=0.35, showmeans=False, showmedians=True)
            for pc in parts_obs['bodies']:
                pc.set_facecolor(COLORS['observed'])
                pc.set_alpha(0.5)

    # Add legend manually
    if legend:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['ensemble_fill'], alpha=0.7, label='Ensemble'),
        ]
        if obs_by_month is not None:
            legend_elements.append(
                Patch(facecolor=COLORS['observed'], alpha=0.5, label='Observed')
            )
        ax.legend(handles=legend_elements, fontsize=LAYOUT['legend_fontsize'])

    # Set x-axis labels to month names
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(LABELS['month_labels'])

    # Set labels
    xlabel = 'Month'
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f'Monthly Flow Distributions - {site_name}'

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend=False, grid=grid, log_scale=log_scale)

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax
