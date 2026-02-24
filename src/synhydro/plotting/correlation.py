"""
Correlation plotting functions for SynHydro.

This module provides functions for plotting autocorrelation and spatial
correlation analyses.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Literal
from scipy.stats import pearsonr
from synhydro.core.ensemble import Ensemble
from synhydro.core.statistics import compute_autocorrelation, compute_spatial_correlation
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes, apply_default_styling, save_figure, get_site_data,
    resample_data, validate_ensemble_input, validate_observed_input
)


def plot_autocorrelation(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    max_lag: int = 30,
    timestep: str = 'daily',
    show_members: Optional[int] = None,
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
    Plot autocorrelation function with ensemble uncertainty.

    Shows temporal autocorrelation structure and how well ensemble preserves it
    compared to observed data.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    max_lag : int, default 30
        Maximum lag to compute
    timestep : str, default 'daily'
        Temporal resolution for lag calculation: 'daily', 'weekly', 'monthly'
    show_members : int, optional
        Number of individual member ACFs to show with transparency
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses lag with timestep
    ylabel : str, optional
        Y-axis label. If None, uses 'Autocorrelation'
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
    >>> plot_autocorrelation(ensemble, observed=Q_obs, max_lag=24)
    >>> plot_autocorrelation(ensemble, timestep='monthly', show_members=5)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample if needed
    if timestep == 'monthly':
        site_data = resample_data(site_data, 'monthly')
        if observed is not None:
            observed = resample_data(observed.to_frame(), 'monthly').iloc[:, 0]
        time_label = 'months'
    elif timestep == 'weekly':
        site_data = resample_data(site_data, 'weekly')
        if observed is not None:
            observed = resample_data(observed.to_frame(), 'weekly').iloc[:, 0]
        time_label = 'weeks'
    else:  # daily
        time_label = 'days'

    # Calculate autocorrelations for ensemble
    lag_range = range(1, max_lag + 1)
    n_realizations = len(site_data.columns)
    autocorr_ensemble = np.zeros((n_realizations, len(lag_range)))

    for i, realization in enumerate(site_data.columns):
        series = site_data[realization].dropna()
        for j, lag in enumerate(lag_range):
            if len(series) > lag:
                autocorr_ensemble[i, j] = pearsonr(
                    series.values[:-lag],
                    series.values[lag:]
                )[0]
            else:
                autocorr_ensemble[i, j] = np.nan

    # Plot individual members if requested
    if show_members is not None and show_members > 0:
        n_show = min(show_members, n_realizations)
        member_indices = np.random.choice(n_realizations, n_show, replace=False)
        for i in member_indices:
            ax.plot(lag_range, autocorr_ensemble[i, :],
                   color=COLORS['ensemble_members'],
                   alpha=STYLE['member_alpha'],
                   linewidth=STYLE['member_linewidth'],
                   **kwargs)

    # Plot ensemble range and median
    ax.fill_between(lag_range,
                   np.nanmin(autocorr_ensemble, axis=0),
                   np.nanmax(autocorr_ensemble, axis=0),
                   color=COLORS['ensemble_fill'],
                   alpha=STYLE['fill_alpha'],
                   label='Ensemble Range',
                   **kwargs)

    ax.plot(lag_range, np.nanmedian(autocorr_ensemble, axis=0),
           color=COLORS['ensemble_median'],
           linewidth=STYLE['ensemble_linewidth'],
           label='Ensemble Median',
           **kwargs)

    # Plot observed autocorrelation if provided
    if observed is not None:
        obs_series = observed.dropna()
        autocorr_obs = np.zeros(len(lag_range))
        for j, lag in enumerate(lag_range):
            if len(obs_series) > lag:
                autocorr_obs[j] = pearsonr(
                    obs_series.values[:-lag],
                    obs_series.values[lag:]
                )[0]
            else:
                autocorr_obs[j] = np.nan

        ax.plot(lag_range, autocorr_obs,
               color=COLORS['observed'],
               linewidth=STYLE['observed_linewidth'],
               marker=STYLE['observed_marker'],
               markersize=STYLE['observed_markersize'],
               label='Observed',
               **kwargs)

    # Set labels
    if xlabel is None:
        xlabel = f'Lag ({time_label})'
    if ylabel is None:
        ylabel = 'Autocorrelation (Pearson)'
    if title is None:
        title = f'Autocorrelation - {site_name}'

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_spatial_correlation(
    ensemble: Ensemble,
    observed: Optional[pd.DataFrame] = None,
    realization: int = 0,
    timestep: str = 'daily',
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    show_difference: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT['square_figsize'],
    title: Optional[str] = None,
    cmap: str = 'RdBu_r',
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial correlation heatmap across sites.

    Shows correlation matrix between multiple sites for a single ensemble
    realization and optionally compares with observed data.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.DataFrame, optional
        Multi-site observed data (sites as columns, time as index)
    realization : int, default 0
        Which ensemble realization to plot
    timestep : str, default 'daily'
        Temporal resolution: 'daily', 'weekly', 'monthly'
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        Correlation method
    show_difference : bool, default False
        Show difference heatmap (synthetic - observed) instead of side-by-side
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    cmap : str, default 'RdBu_r'
        Colormap for heatmap
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Additional arguments passed to seaborn heatmap

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_spatial_correlation(ensemble, observed=Q_obs_multisite)
    >>> plot_spatial_correlation(ensemble, show_difference=True)

    Notes
    -----
    This function requires multi-site ensemble data. Single-site ensembles
    cannot produce meaningful spatial correlation plots.
    """
    # Validate inputs
    validate_ensemble_input(ensemble)

    if len(ensemble.site_names) < 2:
        raise ValueError("Spatial correlation requires multi-site ensemble. "
                        f"Ensemble has only {len(ensemble.site_names)} site(s).")

    if realization not in ensemble.realization_ids:
        raise ValueError(f"Realization {realization} not found in ensemble. "
                        f"Available: {ensemble.realization_ids}")

    # Get realization data (all sites)
    real_data = ensemble.data_by_realization[realization]

    # Resample if needed
    if timestep != 'daily':
        real_data = resample_data(real_data, timestep)
        if observed is not None:
            observed = resample_data(observed, timestep)

    # Compute correlation matrix for ensemble
    ens_corr = compute_spatial_correlation(real_data, method=method)

    if show_difference and observed is not None:
        # Show difference plot
        fig, ax = setup_axes(ax, figsize)

        # Compute observed correlation
        obs_corr = compute_spatial_correlation(observed, method=method)

        # Calculate difference
        diff_corr = ens_corr - obs_corr

        # Plot difference heatmap
        sns.heatmap(diff_corr, ax=ax, cmap='RdBu_r', center=0,
                   square=True, annot=False,
                   vmin=-1, vmax=1,
                   cbar_kws={'label': 'Correlation Difference'},
                   **kwargs)

        ax.set_xlabel('Site', fontsize=LAYOUT['label_fontsize'])
        ax.set_ylabel('Site', fontsize=LAYOUT['label_fontsize'])

        if title is None:
            title = f'Spatial Correlation Difference\n(Synthetic - Observed)'
        ax.set_title(title, fontsize=LAYOUT['title_fontsize'])

    elif observed is not None:
        # Side-by-side comparison
        if ax is not None:
            raise ValueError("Cannot use provided ax for side-by-side comparison. "
                           "Set ax=None to create new figure.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Observed correlation heatmap
        obs_corr = compute_spatial_correlation(observed, method=method)
        sns.heatmap(obs_corr, ax=ax1, cmap=cmap,
                   square=True, annot=False,
                   vmin=0, vmax=1, cbar=False,
                   **kwargs)
        ax1.set_title('Observed', fontsize=LAYOUT['title_fontsize'])
        ax1.set_xlabel('Site', fontsize=LAYOUT['label_fontsize'])
        ax1.set_ylabel('Site', fontsize=LAYOUT['label_fontsize'])

        # Ensemble correlation heatmap
        sns.heatmap(ens_corr, ax=ax2, cmap=cmap,
                   square=True, annot=False,
                   vmin=0, vmax=1,
                   cbar_kws={'label': f'{method.capitalize()} Correlation'},
                   **kwargs)
        ax2.set_title(f'Synthetic (Realization {realization})',
                     fontsize=LAYOUT['title_fontsize'])
        ax2.set_xlabel('Site', fontsize=LAYOUT['label_fontsize'])
        ax2.set_ylabel('Site', fontsize=LAYOUT['label_fontsize'])

        if title is None:
            title = f'Spatial Correlation - {timestep.capitalize()}'
        fig.suptitle(title, fontsize=LAYOUT['title_fontsize'] + 2)

        ax = (ax1, ax2)  # Return both axes

    else:
        # Single heatmap for ensemble only
        fig, ax = setup_axes(ax, figsize)

        sns.heatmap(ens_corr, ax=ax, cmap=cmap,
                   square=True, annot=False,
                   vmin=0, vmax=1,
                   cbar_kws={'label': f'{method.capitalize()} Correlation'},
                   **kwargs)

        ax.set_xlabel('Site', fontsize=LAYOUT['label_fontsize'])
        ax.set_ylabel('Site', fontsize=LAYOUT['label_fontsize'])

        if title is None:
            title = f'Spatial Correlation - Realization {realization}'
        ax.set_title(title, fontsize=LAYOUT['title_fontsize'])

    # Tight layout
    if LAYOUT['tight_layout']:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax
