"""
Timeseries plotting functions for SynHydro.

This module provides functions for plotting ensemble timeseries data with
uncertainty bands, flow ranges, and seasonal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from synhydro.core.ensemble import Ensemble
from .config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes,
    apply_default_styling,
    save_figure,
    get_site_data,
    get_ylabel,
    subset_date_range,
    compute_ensemble_percentiles,
    format_date_axis,
    resample_data,
    get_temporal_grouper,
    validate_ensemble_input,
    validate_observed_input,
    validate_timestep,
    warn_if_many_realizations,
    warn_if_few_realizations,
)


def plot_timeseries(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    percentiles: Optional[List[float]] = None,
    show_members: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ensemble timeseries with uncertainty bands.

    Displays ensemble percentiles as shaded regions, optionally overlays
    individual ensemble members and observed data.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    percentiles : List[float], optional
        Percentiles to display as uncertainty bands. Default `[10, 50, 90]`.
    show_members : int, optional
        Number of individual ensemble members to show with transparency
    start_date : str, optional
        Start date for plot (ISO format or pandas-compatible)
    end_date : str, optional
        End date for plot
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates from metadata
    xlabel : str, optional
        X-axis label. If None, uses 'Date'
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
        Forwarded to `ax.plot` and `ax.fill_between`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> fig, ax = plot_timeseries(ensemble, observed=Q_obs, site='site_A')

    >>> # Plot on existing axes
    >>> fig, axes = plt.subplots(2, 1)
    >>> plot_timeseries(ensemble, ax=axes[0], percentiles=[25, 50, 75])

    >>> # Show individual members
    >>> plot_timeseries(ensemble, show_members=10, filename='timeseries.png')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    if percentiles is None:
        percentiles = [10, 50, 90]

    # Warn for large/small ensembles when relevant
    n_realizations = len(ensemble.realization_ids)
    if show_members is not None:
        warn_if_many_realizations(n_realizations, context="show_members")
    if percentiles is not None and len(percentiles) > 0:
        warn_if_few_realizations(n_realizations, context="percentile bands")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Subset by date range
    site_data = subset_date_range(site_data, start_date, end_date)
    if observed is not None:
        observed = subset_date_range(observed.to_frame(), start_date, end_date).iloc[
            :, 0
        ]

    # Plot individual ensemble members if requested
    if show_members is not None and show_members > 0:
        n_members = min(show_members, len(site_data.columns))
        member_cols = np.random.choice(site_data.columns, n_members, replace=False)
        for i, col in enumerate(member_cols):
            label = "Ensemble Members" if i == 0 else None
            ax.plot(
                site_data.index,
                site_data[col],
                color=COLORS["ensemble_members"],
                alpha=STYLE["member_alpha"],
                linewidth=STYLE["member_linewidth"],
                label=label,
                **kwargs,
            )

    # Plot percentiles
    if percentiles is not None and len(percentiles) > 0:
        perc_data = compute_ensemble_percentiles(site_data, percentiles)

        # Sort percentiles for proper fill_between ordering
        sorted_percs = sorted(percentiles)

        # Plot fill between outer percentiles
        if len(sorted_percs) >= 2:
            ax.fill_between(
                perc_data.index,
                perc_data[f"p{sorted_percs[0]}"],
                perc_data[f"p{sorted_percs[-1]}"],
                color=COLORS["ensemble_fill"],
                alpha=STYLE["fill_alpha"],
                label=f"Ensemble {sorted_percs[0]}-{sorted_percs[-1]}th %ile",
                **kwargs,
            )

        # Plot median or middle percentile
        if 50 in percentiles:
            ax.plot(
                perc_data.index,
                perc_data["p50"],
                color=COLORS["ensemble_median"],
                linewidth=STYLE["ensemble_linewidth"],
                label="Ensemble Median",
                **kwargs,
            )
        elif len(sorted_percs) > 0:
            mid_idx = len(sorted_percs) // 2
            mid_perc = sorted_percs[mid_idx]
            ax.plot(
                perc_data.index,
                perc_data[f"p{mid_perc}"],
                color=COLORS["ensemble_median"],
                linewidth=STYLE["ensemble_linewidth"],
                label=f"Ensemble {mid_perc}th %ile",
                **kwargs,
            )

    # Plot observed data
    if observed is not None:
        # Align observed data to site_data index
        observed_aligned = observed.reindex(site_data.index)
        ax.plot(
            observed_aligned.index,
            observed_aligned.values,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            label="Observed",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = "Date"
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f"Ensemble Timeseries - {site_name}"

    # Format date axis
    format_date_axis(ax, site_data)

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale)

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_seasonal_cycle(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    statistic: str = "mean",
    timestep: str = "monthly",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot seasonal-cycle line of a per-period statistic with ensemble band.

    Computes the chosen statistic (mean or std) of flow within each period
    (day-of-year, week, or month) for each realization, then plots the
    ensemble central tendency (median across realizations) and a 10th-90th
    percentile band over realizations. Observed data, when provided, is
    plotted as a single line at the same period statistic.

    This shows the spread of the per-realization period statistic across the
    ensemble, which is more meaningful for ensemble validation than the raw
    min/max envelope shown by `plot_flow_ranges`.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data.
    observed : pd.Series, optional
        Observed timeseries for comparison.
    site : str, optional
        Site name to plot. If None, uses first site in ensemble.
    statistic : {'mean', 'std'}, default 'mean'
        Per-period statistic to summarize.
    timestep : str, default 'monthly'
        Period for the seasonal grouper: 'daily' (day-of-year),
        'weekly' (week-of-year), or 'monthly' (month-of-year).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple, default from config
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generates.
    xlabel : str, optional
        X-axis label. If None, uses the timestep label.
    ylabel : str, optional
        Y-axis label. If None, uses units with statistic prefix.
    legend : bool, default True
        Whether to display legend.
    grid : bool, default True
        Whether to display grid.
    log_scale : bool, default False
        Use logarithmic y-axis.
    units : str, default 'cms'
        Flow units for y-axis label.
    filename : str, optional
        Path to save figure.
    dpi : int, default from config
        Resolution for saved figure.
    **kwargs
        Forwarded to `ax.plot` and `ax.fill_between`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If `statistic` is not 'mean' or 'std', or if `timestep` is not
        one of 'daily', 'weekly', 'monthly'.

    Examples
    --------
    >>> fig, ax = plot_seasonal_cycle(ensemble, observed=Q_obs)
    >>> plot_seasonal_cycle(ensemble, statistic='std', timestep='weekly')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    if statistic not in ("mean", "std"):
        raise ValueError(f"statistic must be 'mean' or 'std', got '{statistic}'")
    if timestep not in ("daily", "weekly", "monthly"):
        raise ValueError(
            "timestep for seasonal cycle must be one of "
            f"'daily', 'weekly', 'monthly', got '{timestep}'"
        )
    validate_timestep(ensemble, timestep)
    observed = validate_observed_input(observed, required=False)

    n_realizations = len(ensemble.realization_ids)
    warn_if_many_realizations(n_realizations, context="seasonal cycle")
    warn_if_few_realizations(n_realizations, context="percentile bands")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Compute per-realization period statistic
    grouper = get_temporal_grouper(site_data, timestep)
    if statistic == "mean":
        per_real = site_data.groupby(grouper).mean()
    else:
        per_real = site_data.groupby(grouper).std()

    period_index = per_real.index
    realization_stat = per_real.values  # (n_periods, n_realizations)

    # Ensemble percentile band and central tendency across realizations
    ens_p10 = np.nanpercentile(realization_stat, 10, axis=1)
    ens_p90 = np.nanpercentile(realization_stat, 90, axis=1)
    ens_median = np.nanmedian(realization_stat, axis=1)

    ax.fill_between(
        period_index,
        ens_p10,
        ens_p90,
        color=COLORS["ensemble_fill"],
        alpha=STYLE["fill_alpha"],
        label="Ensemble 10-90th %ile",
        **kwargs,
    )
    ax.plot(
        period_index,
        ens_median,
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label="Ensemble Median",
        **kwargs,
    )

    # Plot observed period statistic
    if observed is not None:
        obs_grouper = get_temporal_grouper(observed.to_frame(), timestep)
        if statistic == "mean":
            obs_stat = observed.groupby(obs_grouper).mean()
        else:
            obs_stat = observed.groupby(obs_grouper).std()

        ax.plot(
            obs_stat.index,
            obs_stat.values,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            marker=STYLE["observed_marker"],
            markersize=STYLE["observed_markersize"],
            label="Observed",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = LABELS["timestep_labels"].get(timestep, timestep.capitalize())
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
        ylabel = f"{statistic.capitalize()} {ylabel}"
    if title is None:
        title = f"Seasonal Cycle ({statistic}) - {site_name}"

    # Use month-name xticks for monthly timestep (cleaner display)
    if timestep == "monthly":
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(LABELS["month_labels"])

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale)

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_flow_ranges(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    timestep: str = "daily",
    aggregate: str = "median",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot flow ranges across temporal aggregation periods.

    Shows min/max ranges and central tendency (median or mean) for each
    period (day of year, week, month, etc.).

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    timestep : str, default 'daily'
        Temporal aggregation: 'daily', 'weekly', 'monthly', 'annual'
    aggregate : str, default 'median'
        Central tendency measure: 'median' or 'mean'
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses timestep label
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
        Forwarded to `ax.plot` and `ax.fill_between`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_flow_ranges(ensemble, observed=Q_obs, timestep='monthly')
    >>> plot_flow_ranges(ensemble, timestep='weekly', log_scale=True)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    validate_timestep(ensemble, timestep)
    observed = validate_observed_input(observed, required=False)

    if aggregate not in ["median", "mean"]:
        raise ValueError(f"aggregate must be 'median' or 'mean', got '{aggregate}'")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample if needed
    if timestep != "daily":
        site_data = resample_data(site_data, timestep)
        if observed is not None:
            observed = resample_data(observed.to_frame(), timestep).iloc[:, 0]

    # Get grouper
    grouper = get_temporal_grouper(site_data, timestep)

    # Compute statistics for ensemble
    s_max = site_data.groupby(grouper).max().max(axis=1)
    s_min = site_data.groupby(grouper).min().min(axis=1)
    if aggregate == "median":
        s_central = site_data.groupby(grouper).median().median(axis=1)
    else:
        s_central = site_data.groupby(grouper).mean().mean(axis=1)

    # Plot ensemble ranges
    xs = s_max.index
    ax.fill_between(
        xs,
        s_min,
        s_max,
        color=COLORS["ensemble_fill"],
        alpha=STYLE["fill_alpha"],
        label="Ensemble Range",
        **kwargs,
    )
    ax.plot(
        xs,
        s_central,
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label=f"Ensemble {aggregate.capitalize()}",
        **kwargs,
    )

    # Plot observed if provided
    if observed is not None:
        obs_grouper = get_temporal_grouper(observed.to_frame(), timestep)
        h_max = observed.groupby(obs_grouper).max()
        h_min = observed.groupby(obs_grouper).min()
        if aggregate == "median":
            h_central = observed.groupby(obs_grouper).median()
        else:
            h_central = observed.groupby(obs_grouper).mean()

        ax.fill_between(
            h_max.index,
            h_min,
            h_max,
            color=COLORS["observed"],
            alpha=STYLE["fill_alpha"],
            label="Observed Range",
            **kwargs,
        )
        ax.plot(
            h_central.index,
            h_central,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            label=f"Observed {aggregate.capitalize()}",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = LABELS["timestep_labels"].get(timestep, timestep.capitalize())
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f"{timestep.capitalize()} Flow Ranges - {site_name}"

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale)

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax
