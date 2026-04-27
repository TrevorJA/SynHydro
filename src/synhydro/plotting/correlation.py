"""
Correlation plotting functions for SynHydro.

This module provides functions for plotting autocorrelation and spatial
correlation analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Literal, Union
from scipy.stats import pearsonr
from synhydro.core.ensemble import Ensemble
from synhydro.core.statistics import (
    compute_autocorrelation,
    compute_spatial_correlation,
)
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes,
    apply_default_styling,
    save_figure,
    get_site_data,
    resample_data,
    validate_ensemble_input,
    validate_observed_input,
    validate_timestep,
    warn_if_many_realizations,
)


def plot_autocorrelation(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    max_lag: int = 30,
    timestep: str = "daily",
    show_members: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
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
        Forwarded to `ax.plot` and `ax.fill_between`.

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
    validate_timestep(ensemble, timestep)
    observed = validate_observed_input(observed, required=False)

    # Warn for large ensembles: per-realization ACF iteration cost scales with N
    n_realizations = len(ensemble.realization_ids)
    warn_if_many_realizations(n_realizations, context="autocorrelation")
    if show_members is not None:
        warn_if_many_realizations(n_realizations, context="show_members")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample if needed
    if timestep == "monthly":
        site_data = resample_data(site_data, "monthly")
        if observed is not None:
            observed = resample_data(observed.to_frame(), "monthly").iloc[:, 0]
        time_label = "months"
    elif timestep == "weekly":
        site_data = resample_data(site_data, "weekly")
        if observed is not None:
            observed = resample_data(observed.to_frame(), "weekly").iloc[:, 0]
        time_label = "weeks"
    else:  # daily
        time_label = "days"

    # Calculate autocorrelations for ensemble
    lag_range = range(1, max_lag + 1)
    n_realizations = len(site_data.columns)
    autocorr_ensemble = np.zeros((n_realizations, len(lag_range)))

    for i, realization in enumerate(site_data.columns):
        series = site_data[realization].dropna()
        for j, lag in enumerate(lag_range):
            if len(series) > lag:
                autocorr_ensemble[i, j] = pearsonr(
                    series.values[:-lag], series.values[lag:]
                )[0]
            else:
                autocorr_ensemble[i, j] = np.nan

    # Plot individual members if requested
    if show_members is not None and show_members > 0:
        n_show = min(show_members, n_realizations)
        member_indices = np.random.choice(n_realizations, n_show, replace=False)
        for i in member_indices:
            ax.plot(
                lag_range,
                autocorr_ensemble[i, :],
                color=COLORS["ensemble_members"],
                alpha=STYLE["member_alpha"],
                linewidth=STYLE["member_linewidth"],
                **kwargs,
            )

    # Plot ensemble range and median
    ax.fill_between(
        lag_range,
        np.nanmin(autocorr_ensemble, axis=0),
        np.nanmax(autocorr_ensemble, axis=0),
        color=COLORS["ensemble_fill"],
        alpha=STYLE["fill_alpha"],
        label="Ensemble Range",
        **kwargs,
    )

    ax.plot(
        lag_range,
        np.nanmedian(autocorr_ensemble, axis=0),
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label="Ensemble Median",
        **kwargs,
    )

    # Plot observed autocorrelation if provided
    if observed is not None:
        obs_series = observed.dropna()
        autocorr_obs = np.zeros(len(lag_range))
        for j, lag in enumerate(lag_range):
            if len(obs_series) > lag:
                autocorr_obs[j] = pearsonr(
                    obs_series.values[:-lag], obs_series.values[lag:]
                )[0]
            else:
                autocorr_obs[j] = np.nan

        ax.plot(
            lag_range,
            autocorr_obs,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            marker=STYLE["observed_marker"],
            markersize=STYLE["observed_markersize"],
            label="Observed",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = f"Lag ({time_label})"
    if ylabel is None:
        ylabel = "Autocorrelation (Pearson)"
    if title is None:
        title = f"Autocorrelation - {site_name}"

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_spatial_correlation(
    ensemble: Ensemble,
    observed: Optional[Union[pd.Series, pd.DataFrame]] = None,
    realization: Optional[int] = None,
    timestep: str = "daily",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    show_difference: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["square_figsize"],
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot spatial correlation heatmap across sites.

    Shows correlation matrix between multiple sites for the ensemble
    (mean across realizations or a single realization) and optionally
    compares with observed data.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series or pd.DataFrame, optional
        Multi-site observed data (sites as columns, time as index). A
        single-column Series will be promoted to a DataFrame, but spatial
        correlation requires at least two sites.
    realization : int, optional
        Which ensemble realization to plot. If None (default), use the
        ensemble-mean correlation matrix (mean of per-realization
        correlations). If int, use that single realization.
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
        Forwarded to `seaborn.heatmap`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        List of length 1 (single heatmap or difference) or 2 (side-by-side).

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
    validate_timestep(ensemble, timestep)

    if len(ensemble.site_names) < 2:
        raise ValueError(
            "Spatial correlation requires multi-site ensemble. "
            f"Ensemble has only {len(ensemble.site_names)} site(s)."
        )

    # Coerce single-site Series to DataFrame for downstream uniformity, then
    # validate that it actually has multiple columns.
    if isinstance(observed, pd.Series):
        observed = observed.to_frame()
    if observed is not None and len(observed.columns) < 2:
        raise ValueError(
            "Spatial correlation requires multi-site observed data. "
            f"Got {len(observed.columns)} site(s)."
        )

    # Resample observed once (used in all branches below)
    if timestep != "daily" and observed is not None:
        observed = resample_data(observed, timestep)

    # Compute ensemble correlation matrix
    if realization is None:
        corr_matrices = []
        for rid in ensemble.realization_ids:
            df = ensemble.data_by_realization[rid]
            if timestep != "daily":
                df = resample_data(df, timestep)
            corr_matrices.append(compute_spatial_correlation(df, method=method))
        ens_corr = sum(corr_matrices) / len(corr_matrices)
        real_label = "Ensemble Mean"
    else:
        if realization not in ensemble.realization_ids:
            raise ValueError(
                f"Realization {realization} not found in ensemble. "
                f"Available: {ensemble.realization_ids}"
            )
        real_data = ensemble.data_by_realization[realization]
        if timestep != "daily":
            real_data = resample_data(real_data, timestep)
        ens_corr = compute_spatial_correlation(real_data, method=method)
        real_label = f"Realization {realization}"

    # Allow callers to override annotation behavior via kwargs without
    # colliding with the explicit values set below.
    annot = kwargs.pop("annot", False)
    fmt = kwargs.pop("fmt", ".2g")
    annot_kws = kwargs.pop("annot_kws", None)

    if show_difference and observed is not None:
        # Show difference plot
        fig, ax = setup_axes(ax, figsize)

        # Compute observed correlation
        obs_corr = compute_spatial_correlation(observed, method=method)

        # Calculate difference
        diff_corr = ens_corr - obs_corr

        # Plot difference heatmap
        sns.heatmap(
            diff_corr,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            square=True,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Correlation Difference"},
            **kwargs,
        )

        ax.set_xlabel("Site", fontsize=LAYOUT["label_fontsize"])
        ax.set_ylabel("Site", fontsize=LAYOUT["label_fontsize"])

        if title is None:
            title = f"Spatial Correlation Difference\n(Synthetic - Observed)"
        ax.set_title(title, fontsize=LAYOUT["title_fontsize"])

        axes_out = [ax]

    elif observed is not None:
        # Side-by-side comparison
        if ax is not None:
            raise ValueError(
                "Cannot use provided ax for side-by-side comparison. "
                "Set ax=None to create new figure."
            )

        # Compute observed correlation
        obs_corr = compute_spatial_correlation(observed, method=method)

        # Use consistent vmin/vmax from both matrices
        all_vals = np.concatenate([obs_corr.values.ravel(), ens_corr.values.ravel()])
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)

        # Use GridSpec with constrained_layout so the manual colorbar axis
        # composes cleanly. tight_layout cannot handle this layout and emits
        # a UserWarning if used.
        fig = plt.figure(
            figsize=(figsize[0] * 2.2, figsize[1]), constrained_layout=True
        )
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[0, 2])

        # Observed correlation heatmap
        sns.heatmap(
            obs_corr,
            ax=ax1,
            cmap=cmap,
            square=True,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            **kwargs,
        )
        ax1.set_title("Observed", fontsize=LAYOUT["title_fontsize"])
        ax1.set_xlabel("Site", fontsize=LAYOUT["label_fontsize"])
        ax1.set_ylabel("Site", fontsize=LAYOUT["label_fontsize"])

        # Ensemble correlation heatmap with shared colorbar
        sns.heatmap(
            ens_corr,
            ax=ax2,
            cmap=cmap,
            square=True,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_ax=cbar_ax,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
            **kwargs,
        )
        ax2.set_title(f"Synthetic ({real_label})", fontsize=LAYOUT["title_fontsize"])
        ax2.set_xlabel("Site", fontsize=LAYOUT["label_fontsize"])
        ax2.set_ylabel("Site", fontsize=LAYOUT["label_fontsize"])

        if title is None:
            title = f"Spatial Correlation - {timestep.capitalize()}"
        fig.suptitle(title, fontsize=LAYOUT["title_fontsize"] + 2)

        axes_out = [ax1, ax2]

    else:
        # Single heatmap for ensemble only
        fig, ax = setup_axes(ax, figsize)

        sns.heatmap(
            ens_corr,
            ax=ax,
            cmap=cmap,
            square=True,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmin=0,
            vmax=1,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
            **kwargs,
        )

        ax.set_xlabel("Site", fontsize=LAYOUT["label_fontsize"])
        ax.set_ylabel("Site", fontsize=LAYOUT["label_fontsize"])

        if title is None:
            title = f"Spatial Correlation - {real_label}"
        ax.set_title(title, fontsize=LAYOUT["title_fontsize"])

        axes_out = [ax]

    # Tight layout (skipped if a constrained layout is already in place,
    # e.g. the side-by-side branch which uses GridSpec + manual colorbar)
    if LAYOUT["tight_layout"] and fig.get_layout_engine() is None:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, axes_out
