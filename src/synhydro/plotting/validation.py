"""
Validation plotting functions for SynHydro.

This module provides functions for comprehensive statistical validation
of synthetic ensembles against observed data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from scipy import stats
from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    apply_default_styling,
    save_figure,
    get_site_data,
    resample_data,
    validate_ensemble_input,
    validate_observed_input,
    validate_timestep,
    warn_if_many_realizations,
)


def _set_box_color(bp, color):
    """Helper to set boxplot colors using STYLE-driven median styling."""
    plt.setp(bp["boxes"], color=color, facecolor=color)
    plt.setp(bp["whiskers"], color=color, linestyle="solid")
    plt.setp(bp["caps"], color=color)
    plt.setp(
        bp["medians"],
        color=STYLE["boxplot_median_color"],
        linewidth=STYLE["boxplot_median_linewidth"],
    )


def plot_validation_panel(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    timestep: str = "monthly",
    log_space: bool = False,
    figsize: Tuple[float, float] = LAYOUT["validation_figsize"],
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Multi-panel validation plot (EXCEPTION to single-panel rule).

    Creates 5-panel figure with comprehensive statistical validation:
    1. Overall distributions (boxplots by month)
    2. Monthly mean comparison
    3. Monthly std comparison
    4. Wilcoxon test p-values
    5. Levene test p-values

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for validation. Panels 4 and 5 (Wilcoxon and
        Levene tests) require observed data and will display a placeholder
        message when it is not provided.
    site : str, optional
        Site name to analyze. If None, uses first site in ensemble
    timestep : str, default 'monthly'
        Temporal aggregation for validation: 'monthly' or 'weekly'
    log_space : bool, default False
        Perform validation in log space
    figsize : tuple, default from config
        Figure size (width, height) in inches
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Reserved for future expansion; currently unused.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        List of 5 axes objects.

    Examples
    --------
    >>> fig, axes = plot_validation_panel(ensemble, observed=Q_obs)
    >>> fig, axes = plot_validation_panel(ensemble, Q_obs, log_space=True)

    Notes
    -----
    This is the only multi-panel function in the plotting module, preserved
    from the original codebase due to its comprehensive validation capabilities.
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    validate_timestep(ensemble, timestep)
    observed = validate_observed_input(observed, required=False)

    warn_if_many_realizations(len(ensemble.realization_ids), context="validation")

    if timestep not in ["monthly", "weekly"]:
        raise ValueError(f"timestep must be 'monthly' or 'weekly', got '{timestep}'")

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample to monthly/weekly
    if timestep == "monthly":
        site_data_agg = resample_data(site_data, "monthly")
        n_periods = 12
        period_labels = LABELS["month_labels"]
        if observed is not None:
            obs_agg = resample_data(observed.to_frame(), "monthly").iloc[:, 0]
    else:  # weekly
        site_data_agg = resample_data(site_data, "weekly")
        n_periods = 52
        period_labels = list(range(1, 53))
        if observed is not None:
            obs_agg = resample_data(observed.to_frame(), "weekly").iloc[:, 0]

    # Build observed pivot (H_pivot) only if observed provided
    H_pivot = None
    if observed is not None:
        if timestep == "monthly":
            H = obs_agg.to_frame()
            H_pivot_df = H.pivot_table(
                index=H.index.year, columns=H.index.month, values=H.columns[0]
            )
            H_pivot_df = H_pivot_df.reindex(columns=range(1, 13)).dropna()
            H_pivot = H_pivot_df.values
        else:
            H_pivot = (
                obs_agg.groupby([obs_agg.index.year, obs_agg.index.isocalendar().week])
                .mean()
                .unstack()
                .reindex(columns=range(1, 53))
                .values
            )

    # Build synthetic stack S, shape (n_realizations, n_complete_years, n_periods)
    if timestep == "monthly":
        S_list = []
        for real_id in site_data_agg.columns:
            real_series = site_data_agg[real_id]
            real_pivot_df = real_series.to_frame().pivot_table(
                index=real_series.index.year,
                columns=real_series.index.month,
                values=real_series.name,
            )
            real_pivot_df = real_pivot_df.reindex(columns=range(1, 13)).dropna()
            S_list.append(real_pivot_df.values)
        S = np.stack(S_list, axis=0)
    else:
        S_list = []
        for real_id in site_data_agg.columns:
            real_series = site_data_agg[real_id]
            real_pivot = (
                real_series.groupby(
                    [real_series.index.year, real_series.index.isocalendar().week]
                )
                .mean()
                .unstack()
                .reindex(columns=range(1, 53))
                .values
            )
            S_list.append(real_pivot)
        S = np.stack(S_list, axis=0)

    # Apply log transform if requested
    if log_space:
        S_proc = np.log(np.clip(S, a_min=1e-6, a_max=None))
        H_proc = (
            np.log(np.clip(H_pivot, a_min=1e-6, a_max=None))
            if H_pivot is not None
            else None
        )
    else:
        S_proc = S
        H_proc = H_pivot

    # Resample historical data to match number of synthetic realizations
    H_resamp = None
    if H_proc is not None:
        n_realizations_S = S_proc.shape[0]
        n_years_H = H_proc.shape[0]
        idx = np.random.choice(
            n_years_H, size=(n_realizations_S, n_years_H), replace=True
        )
        H_resamp = H_proc[idx]

    # Create figure with 5 subplots
    fig, axes_arr = plt.subplots(5, 1, figsize=figsize, dpi=LAYOUT["default_dpi"])
    axes = list(axes_arr)

    # Common positions
    positions_syn = np.arange(1, n_periods + 1) - 0.15
    positions_hist = np.arange(1, n_periods + 1) + 0.15

    # Reshape synthetic data for boxplot/test usage (combine realizations and years)
    S_flat = S_proc.reshape((S_proc.shape[0] * S_proc.shape[1], n_periods))

    def _draw_observed_per_period(ax, data_2d):
        """Draw boxplots column-by-column for an (n_rows, n_periods) array,
        coloring the result with the observed palette. Skips columns that are
        entirely NaN."""
        boxes, whiskers, caps, medians = [], [], [], []
        for i in range(n_periods):
            col = data_2d[:, i]
            if np.all(np.isnan(col)):
                continue
            valid = col[~np.isnan(col)]
            bp = ax.boxplot(
                [valid],
                positions=[positions_hist[i]],
                widths=0.25,
                sym="",
                patch_artist=True,
            )
            boxes.extend(bp["boxes"])
            whiskers.extend(bp["whiskers"])
            caps.extend(bp["caps"])
            medians.extend(bp["medians"])
        bp_collected = {
            "boxes": boxes,
            "whiskers": whiskers,
            "caps": caps,
            "medians": medians,
        }
        _set_box_color(bp_collected, COLORS["observed"])

    # ========================================================================
    # Panel 1: Overall distributions (boxplots)
    # ========================================================================
    ax = axes[0]
    bp_syn = ax.boxplot(
        S_flat, positions=positions_syn, widths=0.25, sym="", patch_artist=True
    )
    _set_box_color(bp_syn, COLORS["ensemble_fill"])

    if H_proc is not None:
        _draw_observed_per_period(ax, H_proc)
        ax.plot([], c=COLORS["ensemble_fill"], label="Ensemble", linewidth=5)
        ax.plot([], c=COLORS["observed"], label="Observed", linewidth=5)
        ax.legend(ncol=2, loc="upper right", fontsize=LAYOUT["legend_fontsize"])
    else:
        ax.plot([], c=COLORS["ensemble_fill"], label="Ensemble", linewidth=5)
        ax.legend(loc="upper right", fontsize=LAYOUT["legend_fontsize"])

    ax.set_ylabel("Log(Q)" if log_space else "Q", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(
        ax, legend=False, grid=True, log_scale=False, hide_xticks=True
    )

    # ========================================================================
    # Panel 2: Monthly means
    # ========================================================================
    ax = axes[1]
    bp_syn = ax.boxplot(
        S_proc.mean(axis=1),
        positions=positions_syn,
        widths=0.25,
        sym="",
        patch_artist=True,
    )
    _set_box_color(bp_syn, COLORS["ensemble_fill"])

    if H_resamp is not None:
        _draw_observed_per_period(ax, H_resamp.mean(axis=1))

    ax.set_ylabel(r"$\hat{\mu}_Q$", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(
        ax, legend=False, grid=True, log_scale=False, hide_xticks=True
    )

    # ========================================================================
    # Panel 3: Monthly standard deviations
    # ========================================================================
    ax = axes[2]
    bp_syn = ax.boxplot(
        S_proc.std(axis=1),
        positions=positions_syn,
        widths=0.25,
        sym="",
        patch_artist=True,
    )
    _set_box_color(bp_syn, COLORS["ensemble_fill"])

    if H_resamp is not None:
        _draw_observed_per_period(ax, H_resamp.std(axis=1))

    ax.set_ylabel(r"$\hat{\sigma}_Q$", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(
        ax, legend=False, grid=True, log_scale=False, hide_xticks=True
    )

    # ========================================================================
    # Panel 4: Wilcoxon rank-sum test p-values
    # ========================================================================
    ax = axes[3]
    if H_proc is not None:
        stat_pvals = np.zeros(n_periods)
        for i in range(n_periods):
            try:
                h_vals = H_proc[:, i]
                s_vals = S_flat[:, i]
                h_valid = h_vals[~np.isnan(h_vals)]
                s_valid = s_vals[~np.isnan(s_vals)]
                if len(h_valid) > 0 and len(s_valid) > 0:
                    stat_pvals[i] = stats.ranksums(h_valid, s_valid)[1]
                else:
                    stat_pvals[i] = np.nan
            except Exception:
                stat_pvals[i] = np.nan

        ax.bar(
            np.arange(1, n_periods + 1),
            stat_pvals,
            facecolor="0.7",
            edgecolor="none",
        )
        ax.axhline(0.05, color="k", linewidth=1, linestyle="--")
        ax.set_xlim([0, n_periods + 1])
        ax.set_ylim([0, 1.05])
    else:
        ax.text(
            0.5,
            0.5,
            "Observed data not provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_ylabel("Wilcoxon $p$", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(
        ax, legend=False, grid=True, log_scale=False, hide_xticks=True
    )

    # ========================================================================
    # Panel 5: Levene test p-values (variance equality)
    # ========================================================================
    ax = axes[4]
    if H_proc is not None:
        levene_pvals = np.zeros(n_periods)
        for i in range(n_periods):
            try:
                h_vals = H_proc[:, i]
                s_vals = S_flat[:, i]
                h_valid = h_vals[~np.isnan(h_vals)]
                s_valid = s_vals[~np.isnan(s_vals)]
                if len(h_valid) > 0 and len(s_valid) > 0:
                    levene_pvals[i] = stats.levene(h_valid, s_valid)[1]
                else:
                    levene_pvals[i] = np.nan
            except Exception:
                levene_pvals[i] = np.nan

        ax.bar(
            np.arange(1, n_periods + 1),
            levene_pvals,
            facecolor="0.7",
            edgecolor="none",
        )
        ax.axhline(0.05, color="k", linewidth=1, linestyle="--")
        ax.set_xlim([0, n_periods + 1])
        ax.set_ylim([0, 1.05])
    else:
        ax.text(
            0.5,
            0.5,
            "Observed data not provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_ylabel("Levene $p$", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(ax, legend=False, grid=True, log_scale=False)

    # Set x-ticks on bottom panel
    if timestep == "monthly":
        ax.set_xticks(range(1, n_periods + 1))
        ax.set_xticklabels(period_labels, fontsize=LAYOUT["tick_fontsize"])
    else:
        ax.set_xticks(np.arange(0, n_periods + 1, 5))
        ax.set_xticklabels(
            np.arange(0, n_periods + 1, 5), fontsize=LAYOUT["tick_fontsize"]
        )

    # Overall title
    title_text = ("Log space" if log_space else "Real space") + f" - {site_name}"
    fig.suptitle(
        f"Statistical Validation - {timestep.capitalize()}\n{title_text}",
        fontsize=LAYOUT["title_fontsize"] + 2,
    )

    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, axes
