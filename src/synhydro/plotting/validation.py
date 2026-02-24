"""
Validation plotting functions for SynHydro.

This module provides functions for comprehensive statistical validation
of synthetic ensembles against observed data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy import stats
from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    save_figure, get_site_data, resample_data,
    validate_ensemble_input, validate_observed_input
)


def _set_box_color(bp, color):
    """Helper to set boxplot colors."""
    plt.setp(bp['boxes'], color=color, facecolor=color)
    plt.setp(bp['whiskers'], color=color, linestyle='solid')
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='k', linewidth=2)


def plot_validation_panel(
    ensemble: Ensemble,
    observed: pd.Series,
    site: Optional[str] = None,
    timestep: str = 'monthly',
    log_space: bool = False,
    figsize: Tuple[float, float] = LAYOUT['validation_figsize'],
    filename: Optional[str] = None,
    dpi: int = LAYOUT['save_dpi'],
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
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
    observed : pd.Series
        Observed timeseries for validation (required)
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
        Additional arguments (for future expansion)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of 5 axes objects

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
    observed = validate_observed_input(observed, required=True)

    if timestep not in ['monthly', 'weekly']:
        raise ValueError(f"timestep must be 'monthly' or 'weekly', got '{timestep}'")

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample to monthly/weekly
    if timestep == 'monthly':
        site_data_agg = resample_data(site_data, 'monthly')
        obs_agg = resample_data(observed.to_frame(), 'monthly').iloc[:, 0]
        n_periods = 12
        period_labels = LABELS['month_labels']
    else:  # weekly
        site_data_agg = resample_data(site_data, 'weekly')
        obs_agg = resample_data(observed.to_frame(), 'weekly').iloc[:, 0]
        n_periods = 52
        period_labels = list(range(1, 53))

    # Prepare data reorganization for monthly case
    # For ensemble: shape (n_realizations, n_years, n_periods)
    # For observed: resample to match
    if timestep == 'monthly':
        # Pivot observed data
        H = obs_agg.to_frame()
        H = H[H.index.month.isin(range(1, 13))]  # Filter complete months
        H_pivot_df = H.pivot_table(
            index=H.index.year,
            columns=H.index.month,
            values=H.columns[0]
        )
        # Ensure all 12 months are present (fill missing months with NaN)
        H_pivot_df = H_pivot_df.reindex(columns=range(1, 13))
        H_pivot = H_pivot_df.values

        # Reorganize ensemble data
        S_list = []
        for real_id in site_data_agg.columns:
            real_series = site_data_agg[real_id]
            real_pivot_df = real_series.to_frame().pivot_table(
                index=real_series.index.year,
                columns=real_series.index.month,
                values=real_series.name
            )
            # Ensure all 12 months are present (fill missing months with NaN)
            real_pivot_df = real_pivot_df.reindex(columns=range(1, 13))
            S_list.append(real_pivot_df.values)
        S = np.stack(S_list, axis=0)  # Shape: (n_realizations, n_years, 12)

    else:  # weekly - simpler grouping
        H_pivot = obs_agg.groupby([obs_agg.index.year, obs_agg.index.isocalendar().week]).mean().unstack().values
        S_list = []
        for real_id in site_data_agg.columns:
            real_series = site_data_agg[real_id]
            real_pivot = real_series.groupby([real_series.index.year,
                                              real_series.index.isocalendar().week]).mean().unstack().values
            S_list.append(real_pivot)
        S = np.stack(S_list, axis=0)

    # Apply log transform if requested
    if log_space:
        H_proc = np.log(np.clip(H_pivot, a_min=1e-6, a_max=None))
        S_proc = np.log(np.clip(S, a_min=1e-6, a_max=None))
    else:
        H_proc = H_pivot
        S_proc = S

    # Resample historical data to match number of synthetic realizations
    n_realizations_S = S_proc.shape[0]
    n_years_H = H_proc.shape[0]
    idx = np.random.choice(n_years_H, size=(n_realizations_S, n_years_H), replace=True)
    H_resamp = H_proc[idx]

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=figsize, dpi=LAYOUT['default_dpi'])

    # ========================================================================
    # Panel 1: Overall distributions (boxplots)
    # ========================================================================
    ax = axes[0]

    # Reshape synthetic data for boxplot (combine realizations and years)
    S_flat = S_proc.reshape((S_proc.shape[0] * S_proc.shape[1], n_periods))

    # Create boxplots
    positions_syn = np.arange(1, n_periods + 1) - 0.15
    positions_hist = np.arange(1, n_periods + 1) + 0.15

    # Plot synthetic data (should always have all months)
    bp_syn = ax.boxplot(S_flat, positions=positions_syn, widths=0.25,
                        sym='', patch_artist=True)
    _set_box_color(bp_syn, COLORS['ensemble_fill'])

    # Plot observed data month-by-month to handle NaN columns
    # Collect boxplot components for later color setting
    bp_hist_boxes = []
    bp_hist_whiskers = []
    bp_hist_caps = []
    bp_hist_medians = []

    for i in range(n_periods):
        month_data = H_proc[:, i]
        # Only plot if this month has valid data
        if not np.all(np.isnan(month_data)):
            valid_data = month_data[~np.isnan(month_data)]
            bp_month = ax.boxplot([valid_data], positions=[positions_hist[i]],
                                 widths=0.25, sym='', patch_artist=True)
            bp_hist_boxes.extend(bp_month['boxes'])
            bp_hist_whiskers.extend(bp_month['whiskers'])
            bp_hist_caps.extend(bp_month['caps'])
            bp_hist_medians.extend(bp_month['medians'])

    # Create a dictionary to match the structure expected by _set_box_color
    bp_hist = {
        'boxes': bp_hist_boxes,
        'whiskers': bp_hist_whiskers,
        'caps': bp_hist_caps,
        'medians': bp_hist_medians
    }
    _set_box_color(bp_hist, COLORS['observed'])

    # Legend
    ax.plot([], c=COLORS['ensemble_fill'], label='Ensemble', linewidth=5)
    ax.plot([], c=COLORS['observed'], label='Observed', linewidth=5)
    ax.legend(ncol=2, loc='upper right', fontsize=LAYOUT['legend_fontsize'])

    ax.set_ylabel('Log(Q)' if log_space else 'Q', fontsize=LAYOUT['label_fontsize'])
    ax.set_xticks([])
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Panel 2: Monthly means
    # ========================================================================
    ax = axes[1]

    # Plot synthetic means
    bp_syn = ax.boxplot(S_proc.mean(axis=1), positions=positions_syn, widths=0.25,
                        sym='', patch_artist=True)
    _set_box_color(bp_syn, COLORS['ensemble_fill'])

    # Plot observed means month-by-month to handle NaN columns
    H_means = H_resamp.mean(axis=1)  # Shape: (n_realizations, n_periods)
    bp_hist_boxes = []
    bp_hist_whiskers = []
    bp_hist_caps = []
    bp_hist_medians = []

    for i in range(n_periods):
        month_means = H_means[:, i]
        if not np.all(np.isnan(month_means)):
            valid_means = month_means[~np.isnan(month_means)]
            bp_month = ax.boxplot([valid_means], positions=[positions_hist[i]],
                                 widths=0.25, sym='', patch_artist=True)
            bp_hist_boxes.extend(bp_month['boxes'])
            bp_hist_whiskers.extend(bp_month['whiskers'])
            bp_hist_caps.extend(bp_month['caps'])
            bp_hist_medians.extend(bp_month['medians'])

    bp_hist = {
        'boxes': bp_hist_boxes,
        'whiskers': bp_hist_whiskers,
        'caps': bp_hist_caps,
        'medians': bp_hist_medians
    }
    _set_box_color(bp_hist, COLORS['observed'])

    ax.set_ylabel(r'$\hat{\mu}_Q$', fontsize=LAYOUT['label_fontsize'])
    ax.set_xticks([])
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Panel 3: Monthly standard deviations
    # ========================================================================
    ax = axes[2]

    # Plot synthetic std devs
    bp_syn = ax.boxplot(S_proc.std(axis=1), positions=positions_syn, widths=0.25,
                        sym='', patch_artist=True)
    _set_box_color(bp_syn, COLORS['ensemble_fill'])

    # Plot observed std devs month-by-month to handle NaN columns
    H_stds = H_resamp.std(axis=1)  # Shape: (n_realizations, n_periods)
    bp_hist_boxes = []
    bp_hist_whiskers = []
    bp_hist_caps = []
    bp_hist_medians = []

    for i in range(n_periods):
        month_stds = H_stds[:, i]
        if not np.all(np.isnan(month_stds)):
            valid_stds = month_stds[~np.isnan(month_stds)]
            bp_month = ax.boxplot([valid_stds], positions=[positions_hist[i]],
                                 widths=0.25, sym='', patch_artist=True)
            bp_hist_boxes.extend(bp_month['boxes'])
            bp_hist_whiskers.extend(bp_month['whiskers'])
            bp_hist_caps.extend(bp_month['caps'])
            bp_hist_medians.extend(bp_month['medians'])

    bp_hist = {
        'boxes': bp_hist_boxes,
        'whiskers': bp_hist_whiskers,
        'caps': bp_hist_caps,
        'medians': bp_hist_medians
    }
    _set_box_color(bp_hist, COLORS['observed'])

    ax.set_ylabel(r'$\hat{\sigma}_Q$', fontsize=LAYOUT['label_fontsize'])
    ax.set_xticks([])
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Panel 4: Wilcoxon rank-sum test p-values
    # ========================================================================
    ax = axes[3]

    stat_pvals = np.zeros(n_periods)
    for i in range(n_periods):
        try:
            # Remove NaN values before statistical test
            h_vals = H_proc[:, i]
            s_vals = S_flat[:, i]
            h_valid = h_vals[~np.isnan(h_vals)]
            s_valid = s_vals[~np.isnan(s_vals)]

            # Only compute test if both samples have data
            if len(h_valid) > 0 and len(s_valid) > 0:
                stat_pvals[i] = stats.ranksums(h_valid, s_valid)[1]
            else:
                stat_pvals[i] = np.nan
        except Exception:
            stat_pvals[i] = np.nan

    ax.bar(np.arange(1, n_periods + 1), stat_pvals,
          facecolor='0.7', edgecolor='none')
    ax.axhline(0.05, color='k', linewidth=1, linestyle='--')
    ax.set_xlim([0, n_periods + 1])
    ax.set_ylabel('Wilcoxon $p$', fontsize=LAYOUT['label_fontsize'])
    ax.set_ylim([0, 1.05])
    ax.set_xticks([])
    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========================================================================
    # Panel 5: Levene test p-values (variance equality)
    # ========================================================================
    ax = axes[4]

    levene_pvals = np.zeros(n_periods)
    for i in range(n_periods):
        try:
            # Remove NaN values before statistical test
            h_vals = H_proc[:, i]
            s_vals = S_flat[:, i]
            h_valid = h_vals[~np.isnan(h_vals)]
            s_valid = s_vals[~np.isnan(s_vals)]

            # Only compute test if both samples have data
            if len(h_valid) > 0 and len(s_valid) > 0:
                levene_pvals[i] = stats.levene(h_valid, s_valid)[1]
            else:
                levene_pvals[i] = np.nan
        except Exception:
            levene_pvals[i] = np.nan

    ax.bar(np.arange(1, n_periods + 1), levene_pvals,
          facecolor='0.7', edgecolor='none')
    ax.axhline(0.05, color='k', linewidth=1, linestyle='--')
    ax.set_xlim([0, n_periods + 1])
    ax.set_ylabel('Levene $p$', fontsize=LAYOUT['label_fontsize'])
    ax.set_ylim([0, 1.05])

    # Set x-ticks on bottom panel
    if timestep == 'monthly':
        ax.set_xticks(range(1, n_periods + 1))
        ax.set_xticklabels(period_labels, fontsize=LAYOUT['tick_fontsize'])
    else:
        ax.set_xticks(np.arange(0, n_periods + 1, 5))
        ax.set_xticklabels(np.arange(0, n_periods + 1, 5), fontsize=LAYOUT['tick_fontsize'])

    ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Overall title
    title_text = ('Log space' if log_space else 'Real space') + f' - {site_name}'
    fig.suptitle(f'Statistical Validation - {timestep.capitalize()}\n{title_text}',
                fontsize=LAYOUT['title_fontsize'] + 2)

    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, axes
