"""
Smoke tests for synhydro.plotting timeseries functions.

Covers plot_timeseries, plot_flow_ranges, plot_seasonal_cycle.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from synhydro.plotting import (
    plot_flow_ranges,
    plot_seasonal_cycle,
    plot_timeseries,
)

logger = logging.getLogger(__name__)


def _has_artists(ax: plt.Axes) -> bool:
    """Return True if the axes has any drawn lines or collections."""
    return (len(ax.lines) + len(ax.collections)) > 0


def _member_ydata(ax: plt.Axes, n_members: int) -> list:
    """Return y-data of the first ``n_members`` lines (the member lines)."""
    return [np.asarray(line.get_ydata()) for line in ax.lines[:n_members]]


# ----------------------------------------------------------------------
# plot_timeseries
# ----------------------------------------------------------------------


def test_plot_timeseries_default(small_ensemble):
    fig, ax = plot_timeseries(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_timeseries_with_observed(small_ensemble, observed_series):
    fig, ax = plot_timeseries(small_ensemble, observed=observed_series)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_timeseries_log_scale(small_ensemble):
    fig, ax = plot_timeseries(small_ensemble, log_scale=True)
    assert ax.get_yscale() == "log"
    assert _has_artists(ax)


def test_plot_timeseries_show_members(small_ensemble):
    fig, ax = plot_timeseries(small_ensemble, show_members=2)
    assert isinstance(fig, plt.Figure)
    assert _has_artists(ax)


def test_plot_timeseries_seed_reproduces_member_selection(small_ensemble):
    """The same seed (int or Generator) draws the same ensemble members."""
    _, ax_a = plot_timeseries(small_ensemble, show_members=2, seed=3)
    _, ax_b = plot_timeseries(small_ensemble, show_members=2, seed=3)
    _, ax_c = plot_timeseries(
        small_ensemble, show_members=2, seed=np.random.default_rng(3)
    )
    ydata_a = _member_ydata(ax_a, 2)
    assert len(ydata_a) == 2
    for y_a, y_b, y_c in zip(ydata_a, _member_ydata(ax_b, 2), _member_ydata(ax_c, 2)):
        np.testing.assert_array_equal(y_a, y_b)
        np.testing.assert_array_equal(y_a, y_c)

    # The drawn lines are real realizations of the plotted site
    site_df = small_ensemble.data_by_site[small_ensemble.site_names[0]]
    for y in ydata_a:
        assert any(np.array_equal(y, site_df[col].values) for col in site_df.columns)


def test_plot_timeseries_seed_none_still_draws_members(small_ensemble):
    fig, ax = plot_timeseries(small_ensemble, show_members=2, seed=None)
    # Two member lines plus the ensemble median
    assert len(ax.lines) >= 3
    assert _has_artists(ax)


def test_plot_timeseries_site_selection(small_ensemble):
    fig, ax = plot_timeseries(small_ensemble, site="site_B")
    assert "site_B" in ax.get_title()


# ----------------------------------------------------------------------
# plot_flow_ranges
# ----------------------------------------------------------------------


def test_plot_flow_ranges_default(small_ensemble):
    fig, ax = plot_flow_ranges(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_flow_ranges_with_observed(small_ensemble, observed_series):
    fig, ax = plot_flow_ranges(small_ensemble, observed=observed_series)
    assert isinstance(fig, plt.Figure)
    assert _has_artists(ax)


def test_plot_flow_ranges_monthly_timestep(small_ensemble):
    fig, ax = plot_flow_ranges(small_ensemble, timestep="monthly")
    assert _has_artists(ax)


def test_plot_flow_ranges_invalid_aggregate(small_ensemble):
    with pytest.raises(ValueError, match="aggregate"):
        plot_flow_ranges(small_ensemble, aggregate="bogus")


# ----------------------------------------------------------------------
# plot_seasonal_cycle
# ----------------------------------------------------------------------


def test_plot_seasonal_cycle_default(small_ensemble):
    fig, ax = plot_seasonal_cycle(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_seasonal_cycle_with_observed(small_ensemble, observed_series):
    fig, ax = plot_seasonal_cycle(small_ensemble, observed=observed_series)
    assert _has_artists(ax)


def test_plot_seasonal_cycle_std_statistic(small_ensemble):
    fig, ax = plot_seasonal_cycle(small_ensemble, statistic="std")
    assert _has_artists(ax)


def test_plot_seasonal_cycle_weekly(small_ensemble):
    fig, ax = plot_seasonal_cycle(small_ensemble, timestep="weekly")
    assert _has_artists(ax)


def test_plot_seasonal_cycle_invalid_statistic(small_ensemble):
    with pytest.raises(ValueError, match="statistic"):
        plot_seasonal_cycle(small_ensemble, statistic="median")


def test_plot_seasonal_cycle_invalid_timestep(small_ensemble):
    with pytest.raises(ValueError, match="timestep"):
        plot_seasonal_cycle(small_ensemble, timestep="annual")
