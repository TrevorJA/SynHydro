"""
Smoke tests for synhydro.plotting timeseries functions.

Covers plot_timeseries, plot_flow_ranges, plot_seasonal_cycle.
"""

import logging

import matplotlib.pyplot as plt
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
