"""
Smoke tests for synhydro.plotting drought functions.

Covers plot_drought_characteristics and plot_ssi_timeseries. These functions
fit an SSI model internally, so they require a longer-than-usual ensemble
(~20+ years) for stable fitting.
"""

import logging

import matplotlib.pyplot as plt
import pytest

from synhydro.plotting import plot_drought_characteristics, plot_ssi_timeseries

logger = logging.getLogger(__name__)


def _has_artists(ax: plt.Axes) -> bool:
    """Return True if the axes has any drawn lines, patches, or collections."""
    return (len(ax.lines) + len(ax.collections) + len(ax.patches)) > 0


# ----------------------------------------------------------------------
# plot_drought_characteristics
# ----------------------------------------------------------------------


def test_plot_drought_characteristics_default(long_daily_ensemble):
    fig, ax = plot_drought_characteristics(long_daily_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_drought_characteristics_with_observed(
    long_daily_ensemble, long_observed_series
):
    fig, ax = plot_drought_characteristics(
        long_daily_ensemble, observed=long_observed_series
    )
    assert isinstance(fig, plt.Figure)


def test_plot_drought_characteristics_alt_metrics(long_daily_ensemble):
    fig, ax = plot_drought_characteristics(
        long_daily_ensemble,
        x_metric="duration",
        y_metric="severity",
        color_metric="magnitude",
    )
    assert isinstance(fig, plt.Figure)


def test_plot_drought_characteristics_invalid_metric(long_daily_ensemble):
    with pytest.raises(ValueError, match="x_metric"):
        plot_drought_characteristics(long_daily_ensemble, x_metric="bogus")


def test_plot_drought_characteristics_invalid_method(long_daily_ensemble):
    with pytest.raises(ValueError, match="ssi"):
        plot_drought_characteristics(long_daily_ensemble, method="spi")


# ----------------------------------------------------------------------
# plot_ssi_timeseries
# ----------------------------------------------------------------------


def test_plot_ssi_timeseries_default(long_daily_ensemble):
    fig, ax = plot_ssi_timeseries(long_daily_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_ssi_timeseries_with_observed(long_daily_ensemble, long_observed_series):
    fig, ax = plot_ssi_timeseries(long_daily_ensemble, observed=long_observed_series)
    assert _has_artists(ax)


def test_plot_ssi_timeseries_custom_window(long_daily_ensemble):
    fig, ax = plot_ssi_timeseries(long_daily_ensemble, window=6)
    assert _has_artists(ax)


def test_plot_ssi_timeseries_custom_percentiles(long_daily_ensemble):
    fig, ax = plot_ssi_timeseries(long_daily_ensemble, percentiles=[25, 50, 75])
    assert _has_artists(ax)
