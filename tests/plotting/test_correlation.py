"""
Smoke tests for synhydro.plotting correlation functions.

Covers plot_autocorrelation, plot_spatial_correlation.
"""

import logging

import matplotlib.pyplot as plt
import pytest

from synhydro.plotting import plot_autocorrelation, plot_spatial_correlation

logger = logging.getLogger(__name__)


def _has_artists(ax: plt.Axes) -> bool:
    """Return True if the axes has any drawn lines or collections."""
    return (len(ax.lines) + len(ax.collections)) > 0


# ----------------------------------------------------------------------
# plot_autocorrelation
# ----------------------------------------------------------------------


def test_plot_autocorrelation_default(small_ensemble):
    fig, ax = plot_autocorrelation(small_ensemble, max_lag=10)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_autocorrelation_with_observed(small_ensemble, observed_series):
    fig, ax = plot_autocorrelation(small_ensemble, observed=observed_series, max_lag=10)
    assert _has_artists(ax)


def test_plot_autocorrelation_show_members(small_ensemble):
    fig, ax = plot_autocorrelation(small_ensemble, max_lag=10, show_members=2)
    assert _has_artists(ax)


def test_plot_autocorrelation_monthly_timestep(small_ensemble):
    fig, ax = plot_autocorrelation(small_ensemble, max_lag=6, timestep="monthly")
    assert _has_artists(ax)


# ----------------------------------------------------------------------
# plot_spatial_correlation
# ----------------------------------------------------------------------


def test_plot_spatial_correlation_default(small_ensemble):
    fig, axes = plot_spatial_correlation(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 1


def test_plot_spatial_correlation_with_observed(small_ensemble, observed_dataframe):
    fig, axes = plot_spatial_correlation(small_ensemble, observed=observed_dataframe)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    # Side-by-side: 2 axes for the heatmaps.
    assert len(axes) == 2


def test_plot_spatial_correlation_show_difference(small_ensemble, observed_dataframe):
    fig, axes = plot_spatial_correlation(
        small_ensemble, observed=observed_dataframe, show_difference=True
    )
    assert isinstance(axes, list)
    assert len(axes) == 1


def test_plot_spatial_correlation_specific_realization(small_ensemble):
    fig, axes = plot_spatial_correlation(small_ensemble, realization=0)
    assert isinstance(axes, list)
    assert len(axes) == 1


def test_plot_spatial_correlation_single_site_raises(single_site_ensemble):
    with pytest.raises(ValueError, match="multi-site"):
        plot_spatial_correlation(single_site_ensemble)


def test_plot_spatial_correlation_unknown_realization_raises(small_ensemble):
    with pytest.raises(ValueError, match="not found"):
        plot_spatial_correlation(small_ensemble, realization=999)
