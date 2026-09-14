"""
Smoke tests for synhydro.plotting correlation functions.

Covers plot_autocorrelation, plot_spatial_correlation.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from synhydro.plotting import plot_autocorrelation, plot_spatial_correlation

logger = logging.getLogger(__name__)


def _has_artists(ax: plt.Axes) -> bool:
    """Return True if the axes has any drawn lines or collections."""
    return (len(ax.lines) + len(ax.collections)) > 0


def _member_ydata(ax: plt.Axes, n_members: int) -> list:
    """Return y-data of the first ``n_members`` lines (the member lines)."""
    return [np.asarray(line.get_ydata()) for line in ax.lines[:n_members]]


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


def test_plot_autocorrelation_seed_reproduces_member_selection(small_ensemble):
    """The same seed (int or Generator) draws the same ensemble members."""
    _, ax_a = plot_autocorrelation(small_ensemble, max_lag=10, show_members=2, seed=3)
    _, ax_b = plot_autocorrelation(small_ensemble, max_lag=10, show_members=2, seed=3)
    _, ax_c = plot_autocorrelation(
        small_ensemble, max_lag=10, show_members=2, seed=np.random.default_rng(3)
    )
    ydata_a = _member_ydata(ax_a, 2)
    assert len(ydata_a) == 2
    for y_a, y_b, y_c in zip(ydata_a, _member_ydata(ax_b, 2), _member_ydata(ax_c, 2)):
        np.testing.assert_array_equal(y_a, y_b)
        np.testing.assert_array_equal(y_a, y_c)


def test_plot_autocorrelation_seed_none_still_draws_members(small_ensemble):
    fig, ax = plot_autocorrelation(
        small_ensemble, max_lag=10, show_members=2, seed=None
    )
    # Two member lines plus the ensemble median
    assert len(ax.lines) >= 3
    assert _has_artists(ax)


def test_plot_autocorrelation_seed_is_keyword_only(small_ensemble):
    with pytest.raises(TypeError):
        plot_autocorrelation(
            small_ensemble,
            None,
            None,
            10,
            "daily",
            2,
            None,
            (6, 4),
            None,
            None,
            None,
            True,
            True,
            None,
            100,
            3,
        )


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
