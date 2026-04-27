"""
Smoke tests for synhydro.plotting distribution functions.

Covers plot_flow_duration_curve, plot_cdf, plot_histogram, plot_monthly_distributions.
"""

import logging

import matplotlib.pyplot as plt
import pytest

from synhydro.plotting import (
    plot_cdf,
    plot_flow_duration_curve,
    plot_histogram,
    plot_monthly_distributions,
)

logger = logging.getLogger(__name__)


def _has_artists(ax: plt.Axes) -> bool:
    """Return True if the axes has any drawn lines, patches, or collections."""
    return (len(ax.lines) + len(ax.collections) + len(ax.patches)) > 0


# ----------------------------------------------------------------------
# plot_flow_duration_curve
# ----------------------------------------------------------------------


def test_plot_flow_duration_curve_default(small_ensemble):
    fig, ax = plot_flow_duration_curve(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_flow_duration_curve_with_observed(small_ensemble, observed_series):
    fig, ax = plot_flow_duration_curve(small_ensemble, observed=observed_series)
    assert _has_artists(ax)


def test_plot_flow_duration_curve_no_annual_range(small_ensemble):
    fig, ax = plot_flow_duration_curve(small_ensemble, show_annual_range=False)
    assert _has_artists(ax)


def test_plot_flow_duration_curve_log_scale(small_ensemble):
    fig, ax = plot_flow_duration_curve(small_ensemble, log_scale=True)
    assert ax.get_yscale() == "log"


# ----------------------------------------------------------------------
# plot_cdf
# ----------------------------------------------------------------------


def test_plot_cdf_default(small_ensemble):
    fig, ax = plot_cdf(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_cdf_with_observed(small_ensemble, observed_series):
    fig, ax = plot_cdf(small_ensemble, observed=observed_series)
    assert _has_artists(ax)


def test_plot_cdf_log_scale(small_ensemble):
    fig, ax = plot_cdf(small_ensemble, log_scale=True)
    assert ax.get_xscale() == "log"


def test_plot_cdf_no_annual_range(small_ensemble):
    fig, ax = plot_cdf(small_ensemble, show_annual_range=False)
    assert _has_artists(ax)


# ----------------------------------------------------------------------
# plot_histogram
# ----------------------------------------------------------------------


def test_plot_histogram_default(small_ensemble):
    fig, ax = plot_histogram(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_histogram_with_observed(small_ensemble, observed_series):
    fig, ax = plot_histogram(small_ensemble, observed=observed_series)
    assert _has_artists(ax)


def test_plot_histogram_no_kde_no_density(small_ensemble):
    fig, ax = plot_histogram(small_ensemble, show_kde=False, density=False)
    assert _has_artists(ax)
    assert ax.get_ylabel() == "Count"


def test_plot_histogram_log_x(small_ensemble):
    fig, ax = plot_histogram(small_ensemble, log_x=True, show_kde=False)
    assert ax.get_xscale() == "log"


# ----------------------------------------------------------------------
# plot_monthly_distributions
# ----------------------------------------------------------------------


def test_plot_monthly_distributions_default(small_ensemble):
    fig, ax = plot_monthly_distributions(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert _has_artists(ax)


def test_plot_monthly_distributions_with_observed(small_ensemble, observed_series):
    fig, ax = plot_monthly_distributions(small_ensemble, observed=observed_series)
    assert _has_artists(ax)


def test_plot_monthly_distributions_violin(small_ensemble):
    fig, ax = plot_monthly_distributions(small_ensemble, plot_type="violin")
    assert _has_artists(ax)


def test_plot_monthly_distributions_invalid_plot_type(small_ensemble):
    with pytest.raises(ValueError, match="plot_type"):
        plot_monthly_distributions(small_ensemble, plot_type="strip")
