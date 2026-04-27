"""
Smoke tests for synhydro.plotting.validation.plot_validation_panel.
"""

import logging

import matplotlib.pyplot as plt
import pytest

from synhydro.plotting import plot_validation_panel

logger = logging.getLogger(__name__)


def test_plot_validation_panel_default(small_ensemble):
    fig, axes = plot_validation_panel(small_ensemble)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    # Documented: 5 panels.
    assert len(axes) == 5


def test_plot_validation_panel_with_observed(small_ensemble, observed_series):
    fig, axes = plot_validation_panel(small_ensemble, observed=observed_series)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 5


def test_plot_validation_panel_log_space(small_ensemble, observed_series):
    fig, axes = plot_validation_panel(
        small_ensemble, observed=observed_series, log_space=True
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 5


def test_plot_validation_panel_weekly(small_ensemble, observed_series):
    fig, axes = plot_validation_panel(
        small_ensemble, observed=observed_series, timestep="weekly"
    )
    assert len(axes) == 5


def test_plot_validation_panel_invalid_timestep(small_ensemble):
    with pytest.raises(ValueError, match="timestep"):
        plot_validation_panel(small_ensemble, timestep="annual")
