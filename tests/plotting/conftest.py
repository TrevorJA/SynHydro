"""
Shared fixtures for plotting module tests.

These fixtures construct small synthetic Ensembles directly from numpy
arrays, bypassing the generator pipeline so that plotting tests stay fast.
"""

import logging

import matplotlib

# Use non-interactive backend so tests do not pop windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble, EnsembleMetadata

logger = logging.getLogger(__name__)


def _build_ensemble(
    n_realizations: int,
    site_names: list,
    dates: pd.DatetimeIndex,
    seed: int,
    time_resolution: str,
) -> Ensemble:
    """Construct an Ensemble from random lognormal flow values.

    Parameters
    ----------
    n_realizations : int
        Number of realizations to build.
    site_names : list of str
        Site column names.
    dates : pd.DatetimeIndex
        Time index for each realization.
    seed : int
        Seed for the numpy Generator (one root seed; per-realization streams
        are derived deterministically).
    time_resolution : str
        Pandas frequency string stored on the Ensemble metadata.

    Returns
    -------
    Ensemble
        Newly constructed ensemble with the requested shape.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(site_names)
    data = {}
    for r in range(n_realizations):
        values = rng.lognormal(mean=2.5, sigma=0.5, size=(len(dates), n_sites))
        data[r] = pd.DataFrame(values, index=dates, columns=site_names)
    metadata = EnsembleMetadata(time_resolution=time_resolution)
    return Ensemble(data, metadata=metadata)


@pytest.fixture(scope="session")
def small_ensemble():
    """5-realization, 2-site, 3-year daily ensemble used by most plotting tests."""
    dates = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    return _build_ensemble(
        n_realizations=5,
        site_names=["site_A", "site_B"],
        dates=dates,
        seed=42,
        time_resolution="D",
    )


@pytest.fixture(scope="session")
def single_site_ensemble():
    """Same shape as small_ensemble but with only one site (for spatial-corr error case)."""
    dates = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    return _build_ensemble(
        n_realizations=5,
        site_names=["site_A"],
        dates=dates,
        seed=43,
        time_resolution="D",
    )


@pytest.fixture(scope="session")
def monthly_ensemble():
    """Monthly-frequency ensemble for the timestep-guardrail tests."""
    dates = pd.date_range("2000-01-01", periods=12 * 5, freq="MS")
    return _build_ensemble(
        n_realizations=5,
        site_names=["site_A", "site_B"],
        dates=dates,
        seed=44,
        time_resolution="MS",
    )


@pytest.fixture(scope="session")
def long_daily_ensemble():
    """Longer daily ensemble for SSI-based tests (need ~20+ years for SSI fitting)."""
    dates = pd.date_range("1980-01-01", periods=365 * 25, freq="D")
    return _build_ensemble(
        n_realizations=5,
        site_names=["site_A", "site_B"],
        dates=dates,
        seed=45,
        time_resolution="D",
    )


@pytest.fixture(scope="session")
def observed_series(small_ensemble):
    """Single-site observed Series aligned to small_ensemble's time index."""
    rng = np.random.default_rng(7)
    site = small_ensemble.site_names[0]
    idx = small_ensemble.data_by_site[site].index
    values = rng.lognormal(mean=2.5, sigma=0.5, size=len(idx))
    return pd.Series(values, index=idx, name=site)


@pytest.fixture(scope="session")
def observed_dataframe(small_ensemble):
    """Multi-site observed DataFrame aligned to small_ensemble's time index."""
    rng = np.random.default_rng(8)
    sites = small_ensemble.site_names
    idx = small_ensemble.data_by_site[sites[0]].index
    data = {s: rng.lognormal(mean=2.5, sigma=0.5, size=len(idx)) for s in sites}
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="session")
def long_observed_series(long_daily_ensemble):
    """Single-site observed Series aligned to long_daily_ensemble's index for SSI tests."""
    rng = np.random.default_rng(9)
    site = long_daily_ensemble.site_names[0]
    idx = long_daily_ensemble.data_by_site[site].index
    values = rng.lognormal(mean=2.5, sigma=0.5, size=len(idx))
    return pd.Series(values, index=idx, name=site)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Close all matplotlib figures after each test to keep memory bounded."""
    yield
    plt.close("all")
