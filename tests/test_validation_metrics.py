"""
Tests for ensemble validation metrics, including L-moment and extreme value categories.
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.validation import validate_ensemble, ValidationResult
from synhydro.core.validation._metrics import (
    _compute_lmoment_metrics,
    _compute_extreme_metrics,
    _lkurtosis,
)


@pytest.fixture
def monthly_obs():
    """20-year monthly observed data, 2 sites, lognormal with seasonal pattern."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", "2019-12-31", freq="MS")
    n = len(dates)
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
    data = {
        "site_A": rng.lognormal(5.0, 0.4, n) * seasonal,
        "site_B": rng.lognormal(4.5, 0.5, n) * seasonal,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def monthly_ensemble(monthly_obs):
    """Ensemble of 10 realizations x 20 years, 2 sites."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    n = len(dates)
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)

    realization_dict = {}
    for i in range(10):
        data = {
            "site_A": rng.lognormal(5.0, 0.4, n) * seasonal,
            "site_B": rng.lognormal(4.5, 0.5, n) * seasonal,
        }
        realization_dict[i] = pd.DataFrame(data, index=dates)

    metadata = EnsembleMetadata(
        generator_class="TestGenerator",
        n_realizations=10,
        n_sites=2,
        time_resolution="MS",
    )
    return Ensemble(realization_dict, metadata=metadata)


# ---------------------------------------------------------------------------
# L-moment metrics
# ---------------------------------------------------------------------------


class TestLmomentMetrics:
    """Tests for L-moment ratio validation metrics."""

    def test_lmoment_metrics_returns_sites(self, monthly_ensemble, monthly_obs):
        result = _compute_lmoment_metrics(
            monthly_ensemble, monthly_obs, ["site_A", "site_B"]
        )
        assert "site_A" in result
        assert "site_B" in result

    def test_lmoment_metrics_keys(self, monthly_ensemble, monthly_obs):
        result = _compute_lmoment_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        metrics = result["site_A"]
        assert "l_cv" in metrics
        assert "l_skewness" in metrics
        assert "l_kurtosis" in metrics

    def test_lmoment_entry_structure(self, monthly_ensemble, monthly_obs):
        result = _compute_lmoment_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        entry = result["site_A"]["l_cv"]
        assert "observed" in entry
        assert "synthetic_median" in entry
        assert "synthetic_p10" in entry
        assert "synthetic_p90" in entry
        assert "relative_error" in entry

    def test_lmoment_values_reasonable(self, monthly_ensemble, monthly_obs):
        """L-CV should be positive for lognormal data; L-skewness in (-1, 1)."""
        result = _compute_lmoment_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        lcv = result["site_A"]["l_cv"]
        assert lcv["observed"] > 0
        assert lcv["synthetic_median"] > 0

        lskew = result["site_A"]["l_skewness"]
        assert -1 < lskew["observed"] < 1
        assert -1 < lskew["synthetic_median"] < 1

    def test_lmoment_skips_short_series(self, monthly_ensemble):
        """Sites with < 10 values should be skipped."""
        short_obs = pd.DataFrame(
            {"site_A": [1.0, 2.0, 3.0]},
            index=pd.date_range("2000-01-01", periods=3, freq="MS"),
        )
        result = _compute_lmoment_metrics(monthly_ensemble, short_obs, ["site_A"])
        assert result.get("site_A") is None or len(result.get("site_A", {})) == 0

    def test_lkurtosis_basic(self):
        """L-kurtosis for uniform distribution should be near 0."""
        rng = np.random.default_rng(123)
        x = rng.uniform(0, 1, 1000)
        lk = _lkurtosis(x)
        # Theoretical L-kurtosis of uniform is 0.0
        assert abs(lk) < 0.05

    def test_lkurtosis_short_array(self):
        """L-kurtosis should return nan for n < 5."""
        assert np.isnan(_lkurtosis(np.array([1.0, 2.0, 3.0])))


# ---------------------------------------------------------------------------
# Extreme value metrics
# ---------------------------------------------------------------------------


class TestExtremeMetrics:
    """Tests for extreme value validation metrics."""

    def test_extreme_metrics_returns_sites(self, monthly_ensemble, monthly_obs):
        result = _compute_extreme_metrics(
            monthly_ensemble, monthly_obs, ["site_A", "site_B"]
        )
        assert "site_A" in result
        assert "site_B" in result

    def test_extreme_metrics_keys(self, monthly_ensemble, monthly_obs):
        result = _compute_extreme_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        metrics = result["site_A"]
        assert "annual_max_mean" in metrics
        assert "annual_max_cv" in metrics
        assert "gev_q10" in metrics
        assert "gev_q50" in metrics
        assert "gev_q100" in metrics
        assert "annual_7q_min_mean" in metrics

    def test_extreme_entry_structure(self, monthly_ensemble, monthly_obs):
        result = _compute_extreme_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        entry = result["site_A"]["annual_max_mean"]
        assert "observed" in entry
        assert "synthetic_median" in entry
        assert "relative_error" in entry

    def test_gev_quantiles_ordered(self, monthly_ensemble, monthly_obs):
        """GEV Q10 < Q50 < Q100 (increasing return period = increasing quantile)."""
        result = _compute_extreme_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        m = result["site_A"]
        obs_q10 = m["gev_q10"]["observed"]
        obs_q50 = m["gev_q50"]["observed"]
        obs_q100 = m["gev_q100"]["observed"]
        assert obs_q10 < obs_q50 < obs_q100

    def test_extreme_skips_short_series(self, monthly_ensemble):
        """Sites with < 60 values should be skipped."""
        short_obs = pd.DataFrame(
            {"site_A": np.random.default_rng(42).lognormal(5, 0.3, 24)},
            index=pd.date_range("2000-01-01", periods=24, freq="MS"),
        )
        result = _compute_extreme_metrics(monthly_ensemble, short_obs, ["site_A"])
        assert result.get("site_A") is None or len(result.get("site_A", {})) == 0

    def test_7q_min_positive(self, monthly_ensemble, monthly_obs):
        """7-day minimum mean should be positive for positive flow data."""
        result = _compute_extreme_metrics(monthly_ensemble, monthly_obs, ["site_A"])
        q7 = result["site_A"]["annual_7q_min_mean"]
        assert q7["observed"] > 0
        assert q7["synthetic_median"] > 0


# ---------------------------------------------------------------------------
# Integration with validate_ensemble
# ---------------------------------------------------------------------------


class TestValidateEnsembleNewMetrics:
    """Test that new metrics integrate correctly with validate_ensemble."""

    def test_lmoments_in_validate_ensemble(self, monthly_ensemble, monthly_obs):
        result = validate_ensemble(monthly_ensemble, monthly_obs, metrics=["lmoments"])
        assert len(result.lmoments) > 0
        assert len(result.marginal) == 0  # should not compute other categories

    def test_extremes_in_validate_ensemble(self, monthly_ensemble, monthly_obs):
        result = validate_ensemble(monthly_ensemble, monthly_obs, metrics=["extremes"])
        assert len(result.extremes) > 0

    def test_all_metrics_includes_new(self, monthly_ensemble, monthly_obs):
        result = validate_ensemble(monthly_ensemble, monthly_obs)
        assert len(result.lmoments) > 0
        assert len(result.extremes) > 0

    def test_to_dataframe_includes_new_categories(self, monthly_ensemble, monthly_obs):
        result = validate_ensemble(monthly_ensemble, monthly_obs)
        df = result.to_dataframe()
        categories = df["category"].unique()
        assert "lmoments" in categories
        assert "extremes" in categories

    def test_summary_includes_new_metrics(self, monthly_ensemble, monthly_obs):
        result = validate_ensemble(monthly_ensemble, monthly_obs)
        assert "mean_absolute_relative_error" in result.summary
        # Summary should aggregate across all categories including new ones
        assert result.summary["mean_absolute_relative_error"] > 0
