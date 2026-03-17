"""
Tests for Valencia-Schaake temporal disaggregator (annual to monthly).
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.disaggregation.temporal.valencia_schaake import (
    ValenciaSchaakeDisaggregator,
)
from synhydro.core.ensemble import Ensemble, EnsembleMetadata


@pytest.fixture
def sample_annual_series():
    """Generate a sample monthly time series for testing (20 years of monthly data)."""
    dates = pd.date_range(start="2000-01-01", end="2019-12-31", freq="MS")
    np.random.seed(42)
    values = np.random.lognormal(mean=6.0, sigma=0.5, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_annual_dataframe():
    """Generate a sample monthly multi-site DataFrame for testing (20 years, 3 sites)."""
    dates = pd.date_range(start="2000-01-01", end="2019-12-31", freq="MS")
    np.random.seed(42)
    n_sites = 3
    data = {}
    for i in range(n_sites):
        base = np.random.lognormal(mean=6.0, sigma=0.5, size=len(dates))
        noise = np.random.normal(0, 50, size=len(dates))
        data[f"site_{i+1}"] = np.maximum(base + noise, 1.0)  # Ensure positive values
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_monthly_ensemble():
    """Generate sample annual ensemble for disaggregation testing."""
    dates = pd.date_range(start="2000-01-01", end="2002-12-31", freq="YS")
    np.random.seed(42)

    ensemble = {}
    for realization in range(3):
        data = {}
        for site in ["site_1"]:
            np.random.seed(42 + realization)
            data[site] = np.random.lognormal(mean=7.0, sigma=0.4, size=len(dates))
        ensemble[realization] = pd.DataFrame(data, index=dates)

    metadata = EnsembleMetadata(
        generator_class="TestGenerator",
        n_realizations=3,
        n_sites=1,
        time_resolution="YS",
    )

    return Ensemble(ensemble, metadata=metadata)


class TestValenciaSchaakeInitialization:
    """Tests for ValenciaSchaakeDisaggregator initialization."""

    def test_initialization_default_params(self, sample_annual_series):
        """Test initialization with default parameters."""
        disagg = ValenciaSchaakeDisaggregator()
        assert disagg.n_subperiods == 12
        assert disagg.transform == "log"
        assert disagg.conservation_method == "proportional"

    def test_initialization_custom_params(self, sample_annual_series):
        """Test initialization with custom parameters."""
        disagg = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="boxcox", conservation_method="none"
        )
        assert disagg.n_subperiods == 4
        assert disagg.transform == "boxcox"
        assert disagg.conservation_method == "none"

    def test_initialization_properties(self, sample_annual_series):
        """Test frequency properties."""
        disagg = ValenciaSchaakeDisaggregator()
        assert disagg.input_frequency == "YS"
        assert disagg.output_frequency == "MS"


class TestValenciaSchaakePreprocessing:
    """Tests for ValenciaSchaakeDisaggregator preprocessing."""

    def test_preprocessing_annual_series(self, sample_annual_series):
        """Test preprocessing with monthly Series (aggregates to annual)."""
        disagg = ValenciaSchaakeDisaggregator()
        disagg.preprocessing(sample_annual_series)
        assert disagg.is_preprocessed is True
        assert hasattr(disagg, "Q_obs")
        assert hasattr(disagg, "Q_annual")

    def test_preprocessing_annual_dataframe(self, sample_annual_dataframe):
        """Test preprocessing with monthly DataFrame (aggregates to annual)."""
        disagg = ValenciaSchaakeDisaggregator()
        disagg.preprocessing(sample_annual_dataframe)
        assert disagg.is_preprocessed is True
        assert disagg.n_sites == 3

    def test_preprocessing_creates_annual_aggregates(self, sample_annual_series):
        """Test that preprocessing creates annual aggregates from monthly data."""
        disagg = ValenciaSchaakeDisaggregator()
        disagg.preprocessing(sample_annual_series)
        # Should aggregate monthly data to annual
        assert len(disagg.Q_annual) > 0
        assert len(disagg.Q_annual) == 20  # 20 years of data


class TestValenciaSchaakeFit:
    """Tests for ValenciaSchaakeDisaggregator fitting."""

    def test_fit_single_site(self, sample_annual_series):
        """Test fitting with single site."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.is_fitted is True
        assert disagg.mu_X_ is not None
        assert disagg.S_XX_ is not None
        assert disagg.mu_Y_ is not None
        assert disagg.sigma_Y_sq_ is not None
        assert disagg.A_ is not None
        assert disagg.C_ is not None

    def test_fit_multiple_sites(self, sample_annual_dataframe):
        """Test fitting with multiple sites."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_dataframe)
        disagg.fit()

        assert disagg.is_fitted is True
        assert disagg.is_multisite is True
        assert disagg.n_sites == 3

    def test_fit_statistics_shapes(self, sample_annual_series):
        """Test that fitted statistics have correct shapes."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.mu_X_.shape == (disagg.n_subperiods,)
        assert disagg.S_XX_.shape == (disagg.n_subperiods, disagg.n_subperiods)
        assert disagg.A_.shape == (disagg.n_subperiods,)
        assert disagg.C_.shape == (disagg.n_subperiods, disagg.n_subperiods)

    def test_fit_with_log_transform(self, sample_annual_series):
        """Test fitting with log transformation."""
        disagg = ValenciaSchaakeDisaggregator(transform="log")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.is_fitted is True
        assert disagg.transform_params_["type"] == "log"

    def test_fit_with_boxcox_transform(self, sample_annual_series):
        """Test fitting with Box-Cox transformation."""
        disagg = ValenciaSchaakeDisaggregator(transform="boxcox")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.is_fitted is True
        assert disagg.transform_params_["type"] == "boxcox"
        assert "lambda" in disagg.transform_params_

    def test_fit_without_transform(self, sample_annual_series):
        """Test fitting without transformation."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.is_fitted is True
        # When no transform is applied, transform_params_ is empty
        assert (
            len(disagg.transform_params_) == 0
            or disagg.transform_params_.get("type") == "none"
        )

    def test_fit_computes_fitted_params(self, sample_annual_series):
        """Test that fit computes fitted parameters object."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        assert disagg.fitted_params_ is not None
        assert disagg.fitted_params_.n_parameters_ > 0
        assert disagg.fitted_params_.sample_size_ > 0


class TestValenciaSchaakeDisaggregation:
    """Tests for disaggregation functionality."""

    def test_disaggregate_single_year(self, sample_annual_series):
        """Test disaggregating single year."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        # Create ensemble for single year
        annual_value = np.array([1200.0])
        annual_dates = pd.DatetimeIndex(["2020-01-01"], freq="YS")
        annual_df = pd.DataFrame({"site_1": annual_value}, index=annual_dates)

        monthly_df = disagg._disaggregate_single_realization(annual_df)

        assert isinstance(monthly_df, pd.DataFrame)
        assert len(monthly_df) == 12  # 12 months
        assert monthly_df.shape[1] == 1  # 1 site

    def test_disaggregate_multiple_years(self, sample_annual_series):
        """Test disaggregating multiple years."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        annual_values = np.random.lognormal(mean=7.0, sigma=0.4, size=3)
        annual_dates = pd.date_range("2020-01-01", periods=3, freq="YS")
        annual_df = pd.DataFrame({"site_1": annual_values}, index=annual_dates)

        monthly_df = disagg._disaggregate_single_realization(annual_df)

        assert isinstance(monthly_df, pd.DataFrame)
        assert len(monthly_df) == 36  # 3 years * 12 months
        assert monthly_df.shape[1] == 1

    def test_disaggregate_ensemble(self, sample_annual_series):
        """Test disaggregating via Ensemble interface."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        # Create test ensemble with single site, annual synthetic data
        # Use 5 years to test multiple years of disaggregation
        dates = pd.date_range(start="2000-01-01", periods=5, freq="YS")
        np.random.seed(42)

        ensemble_data = {}
        for realization in range(2):
            data = {"site_1": np.random.lognormal(mean=7.0, sigma=0.4, size=len(dates))}
            ensemble_data[realization] = pd.DataFrame(data, index=dates)

        metadata = EnsembleMetadata(
            generator_class="TestGenerator",
            n_realizations=2,
            n_sites=1,
            time_resolution="YS",
        )

        ensemble_in = Ensemble(ensemble_data, metadata=metadata)

        # Disaggregate
        ensemble_out = disagg.disaggregate(ensemble_in)

        assert isinstance(ensemble_out, Ensemble)
        assert ensemble_out.metadata.time_resolution == "MS"
        assert len(ensemble_out.realization_ids) == 2
        assert ensemble_out.metadata.n_sites == 1
        # 5 years of annual data should produce 5*12=60 months
        assert len(ensemble_out.data_by_realization[0]) == 60

    def test_disaggregation_preserves_sum(self, sample_annual_series):
        """Test that disaggregation preserves annual total."""
        disagg = ValenciaSchaakeDisaggregator(
            transform="log", conservation_method="proportional"
        )
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        annual_value = 1200.0
        annual_dates = pd.DatetimeIndex(["2020-01-01"], freq="YS")
        annual_df = pd.DataFrame({"site_1": [annual_value]}, index=annual_dates)

        monthly_df = disagg._disaggregate_single_realization(annual_df)
        monthly_sum = monthly_df["site_1"].sum()

        # Should preserve the annual total
        assert np.abs(monthly_sum - annual_value) < 1.0  # Allow small tolerance

    def test_disaggregation_produces_non_negative(self, sample_annual_series):
        """Test that disaggregation produces non-negative flows."""
        disagg = ValenciaSchaakeDisaggregator(transform="log")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        annual_value = 1200.0
        annual_dates = pd.DatetimeIndex(["2020-01-01"], freq="YS")
        annual_df = pd.DataFrame({"site_1": [annual_value]}, index=annual_dates)

        monthly_df = disagg._disaggregate_single_realization(annual_df)

        # All values should be non-negative
        assert (monthly_df.values >= 0).all()

    def test_disaggregation_reproducible(self, sample_annual_series):
        """Test that disaggregation is reproducible with seed."""
        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(sample_annual_series)
        disagg.fit()

        annual_dates = pd.DatetimeIndex(["2020-01-01"], freq="YS")
        annual_df = pd.DataFrame({"site_1": [1200.0]}, index=annual_dates)

        # Run twice with same seed
        np.random.seed(42)
        monthly_1 = disagg._disaggregate_single_realization(annual_df.copy())

        np.random.seed(42)
        monthly_2 = disagg._disaggregate_single_realization(annual_df.copy())

        # Results should be identical
        np.testing.assert_array_almost_equal(monthly_1.values, monthly_2.values)


class TestValenciaSchaakeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fit_before_preprocessing_raises_error(self, sample_annual_series):
        """Test that fit without preprocessing raises error."""
        disagg = ValenciaSchaakeDisaggregator()
        with pytest.raises(ValueError):
            disagg.fit()

    def test_disaggregate_before_fit_raises_error(
        self, sample_monthly_ensemble, sample_annual_series
    ):
        """Test that disaggregate without fit raises error."""
        disagg = ValenciaSchaakeDisaggregator()
        disagg.preprocessing(sample_annual_series)

        with pytest.raises(ValueError):
            disagg.disaggregate(sample_monthly_ensemble)

    def test_small_sample_size(self):
        """Test behavior with very small sample size (3 years of monthly data)."""
        dates = pd.date_range(start="2000-01-01", end="2002-12-31", freq="MS")
        values = np.random.lognormal(mean=6.0, sigma=0.5, size=len(dates))
        Q_obs = pd.Series(values, index=dates, name="site_1")

        disagg = ValenciaSchaakeDisaggregator(transform="none")
        disagg.preprocessing(Q_obs)
        # Should not raise error
        disagg.fit()
        assert disagg.is_fitted

    def test_zero_flows_handled(self):
        """Test that zero flows are handled appropriately (6 years of monthly data)."""
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="MS")
        # Include some zero or near-zero values
        values = np.array(
            [1000.0, 0.1, 1100.0, 500.0, 1200.0, 300.0] * 12
        )  # 6 years of monthly data
        Q_obs = pd.Series(values[: len(dates)], index=dates, name="site_1")

        disagg = ValenciaSchaakeDisaggregator(transform="log")
        disagg.preprocessing(Q_obs)
        disagg.fit()
        assert disagg.is_fitted
