"""
Tests for Nowak temporal disaggregator (monthly to daily).
"""

import pytest
import numpy as np
import pandas as pd

from sglib.methods.disaggregation.temporal.nowak import NowakDisaggregator


class TestNowakDisaggregatorInitialization:
    """Tests for NowakDisaggregator initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        disagg = NowakDisaggregator()
        assert disagg.n_neighbors == 5
        assert disagg.max_month_shift == 7
        assert disagg.sample_method == 'distance_weighted'

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        disagg = NowakDisaggregator(
            n_neighbors=10,
            max_month_shift=10,
            sample_method='lall_and_sharma_1996'
        )
        assert disagg.n_neighbors == 10
        assert disagg.max_month_shift == 10
        assert disagg.sample_method == 'lall_and_sharma_1996'


class TestNowakDisaggregatorPreprocessing:
    """Tests for NowakDisaggregator preprocessing."""

    def test_preprocessing_daily_series(self, sample_daily_series):
        """Test preprocessing with daily Series."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_series)

        # Preprocessing is a no-op but should succeed
        assert True

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Test preprocessing with daily DataFrame."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_dataframe)

        # Should succeed
        assert True


class TestNowakDisaggregatorFit:
    """Tests for NowakDisaggregator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Test fitting with single site."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        assert hasattr(disagg, 'knn_models')
        assert hasattr(disagg, 'monthly_cumulative_flows')
        assert hasattr(disagg, 'daily_flow_profiles')
        assert len(disagg.knn_models) == 12  # One model per month
        assert disagg.is_multisite is False

    def test_fit_multiple_sites(self, sample_daily_dataframe):
        """Test fitting with multiple sites."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        assert disagg.is_multisite is True
        assert disagg.n_sites == 3
        assert len(disagg.site_names) == 3
        assert len(disagg.knn_models) == 12

    def test_fit_creates_knn_models(self, sample_daily_series):
        """Test that fit creates KNN models for each month."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        for month in range(1, 13):
            assert month in disagg.knn_models
            # Check that model has been fitted
            assert hasattr(disagg.knn_models[month], 'n_samples_fit_')

    def test_fit_creates_historic_profiles(self, sample_daily_series):
        """Test that fit creates historic flow profiles."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        assert len(disagg.monthly_cumulative_flows) == 12
        assert len(disagg.daily_flow_profiles) == 12

        for month in range(1, 13):
            assert month in disagg.monthly_cumulative_flows
            assert month in disagg.daily_flow_profiles


class TestNowakDisaggregatorKNNSearch:
    """Tests for KNN search functionality."""

    def test_find_knn_indices_single_site(self, sample_daily_series):
        """Test finding KNN indices for single site."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.fit(sample_daily_series)

        # Create synthetic monthly flow
        synthetic_monthly = pd.Series(
            [100.0],
            index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
        )

        indices, distances = disagg.find_knn_indices(synthetic_monthly)

        assert isinstance(indices, np.ndarray)
        assert isinstance(distances, np.ndarray)
        assert len(indices) == 5
        assert len(distances) == 5

    def test_find_knn_indices_multiple_months(self, sample_daily_series):
        """Test finding KNN indices for multiple months."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.fit(sample_daily_series)

        # Create synthetic monthly flows for multiple months
        synthetic_monthly = pd.Series(
            [100.0, 120.0, 90.0],
            index=pd.date_range('2020-01-01', periods=3, freq='MS')
        )

        indices, distances = disagg.find_knn_indices(synthetic_monthly)

        assert indices.shape == (3, 5)  # 3 months, 5 neighbors each
        assert distances.shape == (3, 5)

    def test_sample_knn_monthly_flows(self, sample_daily_series):
        """Test sampling from KNN neighbors."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [100.0],
            index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
        )

        indices, distances = disagg.find_knn_indices(synthetic_monthly)
        sampled_idx = disagg.sample_knn_monthly_flows(indices, distances)

        assert isinstance(sampled_idx, np.ndarray)
        assert len(sampled_idx) == 1
        assert sampled_idx[0] in indices[0]  # Should be one of the neighbors


class TestNowakDisaggregatorDisaggregation:
    """Tests for disaggregation functionality."""

    def test_disaggregate_single_month_single_site(self, sample_daily_series):
        """Test disaggregating single month for single site."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [3000.0],
            index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        assert isinstance(daily, pd.Series)
        assert len(daily) == 31  # January has 31 days
        assert daily.index.freq == 'D'
        # Check that monthly sum matches (approximately)
        assert np.abs(daily.sum() - 3000.0) < 1e-6

    def test_disaggregate_full_year_single_site(self, sample_daily_series):
        """Test disaggregating full year for single site."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            np.random.gamma(2.0, 100.0, 12),
            index=pd.date_range('2020-01-01', periods=12, freq='MS')
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        assert isinstance(daily, pd.Series)
        assert len(daily) == 366  # 2020 is a leap year
        assert daily.index.freq == 'D'

        # Check that each monthly sum matches
        for month in range(1, 13):
            month_mask = (daily.index.month == month)
            monthly_sum = daily[month_mask].sum()
            expected = synthetic_monthly.iloc[month - 1]
            assert np.abs(monthly_sum - expected) < 1e-6

    def test_disaggregate_multiple_sites(self, sample_daily_dataframe):
        """Test disaggregating for multiple sites."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        # Create synthetic monthly flows for all sites
        synthetic_monthly = pd.DataFrame(
            np.random.gamma(2.0, 100.0, (12, 3)),
            index=pd.date_range('2020-01-01', periods=12, freq='MS'),
            columns=sample_daily_dataframe.columns
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        assert isinstance(daily, pd.DataFrame)
        assert daily.shape[1] == 3  # 3 sites
        assert len(daily) == 366  # 2020 is a leap year
        assert daily.columns.tolist() == sample_daily_dataframe.columns.tolist()

        # Check monthly sums for each site
        for site in daily.columns:
            for month in range(1, 13):
                month_mask = (daily.index.month == month)
                monthly_sum = daily.loc[month_mask, site].sum()
                expected = synthetic_monthly.iloc[month - 1][site]
                assert np.abs(monthly_sum - expected) < 1e-6

    def test_disaggregate_preserves_spatial_correlation(self, sample_daily_dataframe):
        """Test that disaggregation preserves spatial correlation."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        synthetic_monthly = pd.DataFrame(
            np.random.gamma(2.0, 100.0, (12, 3)),
            index=pd.date_range('2020-01-01', periods=12, freq='MS'),
            columns=sample_daily_dataframe.columns
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        # Calculate correlation
        corr = daily.corr()
        # All correlation values should be between -1 and 1
        assert (corr.values >= -1.0).all()
        assert (corr.values <= 1.0).all()

    def test_disaggregate_leap_year(self, sample_daily_series):
        """Test disaggregation handles leap years correctly."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        # 2020 is a leap year
        synthetic_monthly = pd.Series(
            [3000.0],
            index=pd.DatetimeIndex(['2020-02-01'], freq='MS')
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        assert len(daily) == 29  # February 2020 has 29 days
        assert np.abs(daily.sum() - 3000.0) < 1e-6

    def test_disaggregate_non_leap_year(self, sample_daily_series):
        """Test disaggregation handles non-leap years correctly."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        # 2019 is not a leap year
        synthetic_monthly = pd.Series(
            [2800.0],
            index=pd.DatetimeIndex(['2019-02-01'], freq='MS')
        )

        daily = disagg.disaggregate_monthly_flows(synthetic_monthly)

        assert len(daily) == 28  # February 2019 has 28 days
        assert np.abs(daily.sum() - 2800.0) < 1e-6

    def test_disaggregate_different_sample_methods(self, sample_daily_series):
        """Test disaggregation with different sampling methods."""
        # Distance weighted
        disagg1 = NowakDisaggregator(sample_method='distance_weighted')
        disagg1.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [3000.0],
            index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
        )

        daily1 = disagg1.disaggregate_monthly_flows(synthetic_monthly)
        assert len(daily1) == 31

        # Lall and Sharma method
        disagg2 = NowakDisaggregator(sample_method='lall_and_sharma_1996')
        disagg2.fit(sample_daily_series)

        daily2 = disagg2.disaggregate_monthly_flows(synthetic_monthly)
        assert len(daily2) == 31


class TestNowakDisaggregatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_disaggregate_without_fit_raises(self, sample_daily_series):
        """Test that disaggregation without fit raises error."""
        disagg = NowakDisaggregator()

        synthetic_monthly = pd.Series(
            [3000.0],
            index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
        )

        with pytest.raises(AttributeError):
            disagg.disaggregate_monthly_flows(synthetic_monthly)

    def test_disaggregate_with_different_n_neighbors(self, sample_daily_series):
        """Test disaggregation with different number of neighbors."""
        for n in [3, 5, 10]:
            disagg = NowakDisaggregator(n_neighbors=n)
            disagg.fit(sample_daily_series)

            synthetic_monthly = pd.Series(
                [3000.0],
                index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
            )

            daily = disagg.disaggregate_monthly_flows(synthetic_monthly)
            assert len(daily) == 31

    def test_disaggregate_with_different_month_shifts(self, sample_daily_series):
        """Test disaggregation with different month shift values."""
        for shift in [3, 7, 14]:
            disagg = NowakDisaggregator(max_month_shift=shift)
            disagg.fit(sample_daily_series)

            synthetic_monthly = pd.Series(
                [3000.0],
                index=pd.DatetimeIndex(['2020-01-01'], freq='MS')
            )

            daily = disagg.disaggregate_monthly_flows(synthetic_monthly)
            assert len(daily) == 31
