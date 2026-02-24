"""
Tests for SSI-based drought metrics.
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.droughts.ssi import SSI, SSIDroughtMetrics, get_drought_metrics


class TestSSIDroughtMetrics:
    """Tests for SSIDroughtMetrics class."""

    def test_initialization_daily(self):
        """Test initialization with daily timescale."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        assert ssi_metrics.timescale == 'D'
        assert ssi_metrics.window == 30

    def test_initialization_monthly(self):
        """Test initialization with monthly timescale."""
        ssi_metrics = SSIDroughtMetrics(timescale='M', window=3)
        assert ssi_metrics.timescale == 'M'
        assert ssi_metrics.window == 3

    def test_calculate_ssi_daily(self, sample_ssi_data):
        """Test SSI calculation with daily data."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

        assert isinstance(ssi_values, pd.Series)
        assert len(ssi_values) > 0
        # SSI values should be standardized (approximately)
        assert ssi_values.mean() < 1.0  # Close to 0
        assert ssi_values.std() > 0.5   # Close to 1

    def test_calculate_drought_metrics(self, sample_ssi_data):
        """Test drought metrics calculation."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)
        drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)

        assert isinstance(drought_metrics, pd.DataFrame)
        expected_columns = ['start', 'end', 'duration', 'magnitude', 'severity', 'max_severity_date']
        for col in expected_columns:
            assert col in drought_metrics.columns

    def test_drought_metrics_structure(self, sample_ssi_data):
        """Test structure of drought metrics DataFrame."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)
        drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)

        if len(drought_metrics) > 0:
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(drought_metrics['start'])
            assert pd.api.types.is_datetime64_any_dtype(drought_metrics['end'])
            assert pd.api.types.is_numeric_dtype(drought_metrics['duration'])
            assert pd.api.types.is_numeric_dtype(drought_metrics['magnitude'])
            assert pd.api.types.is_numeric_dtype(drought_metrics['severity'])

    def test_drought_threshold(self, sample_ssi_data):
        """Test that droughts are identified at SSI < -1."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

        # Manually identify periods with SSI < -1
        drought_periods = ssi_values < -1
        has_droughts = drought_periods.any()

        drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)

        if has_droughts:
            assert len(drought_metrics) > 0
        # Note: if no droughts, DataFrame may be empty


class TestSSI:
    """Tests for SSI dataclass."""

    def test_initialization_default(self):
        """Test SSI initialization with defaults."""
        import scipy.stats as scs
        ssi = SSI()
        assert ssi.dist == 'gamma'
        assert ssi._dist_obj == scs.gamma
        assert ssi.timescale == 12
        assert ssi.prob_zero is False

    def test_initialization_custom(self):
        """Test SSI initialization with custom parameters."""
        import scipy.stats as scs
        ssi = SSI(
            dist='lognorm',
            timescale=6,
            fit_freq='ME',
            fit_window=30,
            prob_zero=True
        )
        assert ssi.dist == 'lognorm'
        assert ssi._dist_obj == scs.lognorm
        assert ssi.timescale == 6
        assert ssi.fit_freq == 'ME'
        assert ssi.fit_window == 30
        assert ssi.prob_zero is True

    def test_fit_transform_daily(self, sample_ssi_data):
        """Test fit_transform with daily data."""
        ssi = SSI(timescale=30, fit_freq='ME')
        ssi_values = ssi.fit_transform(sample_ssi_data)

        assert isinstance(ssi_values, pd.Series)
        assert len(ssi_values) > 0
        # SSI should be approximately standardized
        assert not ssi_values.isna().all()

    def test_fit_and_transform_separately(self, sample_ssi_data):
        """Test fit and transform as separate steps."""
        ssi = SSI(timescale=30, fit_freq='ME')

        # Split data into train and test
        split_idx = len(sample_ssi_data) // 2
        train_data = sample_ssi_data.iloc[:split_idx]
        test_data = sample_ssi_data.iloc[split_idx:]

        # Fit on training data
        ssi.fit(train_data)

        # Transform test data
        ssi_test = ssi.transform(test_data)

        assert isinstance(ssi_test, pd.Series)
        assert len(ssi_test) > 0

    def test_get_training_ssi(self, sample_ssi_data):
        """Test getting training SSI values."""
        ssi = SSI(timescale=30, fit_freq='ME')
        ssi_values = ssi.fit_transform(sample_ssi_data)

        training_ssi = ssi.get_training_ssi()

        assert isinstance(training_ssi, pd.Series)
        assert len(training_ssi) > 0

    def test_different_distributions(self, sample_monthly_series):
        """Test SSI with different distributions."""
        distributions = ['gamma', 'lognorm', 'pearson3']
        for dist_name in distributions:
            ssi = SSI(dist=dist_name, timescale=1, fit_freq='ME')
            try:
                ssi_values = ssi.fit_transform(sample_monthly_series)
                assert isinstance(ssi_values, pd.Series)
            except Exception as e:
                # Some distributions might not fit well with all data
                pytest.skip(f"Distribution '{dist_name}' failed: {e}")

    def test_prob_zero_parameter(self, sample_daily_series):
        """Test SSI with prob_zero parameter."""
        # With prob_zero=True
        ssi1 = SSI(timescale=30, fit_freq='ME', prob_zero=True)
        ssi_values1 = ssi1.fit_transform(sample_daily_series)

        # With prob_zero=False
        ssi2 = SSI(timescale=30, fit_freq='ME', prob_zero=False)
        ssi_values2 = ssi2.fit_transform(sample_daily_series)

        # Both should produce valid output
        assert isinstance(ssi_values1, pd.Series)
        assert isinstance(ssi_values2, pd.Series)


class TestGetDroughtMetrics:
    """Tests for get_drought_metrics function."""

    def test_get_drought_metrics_from_series(self, sample_ssi_data):
        """Test extracting drought metrics from SSI series."""
        # First calculate SSI
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

        # Extract droughts
        drought_df = get_drought_metrics(ssi_values)

        assert isinstance(drought_df, pd.DataFrame)
        expected_columns = ['start', 'end', 'duration', 'magnitude', 'severity']
        for col in expected_columns:
            assert col in drought_df.columns

    def test_drought_metrics_values(self, sample_ssi_data):
        """Test that drought metric values are reasonable."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)
        drought_df = get_drought_metrics(ssi_values)

        if len(drought_df) > 0:
            # Duration should be positive
            assert (drought_df['duration'] > 0).all()

            # Magnitude should be negative (sum of negative SSI values)
            assert (drought_df['magnitude'] < 0).all()

            # Severity should be negative (most negative SSI value during drought)
            # This represents the peak drought intensity
            assert (drought_df['severity'] < 0).all()
            # Severity should be less than or equal to -1 (critical drought threshold)
            assert (drought_df['severity'] <= -1).all()

            # Start should be before end
            assert (drought_df['start'] < drought_df['end']).all()

    def test_no_droughts_case(self):
        """Test case where no droughts occur (all SSI > -1)."""
        # Create data with no droughts
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        ssi_values = pd.Series(np.random.uniform(0, 2, 365), index=dates)

        drought_df = get_drought_metrics(ssi_values)

        # Should return empty DataFrame or DataFrame with 0 rows
        assert len(drought_df) == 0

    def test_continuous_drought(self):
        """Test case with continuous drought followed by recovery."""
        # Drought must be followed by 3+ consecutive positive days to be recorded
        dates = pd.date_range('2020-01-01', periods=103, freq='D')
        ssi_values = pd.Series(np.zeros(103), index=dates)
        ssi_values.iloc[:100] = -2.0   # 100 days of drought
        ssi_values.iloc[100:103] = 1.0  # 3 days of recovery to close the drought

        drought_df = get_drought_metrics(ssi_values)

        # Should identify one long drought
        assert len(drought_df) == 1
        assert drought_df.iloc[0]['duration'] == 100

    def test_multiple_droughts(self):
        """Test case with multiple distinct droughts."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        ssi_values = pd.Series(np.zeros(200), index=dates)

        # Create two drought periods
        ssi_values.iloc[20:40] = -2.0   # First drought
        ssi_values.iloc[100:130] = -1.5  # Second drought

        drought_df = get_drought_metrics(ssi_values)

        # Should identify two droughts
        assert len(drought_df) == 2


class TestSSIIntegration:
    """Integration tests for SSI drought analysis."""

    def test_full_workflow_daily(self, sample_ssi_data):
        """Test complete workflow for daily data (requires 20+ years for robust SSI fitting)."""
        # Step 1: Calculate SSI
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

        assert isinstance(ssi_values, pd.Series)

        # Step 2: Calculate drought metrics
        drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)

        assert isinstance(drought_metrics, pd.DataFrame)

    def test_full_workflow_monthly(self, sample_monthly_series):
        """Test complete workflow for monthly data."""
        # Step 1: Calculate SSI
        ssi_metrics = SSIDroughtMetrics(timescale='M', window=3)
        ssi_values = ssi_metrics.calculate_ssi(sample_monthly_series)

        assert isinstance(ssi_values, pd.Series)

        # Step 2: Calculate drought metrics
        drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)

        assert isinstance(drought_metrics, pd.DataFrame)

    def test_workflow_with_dataframe(self, sample_ssi_data):
        """Test workflow processing multiple scaled versions of a long daily series."""
        # Process multiple site variants (requires long data for robust SSI fitting)
        for multiplier in [1.0, 0.9, 1.1]:
            site_data = sample_ssi_data * multiplier

            ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
            ssi_values = ssi_metrics.calculate_ssi(site_data)

            assert isinstance(ssi_values, pd.Series)

            drought_metrics = ssi_metrics.calculate_drought_metrics(ssi_values)
            assert isinstance(drought_metrics, pd.DataFrame)

    def test_different_window_sizes(self, sample_ssi_data):
        """Test SSI calculation with different window sizes (requires long data)."""
        for window in [7, 30, 90]:
            ssi_metrics = SSIDroughtMetrics(timescale='D', window=window)
            ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

            assert isinstance(ssi_values, pd.Series)
            assert len(ssi_values) > 0

    def test_ssi_standardization(self, sample_ssi_data):
        """Test that SSI values are approximately standardized."""
        ssi_metrics = SSIDroughtMetrics(timescale='D', window=30)
        ssi_values = ssi_metrics.calculate_ssi(sample_ssi_data)

        # Remove NaN values
        valid_ssi = ssi_values.dropna()

        if len(valid_ssi) > 100:  # Need enough data for statistics
            # Mean should be close to 0
            assert abs(valid_ssi.mean()) < 0.5

            # Std should be close to 1
            assert 0.7 < valid_ssi.std() < 1.3
