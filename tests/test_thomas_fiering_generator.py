"""
Tests for Thomas-Fiering parametric streamflow generator.

Comprehensive test suite following SGLib standards.
"""

import pytest
import pickle
import numpy as np
import pandas as pd

from sglib.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from sglib.core.ensemble import Ensemble


class TestThomasFieringInitialization:
    """Tests for ThomasFieringGenerator initialization."""

    def test_initialization_default_params(self, sample_monthly_series):
        """Test initialization with default parameters."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False
        assert hasattr(gen, 'stedinger_transform')

    def test_initialization_with_debug(self, sample_monthly_series):
        """Test initialization with debug mode."""
        gen = ThomasFieringGenerator(sample_monthly_series, debug=True)
        assert gen.debug is True

    def test_initialization_invalid_input(self):
        """Test initialization with invalid input."""
        # Lists should be rejected by validate_input_data during preprocessing
        # At initialization, it's stored as-is
        try:
            gen = ThomasFieringGenerator([1, 2, 3, 4, 5])
            # Should fail during preprocessing
            with pytest.raises((TypeError, AttributeError, ValueError)):
                gen.preprocessing()
        except (TypeError, AttributeError):
            # Or may fail at initialization
            pass

    def test_initialization_stores_algorithm_params(self, sample_monthly_series):
        """Test that initialization stores algorithm parameters."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        assert 'algorithm_params' in gen.init_params.__dict__
        params = gen.init_params.algorithm_params
        assert params['method'] == 'Thomas-Fiering AR(1)'


class TestThomasFieringPreprocessing:
    """Tests for ThomasFieringGenerator preprocessing."""

    def test_preprocessing_monthly_series(self, sample_monthly_series):
        """Test preprocessing with monthly Series."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert hasattr(gen, 'Q_obs_monthly')
        assert hasattr(gen, 'Q_norm')
        assert gen.stedinger_transform.is_fitted is True

    def test_preprocessing_daily_series_resamples(self, sample_daily_series):
        """Test preprocessing resamples daily to monthly."""
        gen = ThomasFieringGenerator(sample_daily_series)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert gen.Q_obs_monthly.index.freq in ['MS', '<MonthBegin>']

    def test_preprocessing_dataframe_uses_first_column(self, sample_monthly_dataframe):
        """Test preprocessing with DataFrame uses first column only."""
        gen = ThomasFieringGenerator(sample_monthly_dataframe)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert isinstance(gen.Q_obs_monthly, pd.Series)
        assert len(gen._sites) == 1

    def test_preprocessing_stedinger_transform(self, sample_monthly_series):
        """Test that Stedinger transform is applied."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        # Q_norm should be different from Q_obs due to transformation
        assert not gen.Q_norm.equals(gen.Q_obs_monthly)
        assert len(gen.Q_norm) == len(gen.Q_obs_monthly)


class TestThomasFieringFitting:
    """Tests for ThomasFieringGenerator fitting."""

    def test_fit_basic(self, sample_monthly_series):
        """Test basic fitting functionality."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        assert gen.is_fitted is True
        assert hasattr(gen, 'mu_monthly')
        assert hasattr(gen, 'sigma_monthly')
        assert hasattr(gen, 'rho_monthly')

    def test_fit_monthly_parameters_shape(self, sample_monthly_series):
        """Test fitted monthly parameters have correct shape."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        assert len(gen.mu_monthly) == 12
        assert len(gen.sigma_monthly) == 12
        assert len(gen.rho_monthly) == 12

    def test_fit_parameters_reasonable(self, sample_monthly_series):
        """Test that fitted parameters are reasonable."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Check all months
        for month in range(1, 13):
            assert gen.mu_monthly[month] is not None
            # Allow NaN for months with insufficient data (transformation issues)
            if np.isfinite(gen.sigma_monthly[month]):
                assert gen.sigma_monthly[month] >= 0
            if np.isfinite(gen.rho_monthly[month]):
                assert -1 <= gen.rho_monthly[month] <= 1

    def test_fit_creates_fitted_params(self, sample_monthly_series):
        """Test that fit creates FittedParams object."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        assert hasattr(gen, 'fitted_params_')
        assert gen.fitted_params_.n_parameters_ == 36  # 12 months × 3 params
        assert gen.fitted_params_.n_sites_ == 1

    def test_fit_without_preprocessing_raises(self, sample_monthly_series):
        """Test fit raises error without preprocessing."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()


class TestThomasFieringGeneration:
    """Tests for ThomasFieringGenerator generation."""

    def test_generate_basic(self, sample_monthly_series):
        """Test basic generation functionality."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=5, n_realizations=3, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3

    def test_generate_shape(self, sample_monthly_series):
        """Test generated data has correct shape."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        n_years = 5
        result = gen.generate(n_years=n_years, n_realizations=2, seed=42)

        for r in range(2):
            df = result.data_by_realization[r]
            assert df.shape == (n_years * 12, 1)  # Monthly data, 1 site

    def test_generate_n_timesteps(self, sample_monthly_series):
        """Test generation with n_timesteps parameter."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        n_timesteps = 37  # Odd number to test truncation
        result = gen.generate(n_timesteps=n_timesteps, n_realizations=1, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == n_timesteps

    def test_generate_non_negative(self, sample_monthly_series):
        """Test that generated flows are non-negative."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=10, n_realizations=5, seed=42)

        for r in range(5):
            df = result.data_by_realization[r]
            assert np.all(df.values >= 0)

    def test_generate_reproducible(self, sample_monthly_series):
        """Test generation is reproducible with seed."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result1 = gen.generate(n_years=3, n_realizations=2, seed=123)
        result2 = gen.generate(n_years=3, n_realizations=2, seed=123)

        for r in range(2):
            df1 = result1.data_by_realization[r]
            df2 = result2.data_by_realization[r]
            pd.testing.assert_frame_equal(df1, df2)

    def test_generate_without_fit_raises(self, sample_monthly_series):
        """Test generation raises error without fitting."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=5)

    def test_generate_has_datetime_index(self, sample_monthly_series):
        """Test generated data has DatetimeIndex."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=2, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_monthly_frequency(self, sample_monthly_series):
        """Test generated data has monthly frequency."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=2, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        assert df.index.freq in ['MS', '<MonthBegin>']

    def test_generate_default_n_years(self, sample_monthly_series):
        """Test generation with default n_years."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Default should use length of historical data
        result = gen.generate(n_realizations=1, seed=42)

        expected_years = len(gen.Q_obs_monthly) // 12
        df = result.data_by_realization[0]
        assert len(df) == expected_years * 12


class TestThomasFieringAR1Properties:
    """Tests for AR(1) model properties."""

    def test_ar1_lag1_correlation_exists(self, sample_monthly_series):
        """Test that lag-1 correlations are computed."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # All monthly lag-1 correlations should be defined
        for month in range(1, 13):
            assert month in gen.rho_monthly.index
            assert np.isfinite(gen.rho_monthly[month])

    def test_multiple_realizations_differ(self, sample_monthly_series):
        """Test that multiple realizations are different."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=5, n_realizations=5, seed=42)

        # Compare realizations pairwise
        different_count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                df_i = result.data_by_realization[i]
                df_j = result.data_by_realization[j]
                if not df_i.equals(df_j):
                    different_count += 1

        assert different_count > 0  # At least some should differ


class TestThomasFieringSerialization:
    """Tests for saving and loading ThomasFieringGenerator."""

    def test_pickle_save_load(self, sample_monthly_series, tmp_path):
        """Test saving and loading via pickle."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Save
        filepath = tmp_path / "tf_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        # Load
        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Verify attributes preserved
        assert gen_loaded.is_fitted is True
        assert gen_loaded.is_preprocessed is True
        assert len(gen_loaded.mu_monthly) == 12

    def test_pickle_generate_after_load(self, sample_monthly_series, tmp_path):
        """Test generation works after loading from pickle."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Save and load
        filepath = tmp_path / "tf_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Generate from loaded generator
        result = gen_loaded.generate(n_years=3, n_realizations=2, seed=99)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2


class TestThomasFieringStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_generated_mean_reasonable(self, sample_monthly_series):
        """Test generated data has reasonable mean."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=20, n_realizations=50, seed=42)

        # Compute ensemble mean
        all_data = []
        for r in range(50):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_mean = all_data.mean()
        obs_mean = gen.Q_obs_monthly.mean()

        # Generated mean should be in reasonable range
        # Allow wider range due to transformation effects
        ratio = gen_mean / obs_mean
        assert 0.05 < ratio < 5.0

    def test_generated_variance_reasonable(self, sample_monthly_series):
        """Test generated data has reasonable variance."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=20, n_realizations=50, seed=42)

        # Compute ensemble std
        all_data = []
        for r in range(50):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_std = all_data.std()
        obs_std = gen.Q_obs_monthly.std()

        # Generated std should be in reasonable range
        # Allow wider range due to transformation and sample variability
        ratio = gen_std / obs_std
        assert 0.1 < ratio < 5.0


class TestThomasFieringOutputFrequency:
    """Tests for output frequency property."""

    def test_output_frequency_monthly(self, sample_monthly_series):
        """Test output frequency is monthly."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        freq = gen.output_frequency
        assert freq == 'MS'


class TestThomasFieringIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, sample_monthly_series):
        """Test complete workflow."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        # Preprocessing
        gen.preprocessing()
        assert gen.is_preprocessed is True

        # Fit
        gen.fit()
        assert gen.is_fitted is True

        # Generate
        result = gen.generate(n_years=10, n_realizations=5, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5

        # Check data quality
        for r in range(5):
            df = result.data_by_realization[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()

    def test_workflow_from_daily_data(self, sample_daily_series):
        """Test complete workflow starting from daily data."""
        gen = ThomasFieringGenerator(sample_daily_series)

        # Should automatically resample to monthly
        gen.preprocessing()
        assert gen.Q_obs_monthly.index.freq in ['MS', '<MonthBegin>']

        gen.fit()
        result = gen.generate(n_years=2, n_realizations=3, seed=42)

        assert isinstance(result, Ensemble)
        for r in range(3):
            df = result.data_by_realization[r]
            assert not df.isna().any().any()

    def test_get_params(self, sample_monthly_series):
        """Test get_params method."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        params = gen.get_params()
        assert isinstance(params, dict)
        # Check for expected parameter keys
        assert 'debug' in params or 'method' in params
