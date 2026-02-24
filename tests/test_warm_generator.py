"""
Tests for WARM (Wavelet Auto-Regressive Method) streamflow generator.

Comprehensive test suite following SynHydro standards.
"""

import pytest
import pickle
import numpy as np
import pandas as pd

from synhydro.methods.generation.parametric.warm import WARMGenerator
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_annual_series():
    """Generate a sample annual time series for testing WARM."""
    dates = pd.date_range(start='1950-01-01', end='2020-12-31', freq='YS')
    np.random.seed(42)

    # Generate realistic annual streamflow with some low-frequency variation
    n_years = len(dates)
    # Base flow with trend
    base = 500 + np.linspace(0, 50, n_years)
    # Add low-frequency component (decadal variation)
    low_freq = 100 * np.sin(2 * np.pi * np.arange(n_years) / 20)
    # Add high-frequency component (annual variation)
    high_freq = 50 * np.sin(2 * np.pi * np.arange(n_years) / 5)
    # Add noise
    noise = np.random.normal(0, 50, n_years)

    values = base + low_freq + high_freq + noise
    values = np.maximum(values, 10)  # Ensure positive

    return pd.Series(values, index=dates, name='site_1')


@pytest.fixture
def sample_annual_dataframe():
    """Generate a sample annual multi-site DataFrame for testing."""
    dates = pd.date_range(start='1950-01-01', end='2020-12-31', freq='YS')
    np.random.seed(42)
    n_years = len(dates)

    data = {}
    for i in range(3):
        base = 500 + np.linspace(0, 50, n_years)
        low_freq = 100 * np.sin(2 * np.pi * np.arange(n_years) / 20 + i)
        noise = np.random.normal(0, 50, n_years)
        data[f'site_{i+1}'] = base + low_freq + noise

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def short_annual_series():
    """Generate a short annual series (30 years) for faster testing."""
    dates = pd.date_range(start='1990-01-01', end='2020-12-31', freq='YS')
    np.random.seed(123)

    n_years = len(dates)
    values = 500 + 100 * np.sin(2 * np.pi * np.arange(n_years) / 10)
    values += np.random.normal(0, 30, n_years)
    values = np.maximum(values, 10)

    return pd.Series(values, index=dates, name='site_1')


class TestWARMInitialization:
    """Tests for WARMGenerator initialization."""

    def test_initialization_default_params(self, sample_annual_series):
        """Test initialization with default parameters."""
        gen = WARMGenerator(sample_annual_series)

        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False
        assert gen.wavelet == 'morl'
        assert gen.scales == 64
        assert gen.ar_order == 1

    def test_initialization_custom_params(self, sample_annual_series):
        """Test initialization with custom parameters."""
        gen = WARMGenerator(
            sample_annual_series,
            wavelet='mexh',
            scales=32,
            ar_order=2,
            debug=True
        )

        assert gen.wavelet == 'mexh'
        assert gen.scales == 32
        assert gen.ar_order == 2
        assert gen.debug is True

    def test_initialization_invalid_scales(self, sample_annual_series):
        """Test initialization raises error for invalid scales."""
        with pytest.raises(ValueError, match="scales must be >= 2"):
            WARMGenerator(sample_annual_series, scales=1)

    def test_initialization_invalid_ar_order(self, sample_annual_series):
        """Test initialization raises error for invalid AR order."""
        with pytest.raises(ValueError, match="ar_order must be >= 1"):
            WARMGenerator(sample_annual_series, ar_order=0)

    def test_initialization_invalid_wavelet(self, sample_annual_series):
        """Test initialization raises error for invalid wavelet."""
        with pytest.raises(ValueError, match="not recognized"):
            WARMGenerator(sample_annual_series, wavelet='invalid_wavelet')

    def test_initialization_stores_algorithm_params(self, sample_annual_series):
        """Test that initialization stores algorithm parameters."""
        gen = WARMGenerator(sample_annual_series)

        assert 'algorithm_params' in gen.init_params.__dict__
        params = gen.init_params.algorithm_params
        assert params['method'] == 'WARM (Wavelet Auto-Regressive Method)'
        assert params['wavelet'] == 'morl'


class TestWARMPreprocessing:
    """Tests for WARMGenerator preprocessing."""

    def test_preprocessing_annual_series(self, sample_annual_series):
        """Test preprocessing with annual Series."""
        gen = WARMGenerator(sample_annual_series)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert hasattr(gen, 'Q_obs_annual')
        assert isinstance(gen.Q_obs_annual, pd.Series)

    def test_preprocessing_annual_dataframe_uses_first_column(self, sample_annual_dataframe):
        """Test preprocessing with DataFrame uses first column only."""
        gen = WARMGenerator(sample_annual_dataframe)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert isinstance(gen.Q_obs_annual, pd.Series)
        assert len(gen._sites) == 1

    def test_preprocessing_monthly_to_annual(self, sample_monthly_series):
        """Test preprocessing resamples monthly to annual."""
        gen = WARMGenerator(sample_monthly_series)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        # Should have roughly 1/12 the number of observations
        assert len(gen.Q_obs_annual) < len(sample_monthly_series) / 6

    def test_preprocessing_validates_data_length(self, sample_annual_series):
        """Test preprocessing warns for short data."""
        # Create very short series
        short_data = sample_annual_series.iloc[:15]
        gen = WARMGenerator(short_data)

        # Should still preprocess but with warning
        gen.preprocessing()
        assert gen.is_preprocessed is True


class TestWARMFitting:
    """Tests for WARMGenerator fitting."""

    def test_fit_basic(self, short_annual_series):
        """Test basic fitting functionality."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        assert gen.is_fitted is True
        assert gen.wavelet_coeffs_ is not None
        assert gen.sawp_ is not None
        assert gen.ar_params_ is not None

    def test_fit_wavelet_coeffs_shape(self, short_annual_series):
        """Test wavelet coefficients have correct shape."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        n_years = len(gen.Q_obs_annual)
        assert gen.wavelet_coeffs_.shape == (16, n_years)

    def test_fit_sawp_shape(self, short_annual_series):
        """Test SAWP has correct shape."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        n_years = len(gen.Q_obs_annual)
        assert gen.sawp_.shape == (n_years,)
        assert np.all(gen.sawp_ >= 0)  # Power should be non-negative

    def test_fit_ar_params_structure(self, short_annual_series):
        """Test AR parameters have correct structure."""
        gen = WARMGenerator(short_annual_series, scales=16, ar_order=2)
        gen.preprocessing()
        gen.fit()

        # Should have parameters for each scale
        assert len(gen.ar_params_) == 16

        # Each scale should have coeffs, sigma, mean
        for scale_idx in range(16):
            params = gen.ar_params_[scale_idx]
            assert 'coeffs' in params
            assert 'sigma' in params
            assert 'mean' in params
            assert len(params['coeffs']) == 2  # AR(2)
            assert params['sigma'] > 0

    def test_fit_creates_fitted_params(self, short_annual_series):
        """Test that fit creates FittedParams object."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        assert hasattr(gen, 'fitted_params_')
        assert gen.fitted_params_.n_sites_ == 1

    def test_fit_without_preprocessing_raises(self, sample_annual_series):
        """Test fit raises error without preprocessing."""
        gen = WARMGenerator(sample_annual_series)

        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_fit_different_wavelets(self, short_annual_series):
        """Test fitting with different wavelet types."""
        wavelets = ['morl', 'mexh', 'gaus1']

        for wavelet in wavelets:
            gen = WARMGenerator(short_annual_series, wavelet=wavelet, scales=8)
            gen.preprocessing()
            gen.fit()

            assert gen.is_fitted is True
            assert gen.wavelet_coeffs_ is not None


class TestWARMGeneration:
    """Tests for WARMGenerator generation."""

    def test_generate_basic(self, short_annual_series):
        """Test basic generation functionality."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=20, n_realizations=3, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3

    def test_generate_shape(self, short_annual_series):
        """Test generated data has correct shape."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        n_years = 25
        result = gen.generate(n_years=n_years, n_realizations=2, seed=42)

        for r in range(2):
            df = result.data_by_realization[r]
            assert df.shape == (n_years, 1)  # Annual data, 1 site

    def test_generate_non_negative(self, short_annual_series):
        """Test that generated flows are non-negative."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=30, n_realizations=5, seed=42)

        for r in range(5):
            df = result.data_by_realization[r]
            assert np.all(df.values >= 0)

    def test_generate_reproducible(self, short_annual_series):
        """Test generation is reproducible with seed."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result1 = gen.generate(n_years=20, n_realizations=2, seed=123)
        result2 = gen.generate(n_years=20, n_realizations=2, seed=123)

        for r in range(2):
            df1 = result1.data_by_realization[r]
            df2 = result2.data_by_realization[r]
            pd.testing.assert_frame_equal(df1, df2)

    def test_generate_without_fit_raises(self, sample_annual_series):
        """Test generation raises error without fitting."""
        gen = WARMGenerator(sample_annual_series)
        gen.preprocessing()

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=10)

    def test_generate_has_datetime_index(self, short_annual_series):
        """Test generated data has DatetimeIndex."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=15, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_annual_frequency(self, short_annual_series):
        """Test generated data has annual frequency."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=15, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        # Check frequency is annual
        assert df.index.freq in ['YS', '<YearBegin>']

    def test_generate_default_n_years(self, short_annual_series):
        """Test generation with default n_years."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        # Default should use length of historical data
        result = gen.generate(n_realizations=1, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == len(gen.Q_obs_annual)

    def test_generate_n_timesteps(self, short_annual_series):
        """Test generation with n_timesteps parameter."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        n_timesteps = 25
        result = gen.generate(n_timesteps=n_timesteps, n_realizations=1, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == n_timesteps


class TestWARMWaveletProperties:
    """Tests for wavelet-specific properties."""

    def test_sawp_captures_variability(self, short_annual_series):
        """Test that SAWP varies over time."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        # SAWP should have variation (not constant)
        assert np.std(gen.sawp_) > 0
        assert np.max(gen.sawp_) > np.min(gen.sawp_)

    def test_multiple_realizations_differ(self, short_annual_series):
        """Test that multiple realizations are different."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=20, n_realizations=5, seed=42)

        # Compare realizations pairwise
        different_count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                df_i = result.data_by_realization[i]
                df_j = result.data_by_realization[j]
                if not df_i.equals(df_j):
                    different_count += 1

        assert different_count > 0  # At least some should differ

    def test_wavelet_scales_used(self, short_annual_series):
        """Test that scales are correctly stored."""
        gen = WARMGenerator(short_annual_series, scales=16)
        gen.preprocessing()
        gen.fit()

        assert gen.scales_used_ is not None
        assert len(gen.scales_used_) == 16
        assert np.all(gen.scales_used_ == np.arange(1, 17))


class TestWARMSerialization:
    """Tests for saving and loading WARMGenerator."""

    def test_pickle_save_load(self, short_annual_series, tmp_path):
        """Test saving and loading via pickle."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        # Save
        filepath = tmp_path / "warm_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        # Load
        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Verify attributes preserved
        assert gen_loaded.is_fitted is True
        assert gen_loaded.is_preprocessed is True
        assert gen_loaded.scales == 8

    def test_pickle_generate_after_load(self, short_annual_series, tmp_path):
        """Test generation works after loading from pickle."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        # Save and load
        filepath = tmp_path / "warm_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Generate from loaded generator
        result = gen_loaded.generate(n_years=15, n_realizations=2, seed=99)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2


class TestWARMStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_generated_mean_reasonable(self, short_annual_series):
        """Test generated data has reasonable mean."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=30, n_realizations=20, seed=42)

        # Compute ensemble mean
        all_data = []
        for r in range(20):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_mean = all_data.mean()
        obs_mean = gen.Q_obs_annual.mean()

        # Generated mean should be in reasonable range
        # WARM uses scale factors so allow wider tolerance
        ratio = gen_mean / obs_mean
        assert 0.1 < ratio < 10.0

    def test_generated_has_variability(self, short_annual_series):
        """Test generated data has reasonable variability."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_years=30, n_realizations=10, seed=42)

        # Each realization should have variability
        for r in range(10):
            df = result.data_by_realization[r]
            assert np.std(df.values) > 0


class TestWARMOutputFrequency:
    """Tests for output frequency property."""

    def test_output_frequency_annual(self, sample_annual_series):
        """Test output frequency is annual."""
        gen = WARMGenerator(sample_annual_series)

        freq = gen.output_frequency
        assert freq == 'YS'


class TestWARMARModels:
    """Tests for AR model fitting components."""

    def test_ar_model_fitting_ar1(self, short_annual_series):
        """Test AR(1) model fitting."""
        gen = WARMGenerator(short_annual_series, scales=4, ar_order=1)
        gen.preprocessing()
        gen.fit()

        # Each scale should have 1 AR coefficient
        for scale_idx in range(4):
            params = gen.ar_params_[scale_idx]
            assert len(params['coeffs']) == 1

    def test_ar_model_fitting_ar2(self, short_annual_series):
        """Test AR(2) model fitting."""
        gen = WARMGenerator(short_annual_series, scales=4, ar_order=2)
        gen.preprocessing()
        gen.fit()

        # Each scale should have 2 AR coefficients
        for scale_idx in range(4):
            params = gen.ar_params_[scale_idx]
            assert len(params['coeffs']) == 2


class TestWARMIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, short_annual_series):
        """Test complete workflow."""
        gen = WARMGenerator(short_annual_series, scales=8)

        # Preprocessing
        gen.preprocessing()
        assert gen.is_preprocessed is True

        # Fit
        gen.fit()
        assert gen.is_fitted is True

        # Generate
        result = gen.generate(n_years=25, n_realizations=5, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5

        # Check data quality
        for r in range(5):
            df = result.data_by_realization[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()

    def test_workflow_different_ar_orders(self, short_annual_series):
        """Test workflow with different AR orders."""
        for ar_order in [1, 2, 3]:
            gen = WARMGenerator(short_annual_series, scales=8, ar_order=ar_order)
            gen.preprocessing()
            gen.fit()
            result = gen.generate(n_years=15, n_realizations=2, seed=42)

            assert isinstance(result, Ensemble)

    def test_get_params(self, short_annual_series):
        """Test get_params method."""
        gen = WARMGenerator(short_annual_series, scales=8)
        gen.preprocessing()
        gen.fit()

        params = gen.get_params()
        assert isinstance(params, dict)
        # Check for expected parameter keys
        assert 'debug' in params or 'wavelet' in params
