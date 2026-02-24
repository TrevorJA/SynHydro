"""
Tests for Multi-Site Hidden Markov Model (HMM) Generator.

Based on Gold et al. (2025) methodology for multi-site streamflow generation
using Gaussian Mixture Model HMM.
"""

import pytest
import pickle
import numpy as np
import pandas as pd

from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_annual_dataframe():
    """Generate sample annual multi-site DataFrame for HMM testing."""
    dates = pd.date_range(start='2000-01-01', end='2029-12-31', freq='YS')
    np.random.seed(42)
    n_sites = 3
    n_years = len(dates)

    # Generate synthetic annual flows with state-like behavior
    # Alternate between dry and wet states
    data = {}
    for i in range(n_sites):
        base_dry = np.random.gamma(shape=2.0, scale=50.0, size=n_years)
        base_wet = np.random.gamma(shape=3.0, scale=100.0, size=n_years)

        # Create state sequence (0=dry, 1=wet)
        states = (np.arange(n_years) % 3 > 0).astype(int)  # Mostly wet with some dry

        # Blend based on states
        flows = np.where(states == 0, base_dry, base_wet)
        data[f'site_{i+1}'] = flows

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_annual_series():
    """Generate sample annual single-site Series for testing."""
    dates = pd.date_range(start='2000-01-01', end='2029-12-31', freq='YS')
    np.random.seed(42)

    base_dry = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
    base_wet = np.random.gamma(shape=3.0, scale=100.0, size=len(dates))
    states = (np.arange(len(dates)) % 3 > 0).astype(int)
    flows = np.where(states == 0, base_dry, base_wet)

    return pd.Series(flows, index=dates, name='site_1')


class TestMultiSiteHMMInitialization:
    """Tests for MultiSiteHMMGenerator initialization."""

    def test_initialization_default_params(self, sample_annual_dataframe):
        """Test initialization with default parameters."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)

        assert gen.n_states == 2
        assert gen.offset == 1.0
        assert gen.max_iterations == 1000
        assert gen.covariance_type == 'full'
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False

    def test_initialization_custom_params(self, sample_annual_dataframe):
        """Test initialization with custom parameters."""
        gen = MultiSiteHMMGenerator(
            sample_annual_dataframe,
            n_states=3,
            offset=0.5,
            max_iterations=500,
            covariance_type='diag',
            name='test_hmm',
            debug=True
        )

        assert gen.n_states == 3
        assert gen.offset == 0.5
        assert gen.max_iterations == 500
        assert gen.covariance_type == 'diag'
        assert gen.name == 'test_hmm'
        assert gen.debug is True

    def test_initialization_invalid_n_states(self, sample_annual_dataframe):
        """Test initialization with invalid n_states."""
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            MultiSiteHMMGenerator(sample_annual_dataframe, n_states=1)

    def test_initialization_invalid_offset(self, sample_annual_dataframe):
        """Test initialization with invalid offset."""
        with pytest.raises(ValueError, match="offset must be positive"):
            MultiSiteHMMGenerator(sample_annual_dataframe, offset=-1.0)

        with pytest.raises(ValueError, match="offset must be positive"):
            MultiSiteHMMGenerator(sample_annual_dataframe, offset=0.0)

    def test_initialization_invalid_covariance_type(self, sample_annual_dataframe):
        """Test initialization with invalid covariance type."""
        with pytest.raises(ValueError, match="covariance_type must be"):
            MultiSiteHMMGenerator(sample_annual_dataframe, covariance_type='invalid')

    def test_initialization_stores_algorithm_params(self, sample_annual_dataframe):
        """Test that initialization stores algorithm parameters."""
        gen = MultiSiteHMMGenerator(
            sample_annual_dataframe,
            n_states=3,
            offset=2.0
        )

        assert 'algorithm_params' in gen.init_params.__dict__
        params = gen.init_params.algorithm_params
        assert params['n_states'] == 3
        assert params['offset'] == 2.0
        assert params['method'] == 'Multi-Site Hidden Markov Model (Gold et al. 2025)'


class TestMultiSiteHMMPreprocessing:
    """Tests for MultiSiteHMMGenerator preprocessing."""

    def test_preprocessing_multi_site(self, sample_annual_dataframe):
        """Test preprocessing with multi-site DataFrame."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()

        assert gen.is_preprocessed is True
        assert hasattr(gen, 'Q_log_')
        assert hasattr(gen, '_Q_obs')
        assert hasattr(gen, '_sites')
        assert len(gen._sites) == 3
        assert gen.Q_log_.shape == (30, 3)

    def test_preprocessing_site_subset(self, sample_annual_dataframe):
        """Test preprocessing with site subset."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing(sites=['site_1', 'site_2'])

        assert gen.is_preprocessed is True
        assert len(gen._sites) == 2
        assert gen.Q_log_.shape == (30, 2)

    def test_preprocessing_log_transformation(self, sample_annual_dataframe):
        """Test that log transformation is applied correctly."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, offset=1.0)
        gen.preprocessing()

        # Manually compute expected log values
        Q_adj = gen._Q_obs + gen.offset
        Q_log_expected = np.log(Q_adj).values

        np.testing.assert_array_almost_equal(gen.Q_log_, Q_log_expected)

    def test_preprocessing_handles_zeros(self, sample_annual_dataframe):
        """Test preprocessing handles zero values correctly."""
        # Add some zeros
        data = sample_annual_dataframe.copy()
        data.iloc[0, 0] = 0.0
        data.iloc[5, 1] = 0.0

        gen = MultiSiteHMMGenerator(data, offset=1.0)
        gen.preprocessing()

        # Should not have any non-finite values
        assert np.all(np.isfinite(gen.Q_log_))

    def test_preprocessing_invalid_sites(self, sample_annual_dataframe):
        """Test preprocessing with invalid site names."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)

        with pytest.raises(ValueError, match="Sites not found"):
            gen.preprocessing(sites=['invalid_site'])

    def test_preprocessing_single_site_warning(self, sample_annual_series):
        """Test preprocessing warns for single-site data."""
        df = sample_annual_series.to_frame()
        gen = MultiSiteHMMGenerator(df)

        # Should process but warn
        gen.preprocessing()
        assert gen.is_preprocessed is True
        assert len(gen._sites) == 1

    def test_preprocessing_negative_flows(self, sample_annual_dataframe):
        """Test preprocessing rejects negative flows (with small offset)."""
        # Add negative value
        data = sample_annual_dataframe.copy()
        data.iloc[0, 0] = -5.0

        gen = MultiSiteHMMGenerator(data, offset=1.0)

        with pytest.raises(ValueError, match="non-finite values"):
            gen.preprocessing()


class TestMultiSiteHMMFitting:
    """Tests for MultiSiteHMMGenerator fitting."""

    def test_fit_basic(self, sample_annual_dataframe):
        """Test basic fitting functionality."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        assert gen.is_fitted is True
        assert hasattr(gen, 'means_')
        assert hasattr(gen, 'covariances_')
        assert hasattr(gen, 'transition_matrix_')
        assert hasattr(gen, 'stationary_distribution_')

    def test_fit_parameter_shapes(self, sample_annual_dataframe):
        """Test fitted parameter shapes are correct."""
        n_states = 2
        n_sites = 3

        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=n_states)
        gen.preprocessing()
        gen.fit(random_state=42)

        assert gen.means_.shape == (n_states, n_sites)
        assert gen.covariances_.shape == (n_states, n_sites, n_sites)
        assert gen.transition_matrix_.shape == (n_states, n_states)
        assert gen.stationary_distribution_.shape == (n_states,)

    def test_fit_state_ordering(self, sample_annual_dataframe):
        """Test that states are ordered by mean (dry to wet)."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        # State 0 should have lower mean than state 1 for first site
        assert gen.means_[0, 0] < gen.means_[1, 0]

    def test_fit_transition_matrix_valid(self, sample_annual_dataframe):
        """Test transition matrix is valid (rows sum to 1)."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Each row should sum to 1
        row_sums = gen.transition_matrix_.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))

        # All values should be in [0, 1]
        assert np.all(gen.transition_matrix_ >= 0)
        assert np.all(gen.transition_matrix_ <= 1)

    def test_fit_stationary_distribution_valid(self, sample_annual_dataframe):
        """Test stationary distribution is valid (sums to 1)."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Should sum to 1
        assert abs(gen.stationary_distribution_.sum() - 1.0) < 1e-10

        # All values should be in [0, 1]
        assert np.all(gen.stationary_distribution_ >= 0)
        assert np.all(gen.stationary_distribution_ <= 1)

    def test_fit_covariance_matrices_psd(self, sample_annual_dataframe):
        """Test covariance matrices are positive semi-definite."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Check each state's covariance matrix
        for state in range(gen.n_states):
            cov = gen.covariances_[state]
            # All eigenvalues should be non-negative
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_fit_creates_fitted_params(self, sample_annual_dataframe):
        """Test that fit creates FittedParams object."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        assert hasattr(gen, 'fitted_params_')
        assert gen.fitted_params_.n_parameters_ > 0
        assert gen.fitted_params_.sample_size_ == 30
        assert gen.fitted_params_.n_sites_ == 3

    def test_fit_without_preprocessing(self, sample_annual_dataframe):
        """Test fit raises error without preprocessing."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)

        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_fit_reproducible_with_seed(self, sample_annual_dataframe):
        """Test fit is reproducible with same random seed."""
        gen1 = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen1.preprocessing()
        gen1.fit(random_state=42)

        gen2 = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen2.preprocessing()
        gen2.fit(random_state=42)

        np.testing.assert_array_almost_equal(gen1.means_, gen2.means_)
        np.testing.assert_array_almost_equal(
            gen1.transition_matrix_,
            gen2.transition_matrix_
        )

    def test_fit_different_n_states(self, sample_annual_dataframe):
        """Test fitting with different number of states."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=3)
        gen.preprocessing()
        gen.fit(random_state=42)

        assert gen.means_.shape[0] == 3
        assert gen.transition_matrix_.shape == (3, 3)
        assert len(gen.stationary_distribution_) == 3


class TestMultiSiteHMMGeneration:
    """Tests for MultiSiteHMMGenerator generation."""

    def test_generate_basic(self, sample_annual_dataframe):
        """Test basic generation functionality."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=5, n_years=20, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5

    def test_generate_shape(self, sample_annual_dataframe):
        """Test generated data has correct shape."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        n_years = 15
        result = gen.generate(n_realizations=3, n_years=n_years, seed=42)

        for r in range(3):
            df = result.data_by_realization[r]
            assert df.shape == (n_years, 3)  # 3 sites

    def test_generate_n_timesteps(self, sample_annual_dataframe):
        """Test generation with n_timesteps parameter."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=2, n_timesteps=25, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == 25

    def test_generate_non_negative(self, sample_annual_dataframe):
        """Test that generated flows are non-negative."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=10, n_years=20, seed=42)

        for r in range(10):
            df = result.data_by_realization[r]
            assert np.all(df.values >= 0)

    def test_generate_reproducible(self, sample_annual_dataframe):
        """Test generation is reproducible with seed."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result1 = gen.generate(n_realizations=3, n_years=10, seed=123)
        result2 = gen.generate(n_realizations=3, n_years=10, seed=123)

        for r in range(3):
            df1 = result1.data_by_realization[r]
            df2 = result2.data_by_realization[r]
            pd.testing.assert_frame_equal(df1, df2)

    def test_generate_without_fit(self, sample_annual_dataframe):
        """Test generation raises error without fitting."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_realizations=1, n_years=10)

    def test_generate_missing_n_years_and_timesteps(self, sample_annual_dataframe):
        """Test generation raises error without n_years or n_timesteps."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        with pytest.raises(ValueError, match="Must provide either"):
            gen.generate(n_realizations=1)

    def test_generate_has_datetime_index(self, sample_annual_dataframe):
        """Test generated data has DatetimeIndex."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = result.data_by_realization[0]

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_preserves_site_names(self, sample_annual_dataframe):
        """Test generated data preserves site names."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = result.data_by_realization[0]

        assert list(df.columns) == ['site_1', 'site_2', 'site_3']


class TestMultiSiteHMMStationary:
    """Tests for stationary distribution computation."""

    def test_stationary_eigenvector_method(self, sample_annual_dataframe):
        """Test stationary distribution via eigenvector method."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Verify it's the stationary distribution
        # pi * P = pi
        pi = gen.stationary_distribution_
        P = gen.transition_matrix_

        result = pi @ P
        np.testing.assert_array_almost_equal(result, pi, decimal=5)

    def test_stationary_distribution_uniqueness(self, sample_annual_dataframe):
        """Test stationary distribution is consistent."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Recompute manually
        pi = gen._compute_stationary_distribution()

        np.testing.assert_array_almost_equal(
            pi,
            gen.stationary_distribution_,
            decimal=10
        )


class TestMultiSiteHMMStateTrajectory:
    """Tests for state trajectory generation."""

    def test_state_trajectory_length(self, sample_annual_dataframe):
        """Test state trajectory has correct length."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        np.random.seed(42)
        states = gen._generate_state_trajectory(50)

        assert len(states) == 50

    def test_state_trajectory_valid_states(self, sample_annual_dataframe):
        """Test state trajectory contains only valid state indices."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=3)
        gen.preprocessing()
        gen.fit(random_state=42)

        np.random.seed(42)
        states = gen._generate_state_trajectory(100)

        assert all(s in [0, 1, 2] for s in states)

    def test_state_trajectory_uses_transition_matrix(self, sample_annual_dataframe):
        """Test state trajectory respects transition probabilities."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe, n_states=2)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Generate many trajectories
        np.random.seed(42)
        n_trajectories = 1000
        n_steps = 100

        transitions = np.zeros((2, 2))

        for _ in range(n_trajectories):
            states = gen._generate_state_trajectory(n_steps)
            for i in range(len(states) - 1):
                transitions[states[i], states[i+1]] += 1

        # Normalize to probabilities
        transition_probs = transitions / transitions.sum(axis=1, keepdims=True)

        # Should be close to fitted transition matrix
        np.testing.assert_array_almost_equal(
            transition_probs,
            gen.transition_matrix_,
            decimal=1  # Allow some sampling variation
        )


class TestMultiSiteHMMSerialization:
    """Tests for saving and loading MultiSiteHMMGenerator."""

    def test_pickle_save_load(self, sample_annual_dataframe, tmp_path):
        """Test saving and loading via pickle."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Save
        filepath = tmp_path / "hmm_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        # Load
        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Verify attributes preserved
        assert gen_loaded.is_fitted is True
        assert gen_loaded.is_preprocessed is True
        np.testing.assert_array_almost_equal(gen.means_, gen_loaded.means_)
        np.testing.assert_array_almost_equal(
            gen.transition_matrix_,
            gen_loaded.transition_matrix_
        )

    def test_pickle_generate_after_load(self, sample_annual_dataframe, tmp_path):
        """Test generation works after loading from pickle."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Save and load
        filepath = tmp_path / "hmm_generator.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(gen, f)

        with open(filepath, 'rb') as f:
            gen_loaded = pickle.load(f)

        # Generate from loaded generator
        result = gen_loaded.generate(n_realizations=2, n_years=10, seed=99)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2


class TestMultiSiteHMMStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_generated_mean_reasonable(self, sample_annual_dataframe):
        """Test generated data has reasonable mean."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=100, n_years=30, seed=42)

        # Compute ensemble mean
        all_data = []
        for r in range(100):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_mean = all_data.mean(axis=0)
        obs_mean = sample_annual_dataframe.mean(axis=0).values

        # Generated mean should be within reasonable range of observed
        # (not exact due to log transformation and limited sample)
        ratio = gen_mean / obs_mean
        assert np.all((ratio > 0.5) & (ratio < 2.0))

    def test_generated_variance_reasonable(self, sample_annual_dataframe):
        """Test generated data has reasonable variance."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=100, n_years=30, seed=42)

        # Compute ensemble variance
        all_data = []
        for r in range(100):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_std = all_data.std(axis=0)
        obs_std = sample_annual_dataframe.std(axis=0).values

        # Generated std should be in reasonable range
        ratio = gen_std / obs_std
        assert np.all((ratio > 0.3) & (ratio < 3.0))

    def test_spatial_correlation_preserved(self, sample_annual_dataframe):
        """Test spatial correlations are approximately preserved."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()
        gen.fit(random_state=42)

        # Generate large ensemble
        result = gen.generate(n_realizations=200, n_years=30, seed=42)

        # Compute observed correlation
        obs_corr = sample_annual_dataframe.corr().values

        # Compute generated correlation
        all_data = []
        for r in range(200):
            all_data.append(result.data_by_realization[r])
        all_data = pd.concat(all_data, axis=0)
        gen_corr = all_data.corr().values

        # Correlations should be in same ballpark
        # (not exact due to limited sample and log transformation)
        diff = np.abs(obs_corr - gen_corr)
        assert np.all(diff < 0.5)  # Allow substantial difference due to small sample


class TestMultiSiteHMMOutputFrequency:
    """Tests for output frequency property."""

    def test_output_frequency_annual(self, sample_annual_dataframe):
        """Test output frequency matches input (annual)."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)
        gen.preprocessing()

        freq = gen.output_frequency
        # Annual frequencies can have various formats (YS, AS, AS-JAN, etc.)
        assert freq is not None
        assert 'A' in freq or 'Y' in freq  # Should be some form of annual

    def test_output_frequency_before_preprocessing(self, sample_annual_dataframe):
        """Test output frequency returns default before preprocessing."""
        gen = MultiSiteHMMGenerator(sample_annual_dataframe)

        freq = gen.output_frequency
        assert freq == 'YS'  # Default
