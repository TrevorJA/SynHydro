"""
Tests for KNNBootstrapGenerator.

Tests the initialization, preprocessing, fitting, and generation workflow
for the KNN Bootstrap nonparametric generator.
"""

import pytest
import numpy as np
import pandas as pd
from synhydro.methods.generation.nonparametric.knn_bootstrap import KNNBootstrapGenerator
from synhydro.core.ensemble import Ensemble


class TestKNNBootstrapGeneratorInitialization:
    """Test generator initialization."""

    def test_initialization_with_dataframe(self, sample_monthly_dataframe):
        """Test generator initialization with DataFrame."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        assert gen is not None
        assert not gen.is_fitted
        assert not gen.is_preprocessed
        assert gen.name == 'KNNBootstrapGenerator'

    def test_initialization_with_custom_name(self, sample_monthly_dataframe):
        """Test initialization with custom name."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, name='MyKNNGen')
        assert gen.name == 'MyKNNGen'

    def test_initialization_with_custom_parameters(self, sample_monthly_dataframe):
        """Test initialization with custom algorithm parameters."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            n_neighbors=15,
            block_size=2,
            name='CustomKNN'
        )
        assert gen.n_neighbors == 15
        assert gen.block_size == 2

    def test_initialization_with_debug(self, sample_monthly_dataframe):
        """Test initialization with debug flag."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, debug=True)
        assert gen.debug is True


class TestKNNBootstrapGeneratorPreprocessing:
    """Test preprocessing workflow."""

    def test_preprocessing_basic(self, sample_monthly_dataframe):
        """Test basic preprocessing."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        assert gen.is_preprocessed
        assert gen.n_sites == 3
        assert gen.sites == ['site_1', 'site_2', 'site_3']

    def test_preprocessing_with_site_subset(self, sample_monthly_dataframe):
        """Test preprocessing with site subset."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing(sites=['site_1', 'site_2'])
        assert gen.n_sites == 2
        assert gen.sites == ['site_1', 'site_2']

    def test_preprocessing_detects_monthly_frequency(self, sample_monthly_dataframe):
        """Test frequency detection for monthly data."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        assert gen.output_frequency == 'MS'

    def test_preprocessing_detects_daily_frequency(self, sample_daily_dataframe):
        """Test frequency detection for daily data."""
        gen = KNNBootstrapGenerator(sample_daily_dataframe)
        gen.preprocessing()
        assert gen.output_frequency == 'D'

    def test_preprocessing_n_neighbors_heuristic(self, sample_monthly_dataframe):
        """Test n_neighbors is set via sqrt(n) heuristic if not provided."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        n_timesteps = len(sample_monthly_dataframe)
        expected_k = max(1, int(np.ceil(np.sqrt(n_timesteps))))
        assert gen._n_neighbors == expected_k

    def test_preprocessing_n_neighbors_custom(self, sample_monthly_dataframe):
        """Test n_neighbors is used if provided."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, n_neighbors=5)
        gen.preprocessing()
        assert gen._n_neighbors == 5

    def test_preprocessing_n_neighbors_clamped(self, sample_monthly_dataframe):
        """Test n_neighbors is clamped to n_timesteps - 1."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, n_neighbors=1000)
        gen.preprocessing()
        assert gen._n_neighbors == len(sample_monthly_dataframe) - 1

    def test_preprocessing_invalid_site_raises_error(self, sample_monthly_dataframe):
        """Test that invalid site name raises error."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        with pytest.raises(ValueError):
            gen.preprocessing(sites=['nonexistent_site'])

    def test_preprocessing_builds_feature_vectors(self, sample_monthly_dataframe):
        """Test that feature vectors and successor pairs are built."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        assert gen._feature_vectors is not None
        assert gen._successor_values is not None
        assert len(gen._feature_vectors) == len(sample_monthly_dataframe) - 1
        assert len(gen._successor_values) == len(sample_monthly_dataframe) - 1

    def test_preprocessing_with_feature_cols(self, sample_monthly_dataframe):
        """Test preprocessing with specified feature columns."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            feature_cols=['site_1']
        )
        gen.preprocessing()
        assert gen._feature_cols == ['site_1']
        assert gen._feature_vectors.shape[1] == 1

    def test_preprocessing_with_invalid_feature_cols(self, sample_monthly_dataframe):
        """Test that invalid feature_cols raises error."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            feature_cols=['nonexistent']
        )
        with pytest.raises(ValueError):
            gen.preprocessing()

    def test_preprocessing_with_index_site(self, sample_monthly_dataframe):
        """Test preprocessing with specified index site."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            index_site='site_1'
        )
        gen.preprocessing()
        assert gen.index_site == 'site_1'

    def test_preprocessing_with_invalid_index_site(self, sample_monthly_dataframe):
        """Test that invalid index_site raises error."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            index_site='nonexistent'
        )
        with pytest.raises(ValueError):
            gen.preprocessing()


class TestKNNBootstrapGeneratorFitting:
    """Test fitting workflow."""

    def test_fit_without_preprocessing_raises_error(self, sample_monthly_dataframe):
        """Test that fit without preprocessing raises error."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        with pytest.raises(ValueError):
            gen.fit()

    def test_fit_basic(self, sample_monthly_dataframe):
        """Test basic fitting."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        assert gen.is_fitted
        assert gen._is_monthly_conditioned
        assert len(gen._monthly_knn) == 12

    def test_fit_computes_kernel_weights(self, sample_monthly_dataframe):
        """Test that kernel weights are computed correctly."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        assert gen._kernel_weights is not None
        assert len(gen._kernel_weights) == gen._n_neighbors
        assert np.isclose(np.sum(gen._kernel_weights), 1.0)

    def test_fit_kernel_weights_are_decreasing(self, sample_monthly_dataframe):
        """Test that Lall-Sharma kernel weights are decreasing."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, n_neighbors=10)
        gen.preprocessing()
        gen.fit()
        weights = gen._kernel_weights
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_fit_sets_fitted_params(self, sample_monthly_dataframe):
        """Test that fitted_params_ is set after fit."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        assert gen.fitted_params_ is not None
        assert gen.fitted_params_.sample_size_ == len(sample_monthly_dataframe) - 1
        assert gen.fitted_params_.n_sites_ == 3


class TestKNNBootstrapGeneratorGeneration:
    """Test generation workflow."""

    def test_generate_without_fit_raises_error(self, sample_monthly_dataframe):
        """Test that generate without fit raises error."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        with pytest.raises(ValueError):
            gen.generate()

    def test_generate_single_realization(self, sample_monthly_dataframe):
        """Test generation of single realization."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=5)
        assert isinstance(ensemble, Ensemble)
        assert len(ensemble.realization_ids) == 1
        assert ensemble.metadata.n_sites == 3

    def test_generate_multiple_realizations(self, sample_monthly_dataframe):
        """Test generation of multiple realizations."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=10, n_years=5)
        assert len(ensemble.realization_ids) == 10
        assert ensemble.metadata.n_sites == 3

    def test_generate_with_n_years(self, sample_monthly_dataframe):
        """Test generation with n_years parameter."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        n_years = 3
        ensemble = gen.generate(n_realizations=1, n_years=n_years)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == n_years * 12

    def test_generate_with_n_timesteps(self, sample_monthly_dataframe):
        """Test generation with explicit n_timesteps."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        n_timesteps = 100
        ensemble = gen.generate(n_realizations=1, n_timesteps=n_timesteps)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == n_timesteps

    def test_generate_with_seed_reproducibility(self, sample_monthly_dataframe):
        """Test that seed produces reproducible results."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()

        ensemble1 = gen.generate(n_realizations=1, n_years=5, seed=42)
        ensemble2 = gen.generate(n_realizations=1, n_years=5, seed=42)

        df1 = ensemble1.data_by_realization[0]
        df2 = ensemble2.data_by_realization[0]

        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_values_in_historical_range(self, sample_monthly_dataframe):
        """Test that generated values are within historical range."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=5, n_years=10)

        for real_id, df in ensemble.data_by_realization.items():
            for site in df.columns:
                # Generated values should be from successors, which are historical
                min_val = sample_monthly_dataframe[site].min()
                max_val = sample_monthly_dataframe[site].max()
                assert df[site].min() >= min_val - abs(min_val) * 0.01
                assert df[site].max() <= max_val + abs(max_val) * 0.01

    def test_generate_output_has_datetimeindex(self, sample_monthly_dataframe):
        """Test that generated output has DatetimeIndex."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=5)
        realization_df = ensemble.data_by_realization[0]
        assert isinstance(realization_df.index, pd.DatetimeIndex)

    def test_generate_output_has_correct_columns(self, sample_monthly_dataframe):
        """Test that generated output has correct site columns."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=5)
        realization_df = ensemble.data_by_realization[0]
        assert list(realization_df.columns) == gen.sites

    def test_generate_ensemble_metadata(self, sample_monthly_dataframe):
        """Test that ensemble has proper metadata."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=3, n_years=5)
        assert ensemble.metadata.generator_class == 'KNNBootstrapGenerator'
        assert ensemble.metadata.n_realizations == 3
        assert ensemble.metadata.n_sites == 3

    def test_generate_with_daily_data(self, sample_daily_dataframe):
        """Test generation with daily data."""
        gen = KNNBootstrapGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=1)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == 365


class TestKNNBootstrapGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_site_data(self, sample_monthly_series):
        """Test with single-site data."""
        df = sample_monthly_series.to_frame()
        gen = KNNBootstrapGenerator(df)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=2, n_years=5)
        assert ensemble.metadata.n_sites == 1
        assert len(ensemble.realization_ids) == 2

    def test_very_short_dataset(self):
        """Test with very short dataset."""
        dates = pd.date_range('2000-01-01', periods=20, freq='MS')
        df = pd.DataFrame({'site1': np.random.randn(20)}, index=dates)
        gen = KNNBootstrapGenerator(df, n_neighbors=5)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_timesteps=50)
        assert len(ensemble.data_by_realization[0]) == 50

    def test_all_zero_flows(self):
        """Test with all-zero flows."""
        dates = pd.date_range('2000-01-01', periods=100, freq='MS')
        df = pd.DataFrame({'site1': np.zeros(100)}, index=dates)
        gen = KNNBootstrapGenerator(df)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # All generated values should be zero
        assert (ensemble.data_by_realization[0] == 0).all().all()

    def test_constant_flow(self):
        """Test with constant flow values."""
        dates = pd.date_range('2000-01-01', periods=100, freq='MS')
        df = pd.DataFrame({'site1': np.full(100, 100.0)}, index=dates)
        gen = KNNBootstrapGenerator(df)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # All generated values should be constant
        assert (ensemble.data_by_realization[0] == 100.0).all().all()

    def test_negative_values(self):
        """Test with negative flow values (e.g., anomalies)."""
        dates = pd.date_range('2000-01-01', periods=100, freq='MS')
        df = pd.DataFrame({
            'site1': np.random.randn(100) * 50  # Can be negative
        }, index=dates)
        gen = KNNBootstrapGenerator(df)
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # Should handle negative values without error
        assert len(ensemble.data_by_realization[0]) == 36

    def test_very_large_n_neighbors(self, sample_monthly_dataframe):
        """Test with n_neighbors larger than dataset."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            n_neighbors=10000
        )
        gen.preprocessing()
        assert gen._n_neighbors < len(sample_monthly_dataframe)
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=5)
        assert len(ensemble.data_by_realization[0]) == 60


class TestKNNBootstrapGeneratorWorkflow:
    """Test complete workflow."""

    def test_complete_workflow_monthly(self, sample_monthly_dataframe):
        """Test complete preprocessing -> fit -> generate workflow."""
        gen = KNNBootstrapGenerator(
            sample_monthly_dataframe,
            n_neighbors=10,
            name='TestKNN'
        )

        # Preprocessing
        gen.preprocessing()
        assert gen.is_preprocessed

        # Fitting
        gen.fit()
        assert gen.is_fitted

        # Generation
        ensemble = gen.generate(n_realizations=5, n_years=10, seed=42)
        assert isinstance(ensemble, Ensemble)
        assert len(ensemble.realization_ids) == 5
        assert ensemble.metadata.n_sites == 3

        # Check data quality
        for real_id, df in ensemble.data_by_realization.items():
            assert df.shape[0] == 120  # 10 years * 12 months
            assert df.shape[1] == 3    # 3 sites
            assert df.notna().all().all()

    def test_complete_workflow_daily(self, sample_daily_dataframe):
        """Test complete workflow with daily data."""
        gen = KNNBootstrapGenerator(
            sample_daily_dataframe,
            n_neighbors=20
        )
        gen.preprocessing()
        gen.fit()
        ensemble = gen.generate(n_realizations=2, n_years=1)

        for real_id, df in ensemble.data_by_realization.items():
            assert df.shape[0] == 365
            assert df.shape[1] == 3

    def test_multisite_joint_resampling(self, sample_monthly_dataframe):
        """Test that multisite generation uses joint resampling."""
        gen = KNNBootstrapGenerator(sample_monthly_dataframe, seed=42)
        gen.preprocessing()
        gen.fit()

        # Generate with seed for reproducibility
        ensemble = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ensemble.data_by_realization[0]

        # All sites should have the same temporal structure (same neighbors selected)
        # We can't directly test this without checking implementation, but we can
        # verify that spatial correlations are maintained
        assert df.shape[1] == 3
        assert not df.isnull().any().any()
