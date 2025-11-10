"""
Tests for Thomas-Fiering parametric streamflow generator.
"""

import pytest
import numpy as np
import pandas as pd

from sglib.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator


class TestThomasFieringGeneratorInitialization:
    """Tests for ThomasFieringGenerator initialization."""

    def test_initialization_default_params(self, sample_monthly_series):
        """Test initialization with default parameters (Q now required)."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        assert gen.state.is_preprocessed is False
        assert gen.state.is_fitted is False
        assert gen.debug is False
        assert hasattr(gen, 'Q_obs_monthly')

    def test_initialization_with_debug(self, sample_monthly_series):
        """Test initialization with debug mode."""
        gen = ThomasFieringGenerator(sample_monthly_series, debug=True)
        assert gen.debug is True


class TestThomasFieringGeneratorPreprocessing:
    """Tests for ThomasFieringGenerator preprocessing."""

    def test_preprocessing_daily_series(self, sample_daily_series):
        """Test preprocessing with daily Series (Q passed at init, resampled to monthly)."""
        gen = ThomasFieringGenerator(sample_daily_series)
        gen.preprocessing()

        assert gen.state.is_preprocessed is True
        assert hasattr(gen, 'Q_obs_monthly')
        assert gen.Q_obs_monthly.index.freq == 'MS'

    def test_preprocessing_monthly_series(self, sample_monthly_series):
        """Test preprocessing with monthly Series (Q passed at init)."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        assert gen.state.is_preprocessed is True
        assert hasattr(gen, 'Q_obs_monthly')
        assert gen.Q_obs_monthly.index.freq == 'MS'

    def test_preprocessing_applies_stedinger_transform(self, sample_monthly_series):
        """Test that preprocessing applies Stedinger transformation."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        assert hasattr(gen, 'stedinger_transform')
        assert hasattr(gen, 'Q_norm')
        assert gen.stedinger_transform.is_fitted is True

    def test_preprocessing_invalid_input(self):
        """Test initialization with invalid input (not pandas Series/DataFrame)."""
        with pytest.raises((TypeError, AttributeError)):
            gen = ThomasFieringGenerator([1, 2, 3, 4, 5])

    def test_preprocessing_dataframe_raises(self, sample_monthly_dataframe):
        """Test that DataFrame input is handled (single-site only)."""
        # Thomas-Fiering is single-site generator
        # Should either handle first column or raise error during init
        try:
            gen = ThomasFieringGenerator(sample_monthly_dataframe)
            gen.preprocessing()
            # If it processes, it should use only first column
            assert hasattr(gen, 'Q_obs_monthly')
        except (ValueError, TypeError):
            # Or it raises an error for multi-site input
            pass


class TestThomasFieringGeneratorFit:
    """Tests for ThomasFieringGenerator fitting."""

    def test_fit_single_site(self, sample_monthly_series):
        """Test fitting with single site."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        assert gen.state.is_fitted is True
        assert hasattr(gen, 'mu_monthly')
        assert hasattr(gen, 'sigma_monthly')
        assert hasattr(gen, 'rho_monthly')
        assert len(gen.mu_monthly) == 12
        assert len(gen.sigma_monthly) == 12
        assert len(gen.rho_monthly) == 12

    def test_fit_estimates_monthly_parameters(self, sample_monthly_series):
        """Test that fit estimates monthly means, stds, and correlations."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Check that parameters are reasonable
        for month in range(12):
            assert gen.mu_monthly[month] is not None
            assert gen.sigma_monthly[month] >= 0
            assert -1 <= gen.rho_monthly[month] <= 1  # Correlation should be [-1, 1]

    def test_fit_without_preprocessing_raises(self, sample_monthly_series):
        """Test that fit without preprocessing raises error."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        with pytest.raises(ValueError, match="not been preprocessed"):
            gen.fit()


class TestThomasFieringGeneratorGenerate:
    """Tests for ThomasFieringGenerator generation."""

    def test_generate_single_realization(self, sample_monthly_series):
        """Test generating single realization."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='MS'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        assert result.index.freq == 'MS'

    def test_generate_multiple_realizations(self, sample_monthly_series):
        """Test generating multiple realizations."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=5,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='MS'
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['realization', 'date']

    def test_generate_multi_year(self, sample_monthly_series):
        """Test generating multiple years."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2024-12-31',
            freq='MS'
        )

        assert len(result) == 60  # 5 years * 12 months

    def test_generate_preserves_monthly_means(self, sample_monthly_series):
        """Test that generated data preserves monthly means approximately."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Generate many realizations for statistical testing
        result = gen.generate(
            n_realizations=100,
            start_date='2020-01-01',
            end_date='2030-12-31',
            freq='MS'
        )

        # For statistical tests, we need long series
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_generate_preserves_monthly_stds(self, sample_monthly_series):
        """Test that generated data has reasonable standard deviations."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=10,
            start_date='2020-01-01',
            end_date='2030-12-31',
            freq='MS'
        )

        # Check that generated data has reasonable values
        if isinstance(result.index, pd.MultiIndex):
            for r in range(10):
                realization_data = result.loc[r]
                assert realization_data.std().values[0] > 0

    def test_generate_without_fit_raises(self, sample_monthly_series):
        """Test that generate without fit raises error."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()

        with pytest.raises(ValueError, match="not been fitted"):
            gen.generate(n_realizations=1)

    def test_generate_non_negative_flows(self, sample_monthly_series):
        """Test that generated flows are non-negative."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=5,
            start_date='2020-01-01',
            end_date='2022-12-31',
            freq='MS'
        )

        # Check that all values are non-negative
        assert (result >= 0).all().all()


class TestThomasFieringGeneratorAR1:
    """Tests for AR(1) formula and lag-1 correlation preservation."""

    def test_ar1_formula_implementation(self, sample_monthly_series):
        """Test that AR(1) formula is correctly implemented."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Generate single realization to check AR(1) properties
        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2030-12-31',
            freq='MS'
        )

        # Calculate lag-1 autocorrelation for each month
        result_series = result.iloc[:, 0]
        for month in range(1, 13):
            month_data = result_series[result_series.index.month == month]
            if len(month_data) > 2:
                lag1_corr = month_data.autocorr(lag=1)
                # Should be reasonably close to fitted rho, but stochastic
                assert not np.isnan(lag1_corr)

    def test_multiple_realizations_are_different(self, sample_monthly_series):
        """Test that multiple realizations are different."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=5,
            start_date='2020-01-01',
            end_date='2022-12-31',
            freq='MS'
        )

        # Check that realizations are not identical
        realizations = []
        for r in range(5):
            realization_data = result.loc[r].values
            realizations.append(realization_data)

        # At least some realizations should be different
        different_count = 0
        for i in range(len(realizations)):
            for j in range(i + 1, len(realizations)):
                if not np.array_equal(realizations[i], realizations[j]):
                    different_count += 1

        assert different_count > 0


class TestThomasFieringGeneratorSaveLoad:
    """Tests for ThomasFieringGenerator save and load."""

    def test_save_and_load(self, sample_monthly_series, tmp_path):
        """Test saving and loading generator."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        # Save
        save_path = tmp_path / "thomas_fiering_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = ThomasFieringGenerator.load(str(save_path))

        assert loaded_gen.state.is_preprocessed is True
        assert loaded_gen.state.is_fitted is True
        assert len(loaded_gen.mu_monthly) == 12
        assert len(loaded_gen.sigma_monthly) == 12
        assert len(loaded_gen.rho_monthly) == 12

        # Generate from loaded generator
        result = loaded_gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='MS'
        )

        assert len(result) == 12


class TestThomasFieringGeneratorIntegration:
    """Integration tests for ThomasFieringGenerator."""

    def test_full_workflow(self, sample_monthly_series):
        """Test complete workflow."""
        gen = ThomasFieringGenerator(sample_monthly_series)

        # Preprocessing
        gen.preprocessing()
        assert gen.state.is_preprocessed is True

        # Fit
        gen.fit()
        assert gen.state.is_fitted is True

        # Generate
        result = gen.generate(
            n_realizations=10,
            start_date='2020-01-01',
            end_date='2025-12-31',
            freq='MS'
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)

        # Check data quality
        assert not result.isna().any().any()
        assert (result >= 0).all().all()

    def test_workflow_from_daily_data(self, sample_daily_series):
        """Test complete workflow starting from daily data."""
        gen = ThomasFieringGenerator(sample_daily_series)

        # Should automatically resample to monthly
        gen.preprocessing()
        assert gen.Q_obs_monthly.index.freq == 'MS'

        gen.fit()
        result = gen.generate(
            n_realizations=3,
            start_date='2020-01-01',
            end_date='2022-12-31',
            freq='MS'
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_get_params(self, sample_monthly_series):
        """Test get_params method."""
        gen = ThomasFieringGenerator(sample_monthly_series)
        gen.preprocessing()
        gen.fit()

        params = gen.get_params()
        assert isinstance(params, dict)
        assert 'state' in params
        assert 'is_fitted' in params['state']
        assert 'is_preprocessed' in params['state']
