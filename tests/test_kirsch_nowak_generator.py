"""
Tests for Kirsch-Nowak combined generator (monthly generation + daily disaggregation).
"""

import pytest
import numpy as np
import pandas as pd

from sglib.methods.generation.depreciated.kirsch_nowak import KirschNowakGenerator


class TestKirschNowakGeneratorInitialization:
    """Tests for KirschNowakGenerator initialization."""

    def test_initialization_default_params(self, sample_daily_dataframe):
        """Test initialization with default parameters (Q required)."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        assert gen.state.is_preprocessed is False
        assert gen.state.is_fitted is False
        assert gen.n_neighbors == 5
        assert gen.max_month_shift == 7
        assert hasattr(gen, 'Q')

    def test_initialization_custom_params(self, sample_daily_dataframe):
        """Test initialization with custom parameters."""
        gen = KirschNowakGenerator(
            sample_daily_dataframe,
            generate_using_log_flow=True,
            n_neighbors=10,
            max_month_shift=10,
            matrix_repair_method='nearest'
        )
        assert gen.n_neighbors == 10
        assert gen.max_month_shift == 10


class TestKirschNowakGeneratorPreprocessing:
    """Tests for KirschNowakGenerator preprocessing."""

    def test_preprocessing_daily_series(self, sample_daily_series):
        """Test preprocessing with daily Series."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()

        assert gen.state.is_preprocessed is True
        assert hasattr(gen, 'Q')
        assert hasattr(gen, 'Qm')
        assert hasattr(gen, 'nowak_disaggregator')

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Test preprocessing with daily DataFrame."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()

        assert gen.state.is_preprocessed is True
        assert gen.n_sites == 3

    def test_preprocessing_invalid_input(self):
        """Test initialization with invalid input (not DataFrame)."""
        with pytest.raises(TypeError):
            gen = KirschNowakGenerator([1, 2, 3, 4, 5])


class TestKirschNowakGeneratorFit:
    """Tests for KirschNowakGenerator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Test fitting with single site."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        assert gen.state.is_fitted is True
        # Check Kirsch components
        assert hasattr(gen, 'mean_month')
        assert hasattr(gen, 'std_month')
        # Check Nowak components
        assert hasattr(gen.nowak_disaggregator, 'knn_models')
        assert len(gen.nowak_disaggregator.knn_models) == 12

    def test_fit_multiple_sites(self, sample_daily_dataframe):
        """Test fitting with multiple sites."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        assert gen.state.is_fitted is True
        assert gen.nowak_disaggregator.is_multisite is True
        assert gen.nowak_disaggregator.n_sites == 3

    def test_fit_without_preprocessing_raises(self, sample_daily_dataframe):
        """Test that fit without preprocessing raises error."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        with pytest.raises(ValueError, match="not been preprocessed"):
            gen.fit()


class TestKirschNowakGeneratorGenerate:
    """Tests for KirschNowakGenerator generation."""

    def test_generate_single_realization_daily_series(self, sample_daily_series):
        """Test generating single daily realization from Series."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='D'
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.freq == 'D'
        # Q1 2020: Jan (31) + Feb (29, leap year) + Mar (31) = 91 days
        assert len(result) == 91

    def test_generate_multiple_realizations_daily_series(self, sample_daily_series):
        """Test generating multiple daily realizations from Series."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=3,
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='D'
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['realization', 'date']

    def test_generate_single_realization_daily_dataframe(self, sample_daily_dataframe):
        """Test generating single daily realization from DataFrame."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='D'
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3  # 3 sites
        assert len(result) == 91
        assert result.columns.tolist() == sample_daily_dataframe.columns.tolist()

    def test_generate_multiple_realizations_daily_dataframe(self, sample_daily_dataframe):
        """Test generating multiple daily realizations from DataFrame."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=3,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='D'
        )

        assert isinstance(result, dict)
        assert len(result) == 3
        for r in range(3):
            assert r in result
            assert isinstance(result[r], pd.DataFrame)
            assert result[r].shape[1] == 3
            assert len(result[r]) == 366  # 2020 is leap year

    def test_generate_monthly_output(self, sample_daily_series):
        """Test generating monthly output (no disaggregation)."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='MS'
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.freq == 'MS'
        assert len(result) == 12

    def test_generate_preserves_monthly_totals(self, sample_daily_series):
        """Test that daily generation preserves monthly totals."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='D'
        )

        # Aggregate to monthly and check structure
        monthly = result.resample('MS').sum()
        assert len(monthly) == 12
        assert not monthly.isna().any().any()

    def test_generate_leap_year(self, sample_daily_series):
        """Test generation handles leap years correctly."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        # 2020 is a leap year
        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='D'
        )

        assert len(result) == 366

        # Check February has 29 days
        feb_mask = (result.index.month == 2)
        assert feb_mask.sum() == 29

    def test_generate_non_leap_year(self, sample_daily_series):
        """Test generation handles non-leap years correctly."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        # 2019 is not a leap year
        result = gen.generate(
            n_realizations=1,
            start_date='2019-01-01',
            end_date='2019-12-31',
            freq='D'
        )

        assert len(result) == 365

        # Check February has 28 days
        feb_mask = (result.index.month == 2)
        assert feb_mask.sum() == 28

    def test_generate_without_fit_raises(self, sample_daily_series):
        """Test that generate without fit raises error."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()

        with pytest.raises(ValueError, match="not been fitted"):
            gen.generate(n_realizations=1)

    def test_generate_preserves_spatial_correlation(self, sample_daily_dataframe):
        """Test that generation preserves spatial correlation."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2022-12-31',
            freq='D'
        )

        # Calculate correlation matrix
        corr = result.corr()
        # All values should be between -1 and 1
        assert (corr.values >= -1.0).all()
        assert (corr.values <= 1.0).all()
        # Diagonal should be 1
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_generate_with_log_flow(self, sample_daily_series):
        """Test generation with log-transformed flows."""
        gen = KirschNowakGenerator(generate_using_log_flow=True)
        gen.preprocessing(sample_daily_series)
        gen.fit()

        result = gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='D'
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()
        assert (result >= 0).all().all()  # Flows should be non-negative


class TestKirschNowakGeneratorSaveLoad:
    """Tests for KirschNowakGenerator save and load."""

    def test_save_and_load(self, sample_daily_dataframe, tmp_path):
        """Test saving and loading generator."""
        gen = KirschNowakGenerator(n_neighbors=10)
        gen.preprocessing(sample_daily_dataframe)
        gen.fit()

        # Save
        save_path = tmp_path / "kirsch_nowak_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = KirschNowakGenerator.load(str(save_path))

        assert loaded_gen.state.is_preprocessed is True
        assert loaded_gen.state.is_fitted is True
        assert loaded_gen.n_sites == 3
        assert loaded_gen.n_neighbors == 10

        # Generate from loaded generator
        result = loaded_gen.generate(
            n_realizations=1,
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='D'
        )

        assert result.shape == (91, 3)


class TestKirschNowakGeneratorIntegration:
    """Integration tests for KirschNowakGenerator."""

    def test_full_workflow_single_site(self, sample_daily_series):
        """Test complete workflow for single site (Series converted to DataFrame)."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)

        # Preprocessing
        gen.preprocessing()
        assert gen.state.is_preprocessed is True

        # Fit
        gen.fit()
        assert gen.state.is_fitted is True

        # Generate daily
        daily = gen.generate(
            n_realizations=2,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='D'
        )
        assert isinstance(daily, pd.DataFrame)

    def test_full_workflow_multiple_sites(self, sample_daily_dataframe):
        """Test complete workflow for multiple sites."""
        gen = KirschNowakGenerator(n_neighbors=7, max_month_shift=10)

        # Preprocessing
        gen.preprocessing(sample_daily_dataframe)
        assert gen.state.is_preprocessed is True
        assert gen.n_sites == 3

        # Fit
        gen.fit()
        assert gen.state.is_fitted is True

        # Generate daily
        daily = gen.generate(
            n_realizations=3,
            start_date='2020-01-01',
            end_date='2021-12-31',
            freq='D'
        )
        assert isinstance(daily, dict)
        assert len(daily) == 3

        # Check data quality
        for r in range(3):
            df = daily[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()
            assert df.shape[1] == 3

    def test_multiple_generations_are_different(self, sample_daily_series):
        """Test that multiple realizations are different."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(
            n_realizations=5,
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='D'
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

        assert different_count > 0  # At least some pairs should be different
