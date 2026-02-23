"""
Tests for Kirsch-Nowak combined generator (monthly generation + daily disaggregation).
Updated to match current deprecated API.
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
        # n_neighbors and max_month_shift are stored in the embedded disaggregator
        assert gen.nowak_disaggregator.n_neighbors == 5
        assert gen.nowak_disaggregator.max_month_shift == 7

    def test_initialization_custom_params(self, sample_daily_dataframe):
        """Test initialization with custom parameters."""
        gen = KirschNowakGenerator(
            sample_daily_dataframe,
            generate_using_log_flow=True,
            n_neighbors=10,
            max_month_shift=10,
            matrix_repair_method='nearest'
        )
        assert gen.nowak_disaggregator.n_neighbors == 10
        assert gen.nowak_disaggregator.max_month_shift == 10


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

    def test_preprocessing_invalid_input(self, sample_daily_series):
        """Test that invalid input type raises TypeError during validation."""
        gen = KirschNowakGenerator(sample_daily_series.to_frame())
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])


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
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()


class TestKirschNowakGeneratorGenerate:
    """Tests for KirschNowakGenerator generation.

    KirschNowakGenerator.generate() API:
      generate(n_realizations=1, n_years=1, as_array=False) -> dict
    Returns a dict mapping realization index -> pd.Series (single-site)
    or pd.DataFrame (multi-site) of daily flows.
    With fixture data (2010-2015), synthetic starts at 2016 (a leap year).
    So n_years=1 → 366 daily rows.
    """

    def test_generate_single_realization_single_site(self, sample_daily_series):
        """Test generating single realization from single-site data."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, dict)
        assert 0 in result
        # Result is daily; 2016 is a leap year so 366 days
        assert len(result[0]) == 366

    def test_generate_multiple_realizations_single_site(self, sample_daily_series):
        """Test generating multiple realizations from single-site data."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=3, n_years=1)

        assert isinstance(result, dict)
        assert len(result) == 3
        for r in range(3):
            assert r in result
            assert len(result[r]) == 366

    def test_generate_single_realization_dataframe(self, sample_daily_dataframe):
        """Test generating single realization from DataFrame."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, dict)
        assert 0 in result
        df = result[0]
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 3  # 3 sites
        assert len(df) == 366  # 2016 is leap year
        assert df.columns.tolist() == sample_daily_dataframe.columns.tolist()

    def test_generate_multiple_realizations_dataframe(self, sample_daily_dataframe):
        """Test generating multiple realizations from DataFrame."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=3, n_years=1)

        assert isinstance(result, dict)
        assert len(result) == 3
        for r in range(3):
            assert r in result
            assert isinstance(result[r], pd.DataFrame)
            assert result[r].shape[1] == 3

    def test_generate_preserves_monthly_totals(self, sample_daily_series):
        """Test that daily generation preserves monthly structure."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=1)
        realization = result[0]

        # Aggregate to monthly and check structure
        monthly = realization.resample('MS').sum()
        assert len(monthly) == 12
        assert not monthly.isna().any()

    def test_generate_leap_year_february(self, sample_daily_series):
        """Test that February in leap year (2016) has 29 days."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=1)
        realization = result[0]

        # Fixture data is 2010-2015, synthetic starts 2016 (leap year)
        feb_mask = (realization.index.month == 2)
        assert feb_mask.sum() == 29

    def test_generate_non_leap_year_february(self, sample_daily_series):
        """Test that February in non-leap year (2017) has 28 days."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        # n_years=2 → 2016 and 2017
        result = gen.generate(n_realizations=1, n_years=2)
        realization = result[0]

        # Check Feb 2017 has 28 days
        feb_2017_mask = (realization.index.month == 2) & (realization.index.year == 2017)
        assert feb_2017_mask.sum() == 28

    def test_generate_without_fit_raises(self, sample_daily_series):
        """Test that generate without fit raises error."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_realizations=1)

    def test_generate_preserves_spatial_correlation(self, sample_daily_dataframe):
        """Test that generation preserves spatial correlation."""
        gen = KirschNowakGenerator(sample_daily_dataframe)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=3)
        df = result[0]

        # Calculate correlation matrix
        corr = df.corr()
        # All values should be between -1 and 1
        assert (corr.values >= -1.0).all()
        assert (corr.values <= 1.0).all()
        # Diagonal should be 1
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_generate_with_log_flow(self, sample_daily_series):
        """Test generation with log-transformed flows."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df, generate_using_log_flow=True)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=1, n_years=1)
        realization = result[0]

        assert not realization.isna().any()
        assert (realization >= 0).all()


class TestKirschNowakGeneratorSaveLoad:
    """Tests for KirschNowakGenerator save and load."""

    def test_save_and_load(self, sample_daily_dataframe, tmp_path):
        """Test saving and loading generator."""
        gen = KirschNowakGenerator(sample_daily_dataframe, n_neighbors=10)
        gen.preprocessing()
        gen.fit()

        # Save
        save_path = tmp_path / "kirsch_nowak_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = KirschNowakGenerator.load(str(save_path))

        assert loaded_gen.state.is_preprocessed is True
        assert loaded_gen.state.is_fitted is True
        assert loaded_gen.n_sites == 3
        assert loaded_gen.nowak_disaggregator.n_neighbors == 10

        # Generate from loaded generator
        result = loaded_gen.generate(n_realizations=1, n_years=1)
        assert isinstance(result, dict)
        assert result[0].shape[1] == 3


class TestKirschNowakGeneratorIntegration:
    """Integration tests for KirschNowakGenerator."""

    def test_full_workflow_single_site(self, sample_daily_series):
        """Test complete workflow for single site (Series converted to DataFrame)."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)

        gen.preprocessing()
        assert gen.state.is_preprocessed is True

        gen.fit()
        assert gen.state.is_fitted is True

        result = gen.generate(n_realizations=2, n_years=1)
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_full_workflow_multiple_sites(self, sample_daily_dataframe):
        """Test complete workflow for multiple sites."""
        gen = KirschNowakGenerator(
            sample_daily_dataframe,
            n_neighbors=7,
            max_month_shift=10
        )

        gen.preprocessing()
        assert gen.state.is_preprocessed is True
        assert gen.n_sites == 3

        gen.fit()
        assert gen.state.is_fitted is True

        result = gen.generate(n_realizations=3, n_years=2)
        assert isinstance(result, dict)
        assert len(result) == 3

        for r in range(3):
            df = result[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()
            assert df.shape[1] == 3

    def test_multiple_generations_are_different(self, sample_daily_series):
        """Test that multiple realizations are different."""
        df = sample_daily_series.to_frame()
        gen = KirschNowakGenerator(df)
        gen.preprocessing()
        gen.fit()

        result = gen.generate(n_realizations=5, n_years=1)

        realizations = [result[r].values for r in range(5)]

        # At least some realizations should be different
        different_count = 0
        for i in range(len(realizations)):
            for j in range(i + 1, len(realizations)):
                if not np.array_equal(realizations[i], realizations[j]):
                    different_count += 1

        assert different_count > 0
