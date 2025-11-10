"""
Tests for sglib.core.base module (Generator base class and GeneratorState).
Updated to match the current API.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import pickle

from sglib.core.base import Generator, GeneratorState


class MockGenerator(Generator):
    """Concrete implementation of Generator for testing."""

    def __init__(self, Q_obs=None, debug=False):
        # Create default data if none provided
        if Q_obs is None:
            index = pd.date_range(start='2020-01-01', periods=100, freq='D')
            Q_obs = pd.DataFrame(
                np.random.randn(100, 2),
                index=index,
                columns=['site_1', 'site_2']
            )
        super().__init__(Q_obs=Q_obs, debug=debug)
        self.preprocessing_called = False
        self.fit_called = False
        self.generate_called = False

    @property
    def output_frequency(self) -> str:
        """Return output frequency for this generator."""
        return 'D'  # Daily output for testing

    def preprocessing(self):
        """Mock preprocessing implementation."""
        Q_validated = self.validate_input_data(self._Q_obs_raw)
        self._Q_obs = Q_validated
        self._sites = Q_validated.columns.tolist()
        self.preprocessing_called = True
        self.update_state(preprocessed=True)

    def _compute_fitted_params(self):
        """Mock implementation of required abstract method."""
        from sglib.core.base import FittedParams
        # Return minimal FittedParams for testing
        return FittedParams()

    def fit(self):
        """Mock fit implementation."""
        self.validate_preprocessing()
        self.fit_called = True
        self.fitted_params_ = self._compute_fitted_params()
        self.update_state(fitted=True)

    def generate(self, n_realizations=1, start_date='2020-01-01', end_date='2020-12-31', freq='D', **kwargs):
        """Mock generate implementation - returns Ensemble."""
        from sglib.core.ensemble import Ensemble

        self.validate_fit()
        self.generate_called = True

        # Create synthetic data
        index = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_timesteps = len(index)

        # Generate multiple realizations as dict
        realization_dict = {}
        for r in range(n_realizations):
            data = np.random.randn(n_timesteps, len(self._sites))
            realization_dict[r] = pd.DataFrame(data, index=index, columns=self._sites)

        # Return as Ensemble
        return Ensemble.from_dict(realization_dict, frequency=self.output_frequency)


class TestGeneratorState:
    """Tests for GeneratorState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = GeneratorState()
        assert state.is_preprocessed is False
        assert state.is_fitted is False
        assert state.preprocessing_params == {}
        assert state.fit_params == {}
        assert state.fit_timestamp is None

    def test_state_modification(self):
        """Test state can be modified."""
        state = GeneratorState()
        state.is_preprocessed = True
        state.preprocessing_params = {'param1': 'value1'}
        state.is_fitted = True
        state.fit_timestamp = datetime.now().isoformat()

        assert state.is_preprocessed is True
        assert state.preprocessing_params == {'param1': 'value1'}
        assert state.is_fitted is True
        assert state.fit_timestamp is not None


class TestGenerator:
    """Tests for Generator abstract base class."""

    def test_instantiation(self):
        """Test that MockGenerator can be instantiated."""
        gen = MockGenerator()
        assert gen.state.is_preprocessed is False
        assert gen.state.is_fitted is False
        assert gen.debug is False

    def test_debug_mode(self):
        """Test debug mode initialization."""
        gen = MockGenerator(debug=True)
        assert gen.debug is True

    def test_validate_input_data_series(self, sample_daily_series):
        """Test input validation with pandas Series."""
        gen = MockGenerator()
        result = gen.validate_input_data(sample_daily_series)
        assert isinstance(result, pd.DataFrame)

    def test_validate_input_data_dataframe(self, sample_daily_dataframe):
        """Test input validation with pandas DataFrame."""
        gen = MockGenerator()
        result = gen.validate_input_data(sample_daily_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_validate_input_data_invalid_type(self):
        """Test input validation with invalid type."""
        gen = MockGenerator()
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])

    def test_validate_input_data_no_datetime_index(self):
        """Test input validation converts non-DatetimeIndex."""
        gen = MockGenerator()
        # New behavior: tries to convert index to DatetimeIndex
        data = pd.Series([1, 2, 3, 4, 5])
        result = gen.validate_input_data(data)
        # Should convert to DatetimeIndex automatically
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_preprocessing_workflow(self, sample_daily_series):
        """Test preprocessing workflow."""
        gen = MockGenerator()
        assert gen.state.is_preprocessed is False

        gen.preprocessing(sample_daily_series)

        assert gen.preprocessing_called is True
        assert gen.state.is_preprocessed is True

    def test_validate_preprocessing_passes_when_preprocessed(self, sample_daily_series):
        """Test validate_preprocessing passes after preprocessing."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.validate_preprocessing()  # Should not raise

    def test_validate_preprocessing_fails_when_not_preprocessed(self):
        """Test validate_preprocessing fails before preprocessing."""
        gen = MockGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.validate_preprocessing()

    def test_fit_workflow(self, sample_daily_series):
        """Test fit workflow."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)

        assert gen.state.is_fitted is False
        gen.fit()

        assert gen.fit_called is True
        assert gen.state.is_fitted is True
        assert gen.state.fit_timestamp is not None

    def test_fit_fails_without_preprocessing(self):
        """Test fit fails without preprocessing."""
        gen = MockGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_validate_fit_passes_when_fitted(self, sample_daily_series):
        """Test validate_fit passes after fitting."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()
        gen.validate_fit()  # Should not raise

    def test_validate_fit_fails_when_not_fitted(self, sample_daily_series):
        """Test validate_fit fails before fitting."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        with pytest.raises(ValueError, match="fit"):
            gen.validate_fit()

    def test_generate_workflow(self, sample_daily_series):
        """Test generate workflow."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()

        result = gen.generate(n_realizations=1)

        assert gen.generate_called is True
        assert isinstance(result, pd.DataFrame)

    def test_generate_fails_without_fit(self, sample_daily_series):
        """Test generate fails without fitting."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        with pytest.raises(ValueError, match="fit"):
            gen.generate()

    def test_n_sites_property_series(self, sample_daily_series):
        """Test n_sites property with Series input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        assert gen.n_sites == 1

    def test_n_sites_property_dataframe(self, sample_daily_dataframe):
        """Test n_sites property with DataFrame input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        assert gen.n_sites == 3

    def test_sites_property_series(self, sample_daily_series):
        """Test sites property with Series input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        expected_name = sample_daily_series.name if sample_daily_series.name else 'flow'
        assert gen.sites == [expected_name]

    def test_sites_property_dataframe(self, sample_daily_dataframe):
        """Test sites property with DataFrame input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        assert gen.sites == sample_daily_dataframe.columns.tolist()

    def test_format_output_single_realization_series(self, sample_daily_series):
        """Test output format with single realization from Series."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()

        result = gen.generate(n_realizations=1, start_date='2020-01-01', end_date='2020-01-31')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 31

    def test_format_output_multiple_realizations_series(self, sample_daily_series):
        """Test output format with multiple realizations from Series."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()

        result = gen.generate(n_realizations=3, start_date='2020-01-01', end_date='2020-01-31')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 31
        # For single site with multiple realizations, columns are r0, r1, r2
        assert result.shape[1] == 3

    def test_format_output_single_realization_dataframe(self, sample_daily_dataframe):
        """Test output format with single realization from DataFrame."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        gen.fit()

        result = gen.generate(n_realizations=1, start_date='2020-01-01', end_date='2020-01-31')

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3
        assert len(result) == 31
        assert result.columns.tolist() == sample_daily_dataframe.columns.tolist()

    def test_format_output_multiple_realizations_dataframe(self, sample_daily_dataframe):
        """Test output format with multiple realizations from DataFrame."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        gen.fit()

        result = gen.generate(n_realizations=2, start_date='2020-01-01', end_date='2020-01-31')

        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result.values())

    def test_get_params(self, sample_daily_series):
        """Test get_params method."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()

        params = gen.get_params()
        assert isinstance(params, dict)

    def test_save_and_load(self, sample_daily_series, tmp_path):
        """Test save and load functionality."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.fit()

        # Save
        save_path = tmp_path / "test_generator.pkl"
        gen.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded_gen = MockGenerator.load(str(save_path))
        assert loaded_gen.state.is_preprocessed is True
        assert loaded_gen.state.is_fitted is True
        assert loaded_gen.preprocessing_called is True
        assert loaded_gen.fit_called is True

    def test_update_state(self, sample_daily_series):
        """Test update_state method."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)

        # Update state
        gen.update_state(fitted=True)

        assert gen.state.is_fitted is True
        assert gen.state.fit_timestamp is not None
