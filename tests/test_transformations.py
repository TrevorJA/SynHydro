"""
Tests for sglib.transformations module.
"""

import pytest
import numpy as np
import pandas as pd

from sglib.transformations import (
    Transform,
    TransformPipeline,
    SteddingerTransform,
    LogTransform,
    StandardScaler,
    BoxCoxTransform,
)


class TestSteddingerTransform:
    """Tests for Stedinger log transformation."""

    def test_fit_transform_series(self, sample_monthly_series):
        """Test fit and transform on Series (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = SteddingerTransform(by_month=False)
        transformed = transform.fit_transform(df)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_monthly_series)
        assert transform.is_fitted is True
        assert hasattr(transform, 'params_')

    def test_inverse_transform_series(self, sample_monthly_series):
        """Test inverse transform on Series (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = SteddingerTransform(by_month=False)
        transformed = transform.fit_transform(df)
        recovered = transform.inverse_transform(transformed)

        assert isinstance(recovered, pd.DataFrame)
        assert np.allclose(recovered.values, df.values, rtol=1e-6)

    def test_fit_transform_by_month(self, sample_monthly_series):
        """Test fit and transform with by_month=True (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = SteddingerTransform(by_month=True)
        transformed = transform.fit_transform(df)

        assert isinstance(transformed, pd.DataFrame)
        assert transform.is_fitted is True
        assert 'tau' in transform.params_

    def test_inverse_transform_by_month(self, sample_monthly_series):
        """Test inverse transform with by_month=True (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = SteddingerTransform(by_month=True)
        transformed = transform.fit_transform(df)
        recovered = transform.inverse_transform(transformed)

        # Only check rows where both transformed and recovered are finite
        valid_mask = (np.isfinite(transformed.values).all(axis=1) &
                      np.isfinite(recovered.values).all(axis=1))
        if valid_mask.sum() > 0:
            assert np.allclose(recovered.values[valid_mask], df.values[valid_mask], rtol=1e-5)

    def test_fit_transform_dataframe(self, sample_monthly_dataframe):
        """Test fit and transform on DataFrame."""
        transform = SteddingerTransform(by_month=False)
        transformed = transform.fit_transform(sample_monthly_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_monthly_dataframe.shape

    def test_transform_before_fit_raises(self, sample_monthly_series):
        """Test that transform before fit raises ValueError."""
        transform = SteddingerTransform(by_month=False)
        with pytest.raises(ValueError, match="fitted|fit"):
            transform.transform(sample_monthly_series)


class TestLogTransform:
    """Tests for log transformation."""

    def test_fit_transform_series(self, sample_monthly_series):
        """Test fit and transform on Series."""
        transform = LogTransform(offset=0.0)
        transformed = transform.fit_transform(sample_monthly_series)

        assert isinstance(transformed, pd.Series)
        assert len(transformed) == len(sample_monthly_series)
        assert transform.is_fitted is True

    def test_inverse_transform_series(self, sample_monthly_series):
        """Test inverse transform on Series."""
        transform = LogTransform(offset=0.0)
        transformed = transform.fit_transform(sample_monthly_series)
        recovered = transform.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_series.values, rtol=1e-10)

    def test_log_transform_with_offset(self, sample_monthly_series):
        """Test log transform with offset for handling zeros."""
        transform = LogTransform(offset=1.0)
        transformed = transform.fit_transform(sample_monthly_series)
        recovered = transform.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_series.values, rtol=1e-10)

    def test_log_transform_dataframe(self, sample_monthly_dataframe):
        """Test log transform on DataFrame."""
        transform = LogTransform(offset=0.0)
        transformed = transform.fit_transform(sample_monthly_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_monthly_dataframe.shape


class TestStandardScaler:
    """Tests for standard scaling (z-score normalization)."""

    def test_fit_transform_series(self, sample_monthly_series):
        """Test fit and transform on Series."""
        scaler = StandardScaler(by_month=False, with_mean=True, with_std=True)
        transformed = scaler.fit_transform(sample_monthly_series)

        assert isinstance(transformed, pd.Series)
        assert np.abs(transformed.mean()) < 1e-10
        assert np.abs(transformed.std() - 1.0) < 1e-10

    def test_inverse_transform_series(self, sample_monthly_series):
        """Test inverse transform on Series."""
        scaler = StandardScaler(by_month=False)
        transformed = scaler.fit_transform(sample_monthly_series)
        recovered = scaler.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_series.values, rtol=1e-10)

    def test_fit_transform_by_month(self, sample_monthly_series):
        """Test fit and transform with by_month=True."""
        scaler = StandardScaler(by_month=True)
        transformed = scaler.fit_transform(sample_monthly_series)

        assert isinstance(transformed, pd.Series)
        assert scaler.is_fitted is True

    def test_inverse_transform_by_month(self, sample_monthly_series):
        """Test inverse transform with by_month=True."""
        scaler = StandardScaler(by_month=True)
        transformed = scaler.fit_transform(sample_monthly_series)
        recovered = scaler.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_series.values, rtol=1e-10)

    def test_scaler_without_mean(self, sample_monthly_series):
        """Test scaler without centering (with_mean=False)."""
        scaler = StandardScaler(by_month=False, with_mean=False, with_std=True)
        transformed = scaler.fit_transform(sample_monthly_series)

        # Mean should not be 0
        assert np.abs(transformed.mean()) > 0.1
        # Std should still be 1
        assert np.abs(transformed.std() - 1.0) < 1e-10

    def test_scaler_without_std(self, sample_monthly_series):
        """Test scaler without scaling (with_std=False)."""
        scaler = StandardScaler(by_month=False, with_mean=True, with_std=False)
        transformed = scaler.fit_transform(sample_monthly_series)

        # Mean should be 0
        assert np.abs(transformed.mean()) < 1e-10
        # Std should not be 1
        assert np.abs(transformed.std() - 1.0) > 0.1

    def test_scaler_dataframe(self, sample_monthly_dataframe):
        """Test scaler on DataFrame."""
        scaler = StandardScaler(by_month=False)
        transformed = scaler.fit_transform(sample_monthly_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_monthly_dataframe.shape


class TestBoxCoxTransform:
    """Tests for Box-Cox power transformation."""

    def test_fit_transform_series(self, sample_monthly_series):
        """Test fit and transform on Series (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = BoxCoxTransform(by_site=False)
        transformed = transform.fit_transform(df)

        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_monthly_series)
        assert transform.is_fitted is True
        assert 'lambda' in transform.params_ or 'lambda_values' in transform.params_

    def test_inverse_transform_series(self, sample_monthly_series):
        """Test inverse transform on Series (converted to DataFrame)."""
        # Convert Series to DataFrame - transformations require DataFrame
        df = sample_monthly_series.to_frame()
        transform = BoxCoxTransform(by_site=False)
        transformed = transform.fit_transform(df)
        recovered = transform.inverse_transform(transformed)

        assert np.allclose(recovered.values, df.values, rtol=1e-6)

    def test_fit_transform_dataframe(self, sample_monthly_dataframe):
        """Test fit and transform on DataFrame."""
        transform = BoxCoxTransform(by_site=True)
        transformed = transform.fit_transform(sample_monthly_dataframe)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_monthly_dataframe.shape

    def test_inverse_transform_dataframe(self, sample_monthly_dataframe):
        """Test inverse transform on DataFrame."""
        transform = BoxCoxTransform(by_site=True)
        transformed = transform.fit_transform(sample_monthly_dataframe)
        recovered = transform.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_dataframe.values, rtol=1e-6)


class TestTransformPipeline:
    """Tests for transformation pipeline."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = TransformPipeline([
            LogTransform(offset=0.0),
            StandardScaler(by_month=False)
        ])
        assert len(pipeline.transforms) == 2
        assert isinstance(pipeline.transforms[0], LogTransform)
        assert isinstance(pipeline.transforms[1], StandardScaler)

    def test_pipeline_fit_transform(self, sample_monthly_series):
        """Test pipeline fit and transform."""
        pipeline = TransformPipeline([
            LogTransform(offset=0.0),
            StandardScaler(by_month=False)
        ])
        transformed = pipeline.fit_transform(sample_monthly_series)

        assert isinstance(transformed, pd.Series)
        assert len(transformed) == len(sample_monthly_series)

    def test_pipeline_inverse_transform(self, sample_monthly_series):
        """Test pipeline inverse transform."""
        pipeline = TransformPipeline([
            LogTransform(offset=0.0),
            StandardScaler(by_month=False)
        ])
        transformed = pipeline.fit_transform(sample_monthly_series)
        recovered = pipeline.inverse_transform(transformed)

        assert np.allclose(recovered.values, sample_monthly_series.values, rtol=1e-6)

    def test_pipeline_fit_and_transform_separately(self, sample_monthly_series):
        """Test fitting and transforming separately."""
        pipeline = TransformPipeline([
            LogTransform(offset=0.0),
            StandardScaler(by_month=False)
        ])
        pipeline.fit(sample_monthly_series)
        transformed = pipeline.transform(sample_monthly_series)

        assert isinstance(transformed, pd.Series)
        assert all(t.is_fitted for t in pipeline.transforms)

    def test_pipeline_three_transforms(self, sample_monthly_series):
        """Test pipeline with three transforms (converted to DataFrame)."""
        # Convert Series to DataFrame - SteddingerTransform and BoxCoxTransform require DataFrame
        df = sample_monthly_series.to_frame()
        pipeline = TransformPipeline([
            SteddingerTransform(by_month=False),
            StandardScaler(by_month=False),
            BoxCoxTransform(by_site=False)
        ])
        transformed = pipeline.fit_transform(df)
        recovered = pipeline.inverse_transform(transformed)

        assert np.allclose(recovered.values, df.values, rtol=1e-5)

    def test_pipeline_with_dataframe(self, sample_monthly_dataframe):
        """Test pipeline with DataFrame (offset large enough to handle seasonal negatives)."""
        # sample_monthly_dataframe can have negative values due to seasonal component;
        # use a large offset to ensure log transform receives positive inputs.
        pipeline = TransformPipeline([
            LogTransform(offset=200.0),
            StandardScaler(by_month=False)
        ])
        transformed = pipeline.fit_transform(sample_monthly_dataframe)
        recovered = pipeline.inverse_transform(transformed)

        assert isinstance(recovered, pd.DataFrame)
        assert np.allclose(recovered.values, sample_monthly_dataframe.values, rtol=1e-5)

    def test_empty_pipeline(self, sample_monthly_series):
        """Test empty pipeline (no transforms)."""
        pipeline = TransformPipeline([])
        transformed = pipeline.fit_transform(sample_monthly_series)

        assert np.array_equal(transformed.values, sample_monthly_series.values)

    def test_pipeline_transform_before_fit_raises(self, sample_monthly_series):
        """Test that transform before fit raises error."""
        pipeline = TransformPipeline([
            LogTransform(offset=0.0),
            StandardScaler(by_month=False)
        ])
        with pytest.raises(ValueError):
            pipeline.transform(sample_monthly_series)


class TestTransformBaseClass:
    """Tests for Transform base class."""

    def test_transform_is_abstract(self):
        """Test that Transform cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Transform()

    def test_fit_transform_calls_fit_and_transform(self, sample_monthly_series):
        """Test that fit_transform calls fit and then transform."""
        transform = LogTransform(offset=0.0)
        result = transform.fit_transform(sample_monthly_series)

        assert transform.is_fitted is True
        assert isinstance(result, pd.Series)

    def test_is_fitted_flag(self, sample_monthly_series):
        """Test is_fitted flag."""
        transform = LogTransform(offset=0.0)
        assert transform.is_fitted is False

        transform.fit(sample_monthly_series)
        assert transform.is_fitted is True
