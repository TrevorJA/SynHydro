"""
Transformation Pipeline for SGLib.

This module provides reversible data transformations for preprocessing
hydrologic time series data. All transforms follow the fit/transform/
inverse_transform pattern similar to scikit-learn.

Available Transformations
-------------------------
- LogTransform: Logarithmic transformation
- SteddingerTransform: Stedinger-Taylor normalization with lower bound estimation
- StandardScaler: Z-score standardization (zero mean, unit variance)
- BoxCoxTransform: Power transformation for normalization
- DeseasonalizeTransform: Remove monthly seasonal patterns
- TransformPipeline: Chain multiple transforms together

Examples
--------
>>> from sglib.transformations import LogTransform, StandardScaler, TransformPipeline
>>>
>>> # Single transform
>>> transform = LogTransform(offset=0.01)
>>> Q_transformed = transform.fit_transform(Q_obs)
>>> Q_original = transform.inverse_transform(Q_transformed)
>>>
>>> # Pipeline of transforms
>>> pipeline = TransformPipeline([
...     LogTransform(offset=0.01),
...     StandardScaler(by_month=True)
... ])
>>> Q_transformed = pipeline.fit_transform(Q_obs)
>>> Q_original = pipeline.inverse_transform(Q_transformed)
"""

# Abstract base classes
from sglib.transformations.abstract import Transform, TransformPipeline

# Concrete transformations
from sglib.transformations.log import LogTransform
from sglib.transformations.stedinger import SteddingerTransform
from sglib.transformations.scaler import StandardScaler
from sglib.transformations.boxcox import BoxCoxTransform
from sglib.transformations.deseasonalize import DeseasonalizeTransform

__all__ = [
    # Abstract classes
    'Transform',
    'TransformPipeline',
    # Transformations
    'LogTransform',
    'SteddingerTransform',
    'StandardScaler',
    'BoxCoxTransform',
    'DeseasonalizeTransform',
]
