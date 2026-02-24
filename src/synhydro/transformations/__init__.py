"""
Transformation Pipeline for SynHydro.

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
>>> from synhydro.transformations import LogTransform, StandardScaler, TransformPipeline
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
from synhydro.transformations.abstract import Transform, TransformPipeline

# Concrete transformations
from synhydro.transformations.log import LogTransform
from synhydro.transformations.stedinger import SteddingerTransform
from synhydro.transformations.scaler import StandardScaler
from synhydro.transformations.boxcox import BoxCoxTransform
from synhydro.transformations.deseasonalize import DeseasonalizeTransform

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
