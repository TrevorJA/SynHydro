"""
Abstract base classes for data transformations.

Base classes for reversible data transformations.
All transforms follow fit/transform/inverse_transform pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class Transform(ABC):
    """
    Base class for reversible data transformations.

    All transforms implement fit/transform/inverse_transform pattern
    similar to scikit-learn transformers.
    """

    def __init__(self):
        self.is_fitted: bool = False
        self.params_: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'Transform':
        """
        Fit transformation parameters to data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit transformation to.

        Returns
        -------
        Transform
            Self (for chaining).
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transformation.

        Parameters
        ----------
        data : pd.DataFrame
            Transformed data.

        Returns
        -------
        pd.DataFrame
            Original scale data.
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit and transform.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(data).transform(data)


class TransformPipeline:
    """
    Chain multiple transforms together.

    Transforms are applied in order and reversed in reverse order.

    Parameters
    ----------
    transforms : List[Transform]
        List of transform instances to apply in sequence.

    Examples
    --------
    >>> from sglib.transformations import TransformPipeline, LogTransform, StandardScaler
    >>> pipeline = TransformPipeline([
    ...     LogTransform(offset=0.01),
    ...     StandardScaler(by_month=True)
    ... ])
    >>> transformed = pipeline.fit_transform(Q_obs)
    >>> original = pipeline.inverse_transform(transformed)
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def fit(self, data: pd.DataFrame) -> 'TransformPipeline':
        """
        Fit all transforms in sequence.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit all transforms to.

        Returns
        -------
        TransformPipeline
            Self (for chaining).
        """
        current_data = data
        for transform in self.transforms:
            transform.fit(current_data)
            current_data = transform.transform(current_data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transforms in sequence.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data after applying all transforms.
        """
        current_data = data
        for transform in self.transforms:
            current_data = transform.transform(current_data)
        return current_data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transforms in reverse order.

        Parameters
        ----------
        data : pd.DataFrame
            Transformed data.

        Returns
        -------
        pd.DataFrame
            Original scale data.
        """
        current_data = data
        for transform in reversed(self.transforms):
            current_data = transform.inverse_transform(current_data)
        return current_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit and transform.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(data).transform(data)

    def __repr__(self) -> str:
        """
        String representation.

        Returns
        -------
        str
            String showing pipeline of transforms.
        """
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"TransformPipeline({' -> '.join(transform_names)})"
