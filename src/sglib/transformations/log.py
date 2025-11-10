"""
Logarithmic transformation for flow data.

Useful for positively-skewed flow distributions.
"""
import numpy as np
import pandas as pd

from sglib.transformations.abstract import Transform


class LogTransform(Transform):
    """
    Logarithmic transformation.

    Useful for positive-skewed flow distributions.

    Parameters
    ----------
    offset : float, default=0.0
        Offset to add before log transform (to handle zeros).

    Examples
    --------
    >>> from sglib.transformations import LogTransform
    >>> transform = LogTransform(offset=0.01)
    >>> Q_log = transform.fit_transform(Q_obs)
    >>> Q_orig = transform.inverse_transform(Q_log)
    """

    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.offset = offset

    def fit(self, data: pd.DataFrame) -> 'LogTransform':
        """
        Fit transform (no parameters to learn).

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit transformation to.

        Returns
        -------
        LogTransform
            Self (for chaining).
        """
        self.is_fitted = True
        self.params_['offset'] = self.offset
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Log-transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before transform()")
        return np.log(data + self.offset)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse log transformation.

        Parameters
        ----------
        data : pd.DataFrame
            Transformed data.

        Returns
        -------
        pd.DataFrame
            Original scale data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before inverse_transform()")
        return np.exp(data) - self.offset
