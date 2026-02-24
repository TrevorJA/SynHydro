"""
Deseasonalization transformation for removing monthly patterns.

Removes seasonal patterns (monthly mean and variance) from flow data.
"""
from typing import Literal
import pandas as pd

from synhydro.transformations.abstract import Transform
from synhydro.core.statistics import compute_monthly_statistics


class DeseasonalizeTransform(Transform):
    """
    Remove seasonal patterns (monthly mean and variance).

    Deseasonalizes data by removing monthly means and optionally normalizing
    by monthly standard deviations.

    Parameters
    ----------
    method : {'mean', 'mean_std'}, default='mean_std'
        - 'mean': Remove monthly means only
        - 'mean_std': Remove means and normalize by monthly std

    Examples
    --------
    >>> from synhydro.transformations import DeseasonalizeTransform
    >>> transform = DeseasonalizeTransform(method='mean_std')
    >>> Q_deseas = transform.fit_transform(Q_obs)
    >>> Q_orig = transform.inverse_transform(Q_deseas)
    """

    def __init__(self, method: Literal['mean', 'mean_std'] = 'mean_std'):
        super().__init__()
        self.method = method

    def fit(self, data: pd.DataFrame) -> 'DeseasonalizeTransform':
        """
        Fit monthly statistics.

        Parameters
        ----------
        data : pd.DataFrame
            Flow data with DatetimeIndex and sites as columns.

        Returns
        -------
        DeseasonalizeTransform
            Self (for chaining).
        """
        # Use centralized compute_monthly_statistics function
        monthly_stats = compute_monthly_statistics(data)
        self.params_['monthly_mean'] = monthly_stats['mean']
        if self.method == 'mean_std':
            self.params_['monthly_std'] = monthly_stats['std']

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deseasonalization.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Deseasonalized data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before transform()")

        transformed = data.copy()

        for month in range(1, 13):
            mask = data.index.month == month
            transformed.loc[mask] = transformed.loc[mask] - self.params_['monthly_mean'].loc[month]

            if self.method == 'mean_std':
                transformed.loc[mask] = transformed.loc[mask] / self.params_['monthly_std'].loc[month]

        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse deseasonalization.

        Parameters
        ----------
        data : pd.DataFrame
            Deseasonalized data.

        Returns
        -------
        pd.DataFrame
            Original scale data with seasonal patterns restored.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before inverse_transform()")

        result = data.copy()

        for month in range(1, 13):
            mask = data.index.month == month

            if self.method == 'mean_std':
                result.loc[mask] = result.loc[mask] * self.params_['monthly_std'].loc[month]

            result.loc[mask] = result.loc[mask] + self.params_['monthly_mean'].loc[month]

        return result
