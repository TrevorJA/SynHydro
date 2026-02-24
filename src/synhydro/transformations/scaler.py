"""
Z-score standardization transformation.

Standardizes data to zero mean and unit variance.
"""
import pandas as pd

from synhydro.transformations.abstract import Transform


class StandardScaler(Transform):
    """
    Z-score standardization.

    Standardizes data by removing mean and scaling to unit variance.
    Can be applied globally or separately for each month.

    Parameters
    ----------
    by_month : bool, default=True
        Standardize separately for each month.
    with_mean : bool, default=True
        Center data to zero mean.
    with_std : bool, default=True
        Scale data to unit variance.

    Examples
    --------
    >>> from synhydro.transformations import StandardScaler
    >>> scaler = StandardScaler(by_month=True)
    >>> Q_scaled = scaler.fit_transform(Q_obs)
    >>> Q_orig = scaler.inverse_transform(Q_scaled)
    """

    def __init__(
        self,
        by_month: bool = True,
        with_mean: bool = True,
        with_std: bool = True
    ):
        super().__init__()
        self.by_month = by_month
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, data: pd.DataFrame) -> 'StandardScaler':
        """
        Fit mean and std parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Flow data with DatetimeIndex and sites as columns.

        Returns
        -------
        StandardScaler
            Self (for chaining).
        """
        # Compute mean and std parameters directly
        if self.by_month:
            self.params_['mean'] = data.groupby(data.index.month).mean()
            self.params_['std'] = data.groupby(data.index.month).std()
        else:
            self.params_['mean'] = data.mean()
            self.params_['std'] = data.std()

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standardization.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Standardized data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before transform()")

        # Apply standardization manually to respect with_mean and with_std flags
        transformed = data.copy()

        if self.by_month:
            for month in range(1, 13):
                mask = data.index.month == month
                if self.with_mean:
                    transformed.loc[mask] = transformed.loc[mask] - self.params_['mean'].loc[month]
                if self.with_std:
                    transformed.loc[mask] = transformed.loc[mask] / self.params_['std'].loc[month]
        else:
            if self.with_mean:
                transformed = transformed - self.params_['mean']
            if self.with_std:
                transformed = transformed / self.params_['std']

        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse standardization.

        Parameters
        ----------
        data : pd.DataFrame
            Standardized data.

        Returns
        -------
        pd.DataFrame
            Original scale data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before inverse_transform()")

        # Apply inverse standardization manually to respect with_mean and with_std flags
        result = data.copy()

        if self.by_month:
            for month in range(1, 13):
                mask = data.index.month == month
                if self.with_std:
                    result.loc[mask] = result.loc[mask] * self.params_['std'].loc[month]
                if self.with_mean:
                    result.loc[mask] = result.loc[mask] + self.params_['mean'].loc[month]
        else:
            if self.with_std:
                result = result * self.params_['std']
            if self.with_mean:
                result = result + self.params_['mean']

        return result
