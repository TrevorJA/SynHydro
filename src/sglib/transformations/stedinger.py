"""
Stedinger-Taylor normalization transformation.

Implements the Stedinger and Taylor (1982) transformation with lower bound estimation.
"""
import numpy as np
import pandas as pd

from sglib.transformations.abstract import Transform


class SteddingerTransform(Transform):
    """
    Stedinger normalization (Stedinger and Taylor, 1982).

    Preserves skewness in transformed space by using lower bound estimation.
    Applies log transformation with monthly or global lower-bound adjustment
    to improve normality of streamflow distributions.

    Parameters
    ----------
    by_month : bool, default=True
        Compute separate lower bounds for each month.

    Examples
    --------
    >>> from sglib.transformations import SteddingerTransform
    >>> transform = SteddingerTransform(by_month=True)
    >>> Q_norm = transform.fit_transform(Q_obs)
    >>> Q_orig = transform.inverse_transform(Q_norm)

    References
    ----------
    Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation:
    1. Model verification and validation. Water Resources Research, 18(4), 909-918.
    """

    def __init__(self, by_month: bool = True):
        super().__init__()
        self.by_month = by_month

    def fit(self, data: pd.DataFrame) -> 'SteddingerTransform':
        """
        Fit lower bound parameters.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Flow data with DatetimeIndex and sites as columns (or single Series).

        Returns
        -------
        SteddingerTransform
            Self (for chaining).
        """
        # Convert Series to DataFrame for consistent handling
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if self.by_month:
            # Compute tau for each month and site
            tau = pd.DataFrame(index=range(1, 13), columns=data.columns)
            for month in range(1, 13):
                month_data = data[data.index.month == month]
                tau.loc[month] = self._compute_lower_bound(month_data)
            self.params_['tau'] = tau
        else:
            # Single tau for entire series
            self.params_['tau'] = self._compute_lower_bound(data)

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Stedinger normalization.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Data to transform.

        Returns
        -------
        pd.DataFrame or pd.Series
            Normalized data (same type as input).
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before transform()")

        # Track if input was Series
        was_series = isinstance(data, pd.Series)
        if was_series:
            data = data.to_frame()

        transformed = data.copy()
        tau = self.params_['tau']

        if self.by_month:
            for month in range(1, 13):
                mask = data.index.month == month
                if mask.any():
                    tau_month = tau.loc[month]
                    # Subtract tau and apply log transformation
                    if isinstance(tau_month, pd.Series) and len(tau_month) > 1:
                        # Multiple columns: tau_month is a Series with column names as index
                        for col in data.columns:
                            transformed.loc[mask, col] = np.log(data.loc[mask, col] - tau_month.loc[col])
                    elif isinstance(tau_month, pd.Series) and len(tau_month) == 1:
                        # Single column: tau_month is a Series with one element
                        tau_value = tau_month.iloc[0]
                        for col in data.columns:
                            transformed.loc[mask, col] = np.log(data.loc[mask, col] - tau_value)
                    else:
                        # Scalar case
                        for col in data.columns:
                            transformed.loc[mask, col] = np.log(data.loc[mask, col] - tau_month)
        else:
            transformed = np.log(data - tau)

        # Convert back to Series if input was Series
        if was_series:
            transformed = transformed.iloc[:, 0]

        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse Stedinger normalization.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Normalized data.

        Returns
        -------
        pd.DataFrame or pd.Series
            Original scale data (same type as input).
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before inverse_transform()")

        # Track if input was Series
        was_series = isinstance(data, pd.Series)
        if was_series:
            data = data.to_frame()

        result = data.copy()
        tau = self.params_['tau']

        if self.by_month:
            for month in range(1, 13):
                mask = data.index.month == month
                if mask.any():
                    tau_month = tau.loc[month]
                    # Add tau after exp transformation
                    if isinstance(tau_month, pd.Series) and len(tau_month) > 1:
                        # Multiple columns: tau_month is a Series with column names as index
                        for col in data.columns:
                            result.loc[mask, col] = np.exp(data.loc[mask, col]) + tau_month.loc[col]
                    elif isinstance(tau_month, pd.Series) and len(tau_month) == 1:
                        # Single column: tau_month is a Series with one element
                        tau_value = tau_month.iloc[0]
                        for col in data.columns:
                            result.loc[mask, col] = np.exp(data.loc[mask, col]) + tau_value
                    else:
                        # Scalar case
                        for col in data.columns:
                            result.loc[mask, col] = np.exp(data.loc[mask, col]) + tau_month
        else:
            result = np.exp(data) + tau

        # Convert back to Series if input was Series
        if was_series:
            result = result.iloc[:, 0]

        return result

    @staticmethod
    def _compute_lower_bound(data: pd.DataFrame) -> pd.Series:
        """
        Compute lower bound (tau) for each column.

        Uses formula: tau = (qmax*qmin - qmedian^2) / (qmax + qmin - 2*qmedian)

        Parameters
        ----------
        data : pd.DataFrame
            Flow data.

        Returns
        -------
        pd.Series
            Lower bound parameter for each column.
        """
        qmax = data.max()
        qmin = data.min()
        qmedian = data.median()

        tau = (qmax * qmin - qmedian ** 2) / (qmax + qmin - 2 * qmedian)
        tau = tau.clip(lower=0)  # Ensure non-negative

        return tau
