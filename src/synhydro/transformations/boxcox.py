"""
Box-Cox power transformation for normalizing flow data.

Automatically finds optimal lambda parameter for normalization.
"""
import logging
import pandas as pd

from synhydro.transformations.abstract import Transform

logger = logging.getLogger(__name__)


class BoxCoxTransform(Transform):
    """
    Box-Cox power transformation.

    Automatically finds optimal lambda parameter for normalization.
    Transforms data to approximate normality using power transformation.

    Parameters
    ----------
    by_site : bool, default=True
        Fit separate lambda for each site.

    Examples
    --------
    >>> from synhydro.transformations import BoxCoxTransform
    >>> transform = BoxCoxTransform(by_site=True)
    >>> Q_norm = transform.fit_transform(Q_obs)
    >>> Q_orig = transform.inverse_transform(Q_norm)
    """

    def __init__(self, by_site: bool = True):
        super().__init__()
        self.by_site = by_site

    def fit(self, data: pd.DataFrame) -> 'BoxCoxTransform':
        """
        Fit optimal lambda parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Flow data with sites as columns. Must contain positive values.

        Returns
        -------
        BoxCoxTransform
            Self (for chaining).
        """
        from scipy.stats import boxcox

        lambdas = pd.Series(index=data.columns, dtype=float)

        for col in data.columns:
            # Box-Cox requires positive data
            col_data = data[col].dropna()
            if (col_data <= 0).any():
                logger.warning(f"Column {col} has non-positive values, adding offset")
                col_data = col_data - col_data.min() + 1e-6

            _, lambda_val = boxcox(col_data)
            lambdas[col] = lambda_val

        self.params_['lambda'] = lambdas
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Box-Cox transformation.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Box-Cox transformed data.
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before transform()")

        from scipy.stats import boxcox

        transformed = data.copy()
        lambdas = self.params_['lambda']

        for col in data.columns:
            lambda_val = lambdas[col]
            col_data = data[col]

            # Handle non-positive values
            if (col_data <= 0).any():
                offset = -col_data.min() + 1e-6
                col_data = col_data + offset
                if 'offset' not in self.params_:
                    self.params_['offset'] = {}
                self.params_['offset'][col] = offset

            transformed[col] = boxcox(col_data, lmbda=lambda_val)

        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse Box-Cox transformation.

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

        from scipy.special import inv_boxcox

        result = data.copy()
        lambdas = self.params_['lambda']
        offsets = self.params_.get('offset', {})

        for col in data.columns:
            lambda_val = lambdas[col]
            result[col] = inv_boxcox(data[col], lambda_val)

            # Remove offset if applied
            if col in offsets:
                result[col] = result[col] - offsets[col]

        return result
