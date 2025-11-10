"""
Statistical Utilities for SGLib

Specialized statistical functions for flow data analysis, correlation computations,
and distribution fitting.

For standardization/normalization, use sglib.transformations.StandardScaler or
sglib.transformations.DeseasonalizeTransform instead.
"""

from typing import Optional, Union, Tuple, Dict, Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import cholesky
import logging

logger = logging.getLogger(__name__)

### Function to compute autocorrelation ###
def compute_autocorrelation(
    data: pd.DataFrame,
    max_lag: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute autocorrelation for each column in DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input flow data with DatetimeIndex.
    max_lag : int, optional
        Maximum lag to compute. If None, defaults to len(data) - 1.

    Returns
    -------
    pd.DataFrame
        DataFrame of autocorrelations with lags as index and columns as sites.
    """
    n = len(data)
    if max_lag is None:
        max_lag = n - 1

    acf_dict = {}
    for col in data.columns:
        series = data[col].dropna()
        mean = series.mean()
        var = series.var()
        acf_values = []
        for lag in range(max_lag + 1):
            if lag >= len(series):
                acf_values.append(np.nan)
                continue
            cov = ((series[:-lag] - mean) * (series[lag:] - mean)).sum() if lag > 0 else ((series - mean) ** 2).sum()
            acf_values.append(cov / (len(series) - lag) / var)
        acf_dict[col] = acf_values

    acf_df = pd.DataFrame(acf_dict, index=np.arange(max_lag + 1))
    acf_df.index.name = 'lag'
    return acf_df

def compute_spatial_correlation(
    data: pd.DataFrame,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson'
) -> pd.DataFrame:
    """
    Compute spatial correlation matrix between columns in DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input flow data with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with sites as both index and columns.
    """
    corr_matrix = data.corr(method=method)
    
    return corr_matrix


def compute_monthly_statistics(
    data: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics by month.

    Parameters
    ----------
    data : pd.DataFrame
        Flow data with DatetimeIndex.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys: 'mean', 'std', 'min', 'max', 'skew', 'cv'
        Each value is DataFrame with months (1-12) as index, sites as columns.
    """
    monthly = data.groupby(data.index.month)

    return {
        'mean': monthly.mean(),
        'std': monthly.std(),
        'min': monthly.min(),
        'max': monthly.max(),
        'skew': monthly.skew(),
        'cv': monthly.std() / monthly.mean()  # Coefficient of variation
    }


def repair_correlation_matrix(
    corr: NDArray[np.float64],
    method: Literal['spectral', 'hypersphere', 'nearest'] = 'spectral'
) -> NDArray[np.float64]:
    """
    Repair non-positive definite correlation matrix.

    Parameters
    ----------
    corr : NDArray
        Correlation matrix to repair.
    method : {'spectral', 'hypersphere', 'nearest'}, default='spectral'
        Method for repairing matrix.

    Returns
    -------
    NDArray
        Repaired positive definite correlation matrix.

    Notes
    -----
    - 'spectral': Eigenvalue truncation (set negative eigenvalues to small positive)
    - 'hypersphere': Project onto space of valid correlation matrices
    - 'nearest': Find nearest positive definite matrix (Higham algorithm)
    """
    if method == 'spectral':
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Truncate negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct matrix
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Rescale to unit diagonal
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)

        return repaired

    elif method == 'hypersphere':
        # Project correlations onto valid range [-1, 1]
        repaired = corr.copy()
        np.fill_diagonal(repaired, 1.0)
        repaired = np.clip(repaired, -1.0, 1.0)

        # Ensure symmetry
        repaired = (repaired + repaired.T) / 2

        # Check if positive definite, if not use spectral method
        try:
            cholesky(repaired, lower=True)
            return repaired
        except np.linalg.LinAlgError:
            logger.warning("Hypersphere method failed, falling back to spectral")
            return repair_correlation_matrix(corr, method='spectral')

    elif method == 'nearest':
        # Higham's algorithm for nearest correlation matrix
        return _nearest_correlation_matrix(corr)

    else:
        raise ValueError(f"Unknown method: {method}")


def _nearest_correlation_matrix(
    A: NDArray[np.float64],
    max_iter: int = 100,
    tol: float = 1e-7
) -> NDArray[np.float64]:
    """
    Find nearest correlation matrix using Higham's algorithm.

    Parameters
    ----------
    A : NDArray
        Input matrix.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    NDArray
        Nearest correlation matrix.
    """
    n = A.shape[0]
    Y = A.copy()
    dS = np.zeros_like(A)

    for _ in range(max_iter):
        R = Y - dS

        # Project onto positive semidefinite cone
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Project onto unit diagonal
        dS = X - R
        np.fill_diagonal(X, 1.0)

        # Check convergence
        if np.linalg.norm(Y - X, ord='fro') < tol:
            return X

        Y = X

    logger.warning(f"Nearest correlation matrix did not converge in {max_iter} iterations")
    return Y
