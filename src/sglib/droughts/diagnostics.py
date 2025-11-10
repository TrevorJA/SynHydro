"""
Diagnostic tools for distribution selection in drought analysis.

This module provides statistical tests and comparison methods to help users
select appropriate probability distributions for SSI calculations.
"""

from typing import List, Dict, Union, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.stats import kstest, anderson, chisquare

from sglib.droughts.distributions import DISTRIBUTION_REGISTRY, get_distribution
from sglib.droughts.ssi import SSIDroughtMetrics
from spei._typing import ContinuousDist


def kolmogorov_smirnov_test(
    data: pd.Series,
    dist: Union[str, ContinuousDist],
    significance_level: float = 0.05
) -> Dict[str, float]:
    """
    Perform Kolmogorov-Smirnov goodness-of-fit test.

    Parameters
    ----------
    data : pd.Series
        Observed data
    dist : str or ContinuousDist
        Distribution to test
    significance_level : float, default 0.05
        Significance level for the test

    Returns
    -------
    dict
        Dictionary with 'statistic', 'pvalue', and 'reject_null' keys

    Notes
    -----
    The null hypothesis is that the data follows the specified distribution.
    A low p-value (< significance_level) suggests rejecting the null hypothesis.
    """
    dist_obj = get_distribution(dist)

    # Remove NaN values
    data_clean = data.dropna()

    # Fit distribution to data
    params = dist_obj.fit(data_clean)

    # Perform KS test
    statistic, pvalue = kstest(data_clean, dist_obj.name, args=params)

    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'reject_null': pvalue < significance_level,
        'interpretation': 'reject distribution' if pvalue < significance_level else 'accept distribution'
    }


def anderson_darling_test(
    data: pd.Series,
    dist: Union[str, ContinuousDist] = 'norm'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform Anderson-Darling goodness-of-fit test.

    Parameters
    ----------
    data : pd.Series
        Observed data
    dist : str or ContinuousDist, default 'norm'
        Distribution to test. Note: scipy's anderson test only supports
        'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1'

    Returns
    -------
    dict
        Dictionary with 'statistic', 'critical_values', and 'significance_levels' keys

    Notes
    -----
    The null hypothesis is that the data follows the specified distribution.
    If the statistic is larger than the critical value at a given significance
    level, the null hypothesis is rejected.
    """
    dist_obj = get_distribution(dist)

    # Remove NaN values
    data_clean = data.dropna()

    try:
        result = anderson(data_clean, dist=dist_obj.name)
        return {
            'statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_levels': result.significance_level,
        }
    except ValueError as e:
        warnings.warn(f"Anderson-Darling test not available for {dist_obj.name}: {e}")
        return {
            'statistic': np.nan,
            'critical_values': np.array([]),
            'significance_levels': np.array([]),
            'error': str(e)
        }


def compute_aic_bic(
    data: pd.Series,
    dist: Union[str, ContinuousDist]
) -> Dict[str, float]:
    """
    Compute Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Lower values indicate better fit, penalized by model complexity.

    Parameters
    ----------
    data : pd.Series
        Observed data
    dist : str or ContinuousDist
        Distribution to evaluate

    Returns
    -------
    dict
        Dictionary with 'aic', 'bic', 'log_likelihood', and 'n_params' keys

    Notes
    -----
    AIC = 2k - 2ln(L)
    BIC = k*ln(n) - 2ln(L)
    where k is number of parameters, n is sample size, L is likelihood
    """
    dist_obj = get_distribution(dist)

    # Remove NaN values
    data_clean = data.dropna()
    n = len(data_clean)

    # Fit distribution
    params = dist_obj.fit(data_clean)

    # Compute log-likelihood
    log_likelihood = np.sum(dist_obj.logpdf(data_clean, *params))

    # Number of parameters (includes shape, loc, scale)
    k = len(params)

    # Compute AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return {
        'aic': aic,
        'bic': bic,
        'log_likelihood': log_likelihood,
        'n_params': k,
        'sample_size': n
    }


def compare_distributions(
    data: pd.Series,
    distributions: Optional[List[Union[str, ContinuousDist]]] = None,
    window: int = 12,
    tests: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple distributions for SSI calculation.

    Performs goodness-of-fit tests and computes information criteria to help
    select the most appropriate distribution for the data.

    Parameters
    ----------
    data : pd.Series
        Observed flow data
    distributions : list of str or ContinuousDist, optional
        Distributions to compare. If None, uses common hydrological distributions:
        ['gamma', 'lognorm', 'pearson3', 'weibull', 'norm']
    window : int, default 12
        Rolling window size for aggregation (used to prepare data for SSI)
    tests : list of str, optional
        Tests to perform. Options: 'ks', 'aic', 'bic'
        If None, performs all available tests.

    Returns
    -------
    pd.DataFrame
        Comparison results with distributions as rows and metrics as columns.
        Lower AIC/BIC values indicate better fit.
        Higher KS p-values indicate better fit.

    Examples
    --------
    >>> results = compare_distributions(flow_data, window=12)
    >>> best_dist = results.loc[results['aic'].idxmin(), 'distribution']

    Notes
    -----
    - The data is aggregated using a rolling sum before fitting (as done for SSI)
    - AIC and BIC penalize model complexity, preventing overfitting
    - KS test p-value > 0.05 suggests the distribution is adequate
    """
    if distributions is None:
        # Use common hydrological distributions
        distributions = ['gamma', 'lognorm', 'pearson3', 'weibull', 'norm']

    if tests is None:
        tests = ['ks', 'aic', 'bic']

    # Prepare data (aggregate as done for SSI)
    data_agg = data.rolling(window, min_periods=window).sum().dropna()

    results = []

    for dist in distributions:
        dist_obj = get_distribution(dist)
        dist_name = dist if isinstance(dist, str) else dist_obj.name

        row = {'distribution': dist_name}

        try:
            # Kolmogorov-Smirnov test
            if 'ks' in tests:
                ks_result = kolmogorov_smirnov_test(data_agg, dist)
                row['ks_statistic'] = ks_result['statistic']
                row['ks_pvalue'] = ks_result['pvalue']
                row['ks_pass'] = not ks_result['reject_null']

            # AIC/BIC
            if 'aic' in tests or 'bic' in tests:
                ic_result = compute_aic_bic(data_agg, dist)
                if 'aic' in tests:
                    row['aic'] = ic_result['aic']
                if 'bic' in tests:
                    row['bic'] = ic_result['bic']
                row['log_likelihood'] = ic_result['log_likelihood']
                row['n_params'] = ic_result['n_params']

            row['fit_success'] = True
            row['error'] = None

        except Exception as e:
            # Distribution failed to fit
            row['fit_success'] = False
            row['error'] = str(e)
            warnings.warn(f"Failed to fit {dist_name}: {e}")

        results.append(row)

    df = pd.DataFrame(results)

    # Rank distributions
    if 'aic' in df.columns:
        df['aic_rank'] = df['aic'].rank()
    if 'bic' in df.columns:
        df['bic_rank'] = df['bic'].rank()
    if 'ks_pvalue' in df.columns:
        df['ks_rank'] = df['ks_pvalue'].rank(ascending=False)

    return df


def distribution_summary(
    data: pd.Series,
    dist: Union[str, ContinuousDist],
    window: int = 12
) -> Dict:
    """
    Generate comprehensive summary of distribution fit quality.

    Parameters
    ----------
    data : pd.Series
        Observed flow data
    dist : str or ContinuousDist
        Distribution to evaluate
    window : int, default 12
        Rolling window size for aggregation

    Returns
    -------
    dict
        Comprehensive summary including fitted parameters, goodness-of-fit tests,
        and information criteria

    Examples
    --------
    >>> summary = distribution_summary(flow_data, 'gamma', window=12)
    >>> print(summary['recommendation'])
    """
    dist_obj = get_distribution(dist)
    dist_name = dist if isinstance(dist, str) else dist_obj.name

    # Prepare data
    data_agg = data.rolling(window, min_periods=window).sum().dropna()

    # Fit distribution
    params = dist_obj.fit(data_agg)

    # Perform tests
    ks_result = kolmogorov_smirnov_test(data_agg, dist)
    ic_result = compute_aic_bic(data_agg, dist)

    # Generate recommendation
    if ks_result['pvalue'] > 0.05:
        recommendation = f"{dist_name} appears to be a good fit (KS p-value = {ks_result['pvalue']:.4f})"
    else:
        recommendation = f"{dist_name} may not be appropriate (KS p-value = {ks_result['pvalue']:.4f})"

    return {
        'distribution': dist_name,
        'fitted_params': params,
        'ks_test': ks_result,
        'aic': ic_result['aic'],
        'bic': ic_result['bic'],
        'log_likelihood': ic_result['log_likelihood'],
        'n_params': ic_result['n_params'],
        'sample_size': len(data_agg),
        'recommendation': recommendation
    }
