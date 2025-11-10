"""
Distribution management for drought analysis.

This module provides utilities for working with probability distributions
in drought analysis, including a registry of common distributions and
helper functions for distribution selection.
"""

import scipy.stats as scs
from typing import Dict, List, Union, Optional
from spei._typing import ContinuousDist


# Registry of supported distributions for SSI calculation
DISTRIBUTION_REGISTRY: Dict[str, ContinuousDist] = {
    # Most commonly used distributions for hydrology
    'gamma': scs.gamma,
    'lognorm': scs.lognorm,
    'pearson3': scs.pearson3,

    # Additional distributions that may be useful
    'weibull_min': scs.weibull_min,
    'gumbel_r': scs.gumbel_r,
    'norm': scs.norm,
    'expon': scs.expon,
    'genextreme': scs.genextreme,
    'fisk': scs.fisk,  # Log-logistic distribution
}


# Distribution metadata for user guidance
DISTRIBUTION_INFO: Dict[str, Dict[str, str]] = {
    'gamma': {
        'name': 'Gamma',
        'description': 'Most commonly used for precipitation and streamflow. Right-skewed, bounded at zero.',
        'best_for': 'General streamflow data, especially with positive skew',
        'parameters': '2 (shape, scale)',
    },
    'lognorm': {
        'name': 'Lognormal',
        'description': 'Good for highly skewed positive data. Log-transform of normal distribution.',
        'best_for': 'Highly variable streamflow with strong positive skew',
        'parameters': '2 (shape, scale)',
    },
    'pearson3': {
        'name': 'Pearson Type III',
        'description': 'Flexible distribution commonly used in flood frequency analysis.',
        'best_for': 'Data with moderate to high skewness',
        'parameters': '3 (skew, location, scale)',
    },
    'weibull_min': {
        'name': 'Weibull',
        'description': 'Flexible distribution good for modeling minima and extremes.',
        'best_for': 'Low-flow analysis and drought extremes',
        'parameters': '2 (shape, scale)',
    },
    'gumbel_r': {
        'name': 'Gumbel',
        'description': 'Used for extreme value analysis, particularly maxima.',
        'best_for': 'Extreme events (floods, droughts)',
        'parameters': '2 (location, scale)',
    },
    'norm': {
        'name': 'Normal (Gaussian)',
        'description': 'Symmetric distribution. May not fit hydrologic data well.',
        'best_for': 'Data with little skewness (rare in hydrology)',
        'parameters': '2 (mean, std)',
    },
    'expon': {
        'name': 'Exponential',
        'description': 'Simple distribution with constant hazard rate.',
        'best_for': 'Waiting times, simple right-skewed data',
        'parameters': '1 (scale)',
    },
    'genextreme': {
        'name': 'Generalized Extreme Value (GEV)',
        'description': 'Very flexible for extreme value analysis.',
        'best_for': 'Extreme events with varying tail behavior',
        'parameters': '3 (shape, location, scale)',
    },
    'fisk': {
        'name': 'Log-Logistic (Fisk)',
        'description': 'Alternative to lognormal with heavier tails.',
        'best_for': 'Highly skewed data with occasional extreme values',
        'parameters': '2 (shape, scale)',
    },
}


def get_distribution(name: Union[str, ContinuousDist]) -> ContinuousDist:
    """
    Get a distribution object by name or pass through if already a distribution.

    Parameters
    ----------
    name : str or ContinuousDist
        Distribution name (e.g., 'gamma') or scipy distribution object.

    Returns
    -------
    ContinuousDist
        Scipy continuous distribution object.

    Raises
    ------
    ValueError
        If distribution name is not recognized.

    Examples
    --------
    >>> dist = get_distribution('gamma')
    >>> dist = get_distribution(scs.gamma)  # Pass through
    """
    if isinstance(name, str):
        if name not in DISTRIBUTION_REGISTRY:
            available = ', '.join(DISTRIBUTION_REGISTRY.keys())
            raise ValueError(
                f"Distribution '{name}' not recognized. "
                f"Available distributions: {available}"
            )
        return DISTRIBUTION_REGISTRY[name]
    else:
        # Assume it's already a distribution object
        return name


def list_distributions(include_info: bool = False) -> Union[List[str], Dict[str, Dict[str, str]]]:
    """
    List available distributions for drought analysis.

    Parameters
    ----------
    include_info : bool, default False
        If True, returns detailed information about each distribution.
        If False, returns only distribution names.

    Returns
    -------
    list or dict
        List of distribution names or dict with detailed information.

    Examples
    --------
    >>> distributions = list_distributions()
    >>> print(distributions)
    ['gamma', 'lognorm', 'pearson3', ...]

    >>> info = list_distributions(include_info=True)
    >>> print(info['gamma']['description'])
    """
    if include_info:
        return DISTRIBUTION_INFO.copy()
    else:
        return list(DISTRIBUTION_REGISTRY.keys())


def get_distribution_info(name: str) -> Dict[str, str]:
    """
    Get detailed information about a specific distribution.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    dict
        Dictionary with keys: name, description, best_for, parameters.

    Raises
    ------
    ValueError
        If distribution name is not recognized.

    Examples
    --------
    >>> info = get_distribution_info('gamma')
    >>> print(info['description'])
    """
    if name not in DISTRIBUTION_INFO:
        available = ', '.join(DISTRIBUTION_REGISTRY.keys())
        raise ValueError(
            f"Distribution '{name}' not recognized. "
            f"Available distributions: {available}"
        )
    return DISTRIBUTION_INFO[name].copy()


def print_distribution_guide():
    """
    Print a user-friendly guide to available distributions.

    Examples
    --------
    >>> print_distribution_guide()
    """
    print("=" * 80)
    print("Available Distributions for Drought Analysis".center(80))
    print("=" * 80)
    print()

    for dist_name in DISTRIBUTION_REGISTRY.keys():
        info = DISTRIBUTION_INFO.get(dist_name, {})
        print(f"{info.get('name', dist_name)} ('{dist_name}')")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Best for: {info.get('best_for', 'N/A')}")
        print(f"  Parameters: {info.get('parameters', 'N/A')}")
        print()

    print("=" * 80)
    print("Recommendation: Start with 'gamma' (default) for most streamflow data.")
    print("Use diagnostics tools to test if other distributions fit better.")
    print("=" * 80)


def validate_distribution(dist: Union[str, ContinuousDist]) -> ContinuousDist:
    """
    Validate and normalize a distribution specification.

    Parameters
    ----------
    dist : str or ContinuousDist
        Distribution name or scipy distribution object.

    Returns
    -------
    ContinuousDist
        Validated scipy distribution object.

    Raises
    ------
    ValueError
        If distribution is invalid.
    TypeError
        If distribution type is not supported.

    Examples
    --------
    >>> dist = validate_distribution('gamma')
    >>> dist = validate_distribution(scs.lognorm)
    """
    if isinstance(dist, str):
        return get_distribution(dist)
    elif hasattr(dist, 'pdf') and hasattr(dist, 'cdf'):
        # Looks like a scipy distribution object
        return dist
    else:
        raise TypeError(
            f"Distribution must be a string name or scipy distribution object, "
            f"got {type(dist)}"
        )
