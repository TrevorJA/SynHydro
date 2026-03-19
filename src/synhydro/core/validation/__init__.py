"""
Ensemble Validation Framework

Unified metrics for comparing synthetic ensemble properties against
observed streamflow records.

Metric categories
-----------------
- **marginal**: Mean, std, skewness, kurtosis, CV, percentiles, KS test
- **temporal**: Lag-1/2 ACF, Hurst exponent, ACF RMSE
- **spatial**: Cross-site correlation RMSE, max error, bias
- **drought**: Mean/max drought duration and severity, frequency
- **spectral**: Power spectrum RMSE, correlation, low-frequency ratio
- **seasonal**: Per-month mean/std/skewness bias, Wilcoxon p-values
- **annual**: Annual mean, variance, skewness, lag-1 ACF, variance ratio
- **fdc**: Flow duration curve RMSE, bias at Q10/Q50/Q90, envelope coverage
- **lmoments**: L-CV, L-skewness, L-kurtosis ratios (Hosking & Wallis 1997)
- **extremes**: Annual max/min statistics, GEV return-period quantiles, 7Q metrics
- **crps**: Continuous Ranked Probability Score and skill score (Hersbach 2000)
- **ssi_drought**: SSI-based drought duration, severity, frequency (McKee et al. 1993)

References
----------
Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow generation:
1. Model verification and validation. Water Resources Research, 18(4), 909-918.

Srikanthan, R. and McMahon, T.A. (2001). Stochastic generation of annual,
monthly and daily climate data. Stochastic Hydrology and Hydraulics, 15, 369-391.
"""

import logging
from typing import Optional

import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.core.validation._result import ValidationResult
from synhydro.core.validation._helpers import _VALID_METRICS
from synhydro.core.validation._metrics import (
    _compute_marginal_metrics,
    _compute_temporal_metrics,
    _compute_spatial_metrics,
    _compute_drought_metrics,
    _compute_spectral_metrics,
    _compute_seasonal_metrics,
    _compute_annual_metrics,
    _compute_fdc_metrics,
    _compute_lmoment_metrics,
    _compute_extreme_metrics,
    _compute_crps_metrics,
    _compute_ssi_drought_metrics,
    _compute_summary_scores,
)

logger = logging.getLogger(__name__)

__all__ = ["validate_ensemble", "ValidationResult"]


def validate_ensemble(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    drought_threshold: Optional[float] = None,
    hurst_method: str = "rs",
) -> ValidationResult:
    """
    Validate a synthetic ensemble against observed streamflow.

    Computes a comprehensive suite of metrics comparing ensemble properties
    to observed statistics across eight categories.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble to validate.
    Q_obs : pd.DataFrame
        Observed streamflow with DatetimeIndex, sites as columns.
    metrics : list of str, optional
        Subset of metric categories to compute. Options:
        ``'marginal'``, ``'temporal'``, ``'spatial'``, ``'drought'``,
        ``'spectral'``, ``'seasonal'``, ``'annual'``, ``'fdc'``,
        ``'lmoments'``, ``'extremes'``, ``'crps'``, ``'ssi_drought'``.
        If None, all categories are computed.
    drought_threshold : float, optional
        Flow threshold for drought identification. If None, defaults to
        the 20th percentile of observed flows at each site.
    hurst_method : {'rs', 'dfa'}, default 'rs'
        Method for Hurst exponent estimation.

    Returns
    -------
    ValidationResult
        Validation results with per-site and aggregate metrics.

    Raises
    ------
    ValueError
        If no sites are shared between the ensemble and Q_obs, if
        unrecognized metric categories are requested, or if an invalid
        ``hurst_method`` is provided.
    """
    if metrics is None:
        metrics = list(_VALID_METRICS)
    else:
        unknown = set(metrics) - _VALID_METRICS
        if unknown:
            raise ValueError(
                f"Unknown metric categories: {unknown}. "
                f"Valid options: {sorted(_VALID_METRICS)}"
            )

    if hurst_method not in {"rs", "dfa"}:
        raise ValueError(f"hurst_method must be 'rs' or 'dfa', got '{hurst_method}'")

    sites = ensemble.site_names
    obs_sites = [s for s in sites if s in Q_obs.columns]
    if not obs_sites:
        raise ValueError(
            f"No matching sites between ensemble ({sites}) and observed data "
            f"({list(Q_obs.columns)})"
        )

    result = ValidationResult()

    if "marginal" in metrics:
        result.marginal = _compute_marginal_metrics(ensemble, Q_obs, obs_sites)

    if "temporal" in metrics:
        result.temporal = _compute_temporal_metrics(
            ensemble, Q_obs, obs_sites, hurst_method
        )

    if "spatial" in metrics and len(obs_sites) > 1:
        result.spatial = _compute_spatial_metrics(ensemble, Q_obs, obs_sites)

    if "drought" in metrics:
        result.drought = _compute_drought_metrics(
            ensemble, Q_obs, obs_sites, drought_threshold
        )

    if "spectral" in metrics:
        result.spectral = _compute_spectral_metrics(ensemble, Q_obs, obs_sites)

    if "seasonal" in metrics:
        result.seasonal = _compute_seasonal_metrics(ensemble, Q_obs, obs_sites)

    if "annual" in metrics:
        result.annual = _compute_annual_metrics(ensemble, Q_obs, obs_sites)

    if "fdc" in metrics:
        result.fdc = _compute_fdc_metrics(ensemble, Q_obs, obs_sites)

    if "lmoments" in metrics:
        result.lmoments = _compute_lmoment_metrics(ensemble, Q_obs, obs_sites)

    if "extremes" in metrics:
        result.extremes = _compute_extreme_metrics(ensemble, Q_obs, obs_sites)

    if "crps" in metrics:
        result.crps = _compute_crps_metrics(ensemble, Q_obs, obs_sites)

    if "ssi_drought" in metrics:
        result.ssi_drought = _compute_ssi_drought_metrics(ensemble, Q_obs, obs_sites)

    result.summary = _compute_summary_scores(result)

    logger.info(
        "Validation complete: %d sites, %d metric categories",
        len(obs_sites),
        len(metrics),
    )
    return result
