"""
Ensemble Validation Framework

Unified metrics for comparing synthetic ensemble properties against
observed streamflow records. Computes marginal, temporal, spatial,
drought, and spectral statistics to assess generation fidelity.

References
----------
Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow generation:
1. Model verification and validation. Water Resources Research, 18(4), 909-918.
"""

from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

from synhydro.core.ensemble import Ensemble
from synhydro.core.statistics import (
    compute_autocorrelation,
    compute_hurst_exponent,
    compute_power_spectral_density,
    compare_spectral_properties,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Container for ensemble validation results.

    Attributes
    ----------
    marginal : Dict[str, Dict]
        Per-site marginal statistics (mean, std, skew, min, max, L-moments).
    temporal : Dict[str, Dict]
        Per-site temporal statistics (lag-1 ACF, Hurst exponent).
    spatial : Dict[str, float]
        Cross-site correlation metrics.
    drought : Dict[str, Dict]
        Drought duration and severity statistics.
    spectral : Dict[str, Dict]
        Per-site spectral comparison metrics.
    summary : Dict[str, float]
        Aggregate summary scores.
    """
    marginal: Dict[str, Dict] = field(default_factory=dict)
    temporal: Dict[str, Dict] = field(default_factory=dict)
    spatial: Dict[str, float] = field(default_factory=dict)
    drought: Dict[str, Dict] = field(default_factory=dict)
    spectral: Dict[str, Dict] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten results into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: category, metric, site, observed, synthetic_median,
            synthetic_p10, synthetic_p90, relative_error.
        """
        rows = []
        for category, site_metrics in [
            ("marginal", self.marginal),
            ("temporal", self.temporal),
            ("spectral", self.spectral),
        ]:
            for site, metrics in site_metrics.items():
                for metric_name, values in metrics.items():
                    if isinstance(values, dict) and "observed" in values:
                        rows.append({
                            "category": category,
                            "metric": metric_name,
                            "site": site,
                            "observed": values["observed"],
                            "synthetic_median": values.get("synthetic_median"),
                            "synthetic_p10": values.get("synthetic_p10"),
                            "synthetic_p90": values.get("synthetic_p90"),
                            "relative_error": values.get("relative_error"),
                        })
        return pd.DataFrame(rows)


def validate_ensemble(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    drought_threshold: Optional[float] = None,
    hurst_method: str = "rs",
) -> ValidationResult:
    """
    Validate a synthetic ensemble against observed streamflow.

    Computes a comprehensive suite of metrics comparing ensemble properties
    to observed statistics. Results include per-site marginal moments,
    temporal dependence, spatial correlations, drought characteristics, and
    spectral fidelity.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble to validate.
    Q_obs : pd.DataFrame
        Observed streamflow with DatetimeIndex, sites as columns.
    metrics : list of str, optional
        Subset of metric categories to compute. Options:
        'marginal', 'temporal', 'spatial', 'drought', 'spectral'.
        If None, computes all.
    drought_threshold : float, optional
        Threshold for drought analysis (flow below this value).
        If None, defaults to 20th percentile of observed flows.
    hurst_method : {'rs', 'dfa'}, default='rs'
        Method for Hurst exponent estimation.

    Returns
    -------
    ValidationResult
        Validation results with per-site and aggregate metrics.
    """
    if metrics is None:
        metrics = ["marginal", "temporal", "spatial", "drought", "spectral"]

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

    result.summary = _compute_summary_scores(result)

    logger.info(
        f"Validation complete: {len(obs_sites)} sites, "
        f"{len(metrics)} metric categories"
    )
    return result


def _compute_marginal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: List[str],
) -> Dict[str, Dict]:
    """Compute marginal (distributional) metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values

        obs_stats = {
            "mean": float(np.mean(obs)),
            "std": float(np.std(obs, ddof=1)),
            "skewness": float(_skewness(obs)),
            "cv": float(np.std(obs, ddof=1) / np.mean(obs)) if np.mean(obs) != 0 else np.nan,
            "min": float(np.min(obs)),
            "max": float(np.max(obs)),
            "p10": float(np.percentile(obs, 10)),
            "p50": float(np.percentile(obs, 50)),
            "p90": float(np.percentile(obs, 90)),
        }

        syn_stats = {k: [] for k in obs_stats}

        for real_id, df in ensemble.data_by_realization.items():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            syn_stats["mean"].append(float(np.mean(s)))
            syn_stats["std"].append(float(np.std(s, ddof=1)))
            syn_stats["skewness"].append(float(_skewness(s)))
            syn_stats["cv"].append(
                float(np.std(s, ddof=1) / np.mean(s)) if np.mean(s) != 0 else np.nan
            )
            syn_stats["min"].append(float(np.min(s)))
            syn_stats["max"].append(float(np.max(s)))
            syn_stats["p10"].append(float(np.percentile(s, 10)))
            syn_stats["p50"].append(float(np.percentile(s, 50)))
            syn_stats["p90"].append(float(np.percentile(s, 90)))

        site_results = {}
        for metric_name, obs_val in obs_stats.items():
            syn_vals = np.array(syn_stats[metric_name])
            syn_vals = syn_vals[~np.isnan(syn_vals)]
            if len(syn_vals) == 0:
                continue
            rel_err = (
                (np.median(syn_vals) - obs_val) / abs(obs_val)
                if abs(obs_val) > 1e-10
                else np.nan
            )
            site_results[metric_name] = {
                "observed": obs_val,
                "synthetic_median": float(np.median(syn_vals)),
                "synthetic_p10": float(np.percentile(syn_vals, 10)),
                "synthetic_p90": float(np.percentile(syn_vals, 90)),
                "relative_error": float(rel_err),
            }
        results[site] = site_results

    return results


def _compute_temporal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: List[str],
    hurst_method: str,
) -> Dict[str, Dict]:
    """Compute temporal dependence metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()
        obs_vals = obs.values

        obs_acf1 = float(obs.autocorr(lag=1))
        obs_acf2 = float(obs.autocorr(lag=2))

        try:
            obs_hurst = compute_hurst_exponent(obs_vals, method=hurst_method)
            obs_H = obs_hurst["H"]
        except Exception:
            obs_H = np.nan

        syn_acf1 = []
        syn_acf2 = []
        syn_H = []

        for real_id, df in ensemble.data_by_realization.items():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            syn_acf1.append(float(s.autocorr(lag=1)))
            syn_acf2.append(float(s.autocorr(lag=2)))
            try:
                h = compute_hurst_exponent(s.values, method=hurst_method)
                syn_H.append(h["H"])
            except Exception:
                pass

        site_results = {}
        for name, obs_val, syn_vals in [
            ("lag1_acf", obs_acf1, syn_acf1),
            ("lag2_acf", obs_acf2, syn_acf2),
            ("hurst_exponent", obs_H, syn_H),
        ]:
            arr = np.array(syn_vals)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            rel_err = (
                (np.median(arr) - obs_val) / abs(obs_val)
                if abs(obs_val) > 1e-10
                else np.nan
            )
            site_results[name] = {
                "observed": float(obs_val),
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(rel_err),
            }
        results[site] = site_results

    return results


def _compute_spatial_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: List[str],
) -> Dict[str, float]:
    """Compute spatial correlation preservation metrics."""
    obs_corr = Q_obs[sites].corr()

    syn_corr_list = []
    for real_id, df in ensemble.data_by_realization.items():
        available = [s for s in sites if s in df.columns]
        if len(available) > 1:
            syn_corr_list.append(df[available].corr())

    if not syn_corr_list:
        return {}

    syn_corr_mean = sum(syn_corr_list) / len(syn_corr_list)

    mask = np.triu(np.ones_like(obs_corr, dtype=bool), k=1)
    obs_upper = obs_corr.values[mask]
    syn_upper = syn_corr_mean.values[mask]

    rmse = float(np.sqrt(np.mean((obs_upper - syn_upper) ** 2)))
    max_err = float(np.max(np.abs(obs_upper - syn_upper)))
    mean_bias = float(np.mean(syn_upper - obs_upper))

    return {
        "correlation_rmse": rmse,
        "correlation_max_error": max_err,
        "correlation_mean_bias": mean_bias,
    }


def _compute_drought_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: List[str],
    threshold: Optional[float],
) -> Dict[str, Dict]:
    """Compute drought duration and severity metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values

        if threshold is None:
            thresh = float(np.percentile(obs, 20))
        else:
            thresh = threshold

        obs_durations, obs_severities = _extract_droughts(obs, thresh)
        obs_drought_freq = len(obs_durations) / len(obs) if len(obs) > 0 else 0

        syn_durations_all = []
        syn_severities_all = []
        syn_freq_all = []

        for real_id, df in ensemble.data_by_realization.items():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            dur, sev = _extract_droughts(s, thresh)
            syn_durations_all.append(np.mean(dur) if len(dur) > 0 else 0)
            syn_severities_all.append(np.mean(sev) if len(sev) > 0 else 0)
            syn_freq_all.append(len(dur) / len(s) if len(s) > 0 else 0)

        obs_mean_dur = float(np.mean(obs_durations)) if len(obs_durations) > 0 else 0
        obs_mean_sev = float(np.mean(obs_severities)) if len(obs_severities) > 0 else 0

        site_results = {}
        for name, obs_val, syn_vals in [
            ("mean_drought_duration", obs_mean_dur, syn_durations_all),
            ("mean_drought_severity", obs_mean_sev, syn_severities_all),
            ("drought_frequency", obs_drought_freq, syn_freq_all),
        ]:
            arr = np.array(syn_vals)
            if len(arr) == 0:
                continue
            rel_err = (
                (np.median(arr) - obs_val) / abs(obs_val)
                if abs(obs_val) > 1e-10
                else np.nan
            )
            site_results[name] = {
                "observed": float(obs_val),
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(rel_err),
            }
        results[site] = site_results

    return results


def _compute_spectral_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: List[str],
) -> Dict[str, Dict]:
    """Compute spectral comparison metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()
        if len(obs) < 20:
            continue

        syn_rmse = []
        syn_corr = []
        syn_lf_ratio = []

        for real_id, df in ensemble.data_by_realization.items():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            if len(s) < 20:
                continue
            try:
                spec = compare_spectral_properties(obs.values, s.values)
                syn_rmse.append(spec["spectral_rmse"])
                syn_corr.append(spec["spectral_correlation"])
                syn_lf_ratio.append(spec["low_freq_ratio"])
            except Exception:
                pass

        if not syn_rmse:
            continue

        site_results = {}
        for name, vals in [
            ("spectral_rmse", syn_rmse),
            ("spectral_correlation", syn_corr),
            ("low_freq_ratio", syn_lf_ratio),
        ]:
            arr = np.array(vals)
            site_results[name] = {
                "observed": 0.0 if "rmse" in name else 1.0,
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(np.median(arr)),
            }
        results[site] = site_results

    return results


def _compute_summary_scores(result: ValidationResult) -> Dict[str, float]:
    """Compute aggregate summary scores from all metrics."""
    all_rel_errors = []

    for category in [result.marginal, result.temporal, result.spectral]:
        for site, metrics in category.items():
            for metric_name, values in metrics.items():
                if isinstance(values, dict) and "relative_error" in values:
                    re = values["relative_error"]
                    if not np.isnan(re):
                        all_rel_errors.append(abs(re))

    summary = {}
    if all_rel_errors:
        summary["mean_absolute_relative_error"] = float(np.mean(all_rel_errors))
        summary["median_absolute_relative_error"] = float(np.median(all_rel_errors))
        summary["max_absolute_relative_error"] = float(np.max(all_rel_errors))

    if result.spatial:
        summary["spatial_correlation_rmse"] = result.spatial.get(
            "correlation_rmse", np.nan
        )

    return summary


def _skewness(x: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(x)
    if n < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3))


def _extract_droughts(
    flows: np.ndarray, threshold: float
) -> tuple:
    """
    Extract drought events (consecutive below-threshold periods).

    Parameters
    ----------
    flows : np.ndarray
        Flow values.
    threshold : float
        Drought threshold.

    Returns
    -------
    durations : list of int
        Duration of each drought event.
    severities : list of float
        Cumulative deficit of each drought event.
    """
    below = flows < threshold
    durations = []
    severities = []

    in_drought = False
    current_duration = 0
    current_severity = 0.0

    for i, is_below in enumerate(below):
        if is_below:
            if not in_drought:
                in_drought = True
                current_duration = 0
                current_severity = 0.0
            current_duration += 1
            current_severity += threshold - flows[i]
        else:
            if in_drought:
                durations.append(current_duration)
                severities.append(current_severity)
                in_drought = False

    if in_drought:
        durations.append(current_duration)
        severities.append(current_severity)

    return durations, severities
