"""
Metric computation functions for ensemble validation.

Each function receives an Ensemble, the observed DataFrame, and a list of
site names to process. All return dicts keyed by site (or a flat dict for
spatial metrics).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.core.statistics import compute_hurst_exponent, compare_spectral_properties
from synhydro.core.validation._helpers import _skewness, _extract_droughts, _metric_entry
from synhydro.core.validation._result import ValidationResult

logger = logging.getLogger(__name__)


def _compute_marginal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute marginal (distributional) metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values
        obs_mean = float(np.mean(obs))

        obs_stats = {
            "mean": obs_mean,
            "std": float(np.std(obs, ddof=1)),
            "skewness": _skewness(obs),
            "cv": float(np.std(obs, ddof=1) / obs_mean) if abs(obs_mean) > 1e-10 else np.nan,
            "min": float(np.min(obs)),
            "max": float(np.max(obs)),
            "p10": float(np.percentile(obs, 10)),
            "p50": float(np.percentile(obs, 50)),
            "p90": float(np.percentile(obs, 90)),
        }

        syn_stats: dict[str, list[float]] = {k: [] for k in obs_stats}

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            s_mean = float(np.mean(s))
            syn_stats["mean"].append(s_mean)
            syn_stats["std"].append(float(np.std(s, ddof=1)))
            syn_stats["skewness"].append(_skewness(s))
            syn_stats["cv"].append(
                float(np.std(s, ddof=1) / s_mean) if abs(s_mean) > 1e-10 else np.nan
            )
            syn_stats["min"].append(float(np.min(s)))
            syn_stats["max"].append(float(np.max(s)))
            syn_stats["p10"].append(float(np.percentile(s, 10)))
            syn_stats["p50"].append(float(np.percentile(s, 50)))
            syn_stats["p90"].append(float(np.percentile(s, 90)))

        site_results = {}
        for metric_name, obs_val in obs_stats.items():
            entry = _metric_entry(obs_val, syn_stats[metric_name])
            if entry is not None:
                site_results[metric_name] = entry

        results[site] = site_results

    return results


def _compute_temporal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    hurst_method: str,
) -> dict[str, dict]:
    """Compute temporal dependence metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()

        try:
            obs_H = float(compute_hurst_exponent(obs.values, method=hurst_method)["H"])
        except Exception:
            obs_H = np.nan

        syn_acf1: list[float] = []
        syn_acf2: list[float] = []
        syn_H: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            syn_acf1.append(float(s.autocorr(lag=1)))
            syn_acf2.append(float(s.autocorr(lag=2)))
            try:
                syn_H.append(float(compute_hurst_exponent(s.values, method=hurst_method)["H"]))
            except Exception:
                pass

        site_results = {}
        for name, obs_val, syn_vals in [
            ("lag1_acf", float(obs.autocorr(lag=1)), syn_acf1),
            ("lag2_acf", float(obs.autocorr(lag=2)), syn_acf2),
            ("hurst_exponent", obs_H, syn_H),
        ]:
            entry = _metric_entry(obs_val, syn_vals)
            if entry is not None:
                site_results[name] = entry

        results[site] = site_results

    return results


def _compute_spatial_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, float]:
    """Compute spatial correlation preservation metrics."""
    obs_corr = Q_obs[sites].corr().values

    syn_corr_arrays = []
    for df in ensemble.data_by_realization.values():
        available = [s for s in sites if s in df.columns]
        if len(available) > 1:
            syn_corr_arrays.append(df[available].corr().values)

    if not syn_corr_arrays:
        return {}

    syn_corr_mean = np.mean(np.stack(syn_corr_arrays, axis=0), axis=0)

    mask = np.triu(np.ones(obs_corr.shape, dtype=bool), k=1)
    obs_upper = obs_corr[mask]
    syn_upper = syn_corr_mean[mask]

    return {
        "correlation_rmse": float(np.sqrt(np.mean((obs_upper - syn_upper) ** 2))),
        "correlation_max_error": float(np.max(np.abs(obs_upper - syn_upper))),
        "correlation_mean_bias": float(np.mean(syn_upper - obs_upper)),
    }


def _compute_drought_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    threshold: Optional[float],
) -> dict[str, dict]:
    """Compute drought duration, severity, and frequency metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values
        thresh = float(np.percentile(obs, 20)) if threshold is None else threshold

        obs_durations, obs_severities = _extract_droughts(obs, thresh)
        obs_mean_dur = float(np.mean(obs_durations)) if obs_durations else 0.0
        obs_mean_sev = float(np.mean(obs_severities)) if obs_severities else 0.0
        obs_freq = len(obs_durations) / len(obs) if len(obs) > 0 else 0.0

        syn_durations_all: list[float] = []
        syn_severities_all: list[float] = []
        syn_freq_all: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            dur, sev = _extract_droughts(s, thresh)
            syn_durations_all.append(float(np.mean(dur)) if dur else 0.0)
            syn_severities_all.append(float(np.mean(sev)) if sev else 0.0)
            syn_freq_all.append(len(dur) / len(s) if len(s) > 0 else 0.0)

        site_results = {}
        for name, obs_val, syn_vals in [
            ("mean_drought_duration", obs_mean_dur, syn_durations_all),
            ("mean_drought_severity", obs_mean_sev, syn_severities_all),
            ("drought_frequency", obs_freq, syn_freq_all),
        ]:
            entry = _metric_entry(obs_val, syn_vals)
            if entry is not None:
                site_results[name] = entry

        results[site] = site_results

    return results


def _compute_spectral_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute spectral comparison metrics per site."""
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()
        if len(obs) < 20:
            continue

        syn_rmse: list[float] = []
        syn_corr: list[float] = []
        syn_lf_ratio: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            if len(s) < 20:
                continue
            try:
                spec = compare_spectral_properties(obs.values, s.values)
                syn_rmse.append(float(spec["spectral_rmse"]))
                syn_corr.append(float(spec["spectral_correlation"]))
                syn_lf_ratio.append(float(spec["low_freq_ratio"]))
            except Exception:
                pass

        if not syn_rmse:
            continue

        arr_rmse = np.array(syn_rmse)
        arr_corr = np.array(syn_corr)
        arr_lf = np.array(syn_lf_ratio)

        # spectral_rmse: ideal = 0; relative_error is the absolute median value
        # spectral_correlation / low_freq_ratio: ideal = 1; relative_error = median - 1
        site_results = {
            "spectral_rmse": {
                "observed": 0.0,
                "synthetic_median": float(np.median(arr_rmse)),
                "synthetic_p10": float(np.percentile(arr_rmse, 10)),
                "synthetic_p90": float(np.percentile(arr_rmse, 90)),
                "relative_error": float(np.median(arr_rmse)),
            },
            "spectral_correlation": {
                "observed": 1.0,
                "synthetic_median": float(np.median(arr_corr)),
                "synthetic_p10": float(np.percentile(arr_corr, 10)),
                "synthetic_p90": float(np.percentile(arr_corr, 90)),
                "relative_error": float(np.median(arr_corr) - 1.0),
            },
            "low_freq_ratio": {
                "observed": 1.0,
                "synthetic_median": float(np.median(arr_lf)),
                "synthetic_p10": float(np.percentile(arr_lf, 10)),
                "synthetic_p90": float(np.percentile(arr_lf, 90)),
                "relative_error": float(np.median(arr_lf) - 1.0),
            },
        }
        results[site] = site_results

    return results


def _compute_summary_scores(result: ValidationResult) -> dict[str, float]:
    """Compute aggregate summary scores across all metric categories."""
    all_rel_errors = []

    for category in [result.marginal, result.temporal, result.drought, result.spectral]:
        for site_metrics in category.values():
            for values in site_metrics.values():
                if isinstance(values, dict):
                    re = values.get("relative_error")
                    if re is not None and np.isfinite(re):
                        all_rel_errors.append(abs(re))

    summary: dict[str, float] = {}
    if all_rel_errors:
        summary["mean_absolute_relative_error"] = float(np.mean(all_rel_errors))
        summary["median_absolute_relative_error"] = float(np.median(all_rel_errors))
        summary["max_absolute_relative_error"] = float(np.max(all_rel_errors))

    if result.spatial:
        summary["spatial_correlation_rmse"] = float(
            result.spatial.get("correlation_rmse", np.nan)
        )

    return summary
