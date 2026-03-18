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
from scipy import stats as sp_stats

from synhydro.core.ensemble import Ensemble
from synhydro.core.statistics import compute_hurst_exponent, compare_spectral_properties
from synhydro.core.validation._helpers import (
    _skewness,
    _extract_droughts,
    _metric_entry,
)
from synhydro.core.validation._result import ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marginal metrics
# ---------------------------------------------------------------------------


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition). Returns nan if n < 4."""
    n = len(x)
    if n < 4:
        return np.nan
    return float(sp_stats.kurtosis(x, fisher=True, bias=False))


def _compute_marginal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute marginal (distributional) metrics per site.

    Metrics: mean, std, skewness, kurtosis, cv, min, max, p10, p50, p90,
    ks_pvalue (two-sample KS test).
    """
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values
        obs_mean = float(np.mean(obs))

        obs_stats = {
            "mean": obs_mean,
            "std": float(np.std(obs, ddof=1)),
            "skewness": _skewness(obs),
            "kurtosis": _kurtosis(obs),
            "cv": (
                float(np.std(obs, ddof=1) / obs_mean)
                if abs(obs_mean) > 1e-10
                else np.nan
            ),
            "min": float(np.min(obs)),
            "max": float(np.max(obs)),
            "p10": float(np.percentile(obs, 10)),
            "p50": float(np.percentile(obs, 50)),
            "p90": float(np.percentile(obs, 90)),
        }

        syn_stats: dict[str, list[float]] = {k: [] for k in obs_stats}
        syn_ks_pvals: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            s_mean = float(np.mean(s))
            syn_stats["mean"].append(s_mean)
            syn_stats["std"].append(float(np.std(s, ddof=1)))
            syn_stats["skewness"].append(_skewness(s))
            syn_stats["kurtosis"].append(_kurtosis(s))
            syn_stats["cv"].append(
                float(np.std(s, ddof=1) / s_mean) if abs(s_mean) > 1e-10 else np.nan
            )
            syn_stats["min"].append(float(np.min(s)))
            syn_stats["max"].append(float(np.max(s)))
            syn_stats["p10"].append(float(np.percentile(s, 10)))
            syn_stats["p50"].append(float(np.percentile(s, 50)))
            syn_stats["p90"].append(float(np.percentile(s, 90)))

            # Two-sample KS test: observed vs this realization
            try:
                ks_stat, ks_p = sp_stats.ks_2samp(obs, s)
                syn_ks_pvals.append(float(ks_p))
            except Exception:
                pass

        site_results = {}
        for metric_name, obs_val in obs_stats.items():
            entry = _metric_entry(obs_val, syn_stats[metric_name])
            if entry is not None:
                site_results[metric_name] = entry

        # KS p-value: ideal = 1.0 (distributions match)
        if syn_ks_pvals:
            arr = np.array(syn_ks_pvals)
            site_results["ks_pvalue"] = {
                "observed": 1.0,
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(np.median(arr) - 1.0),
            }

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------


def _compute_temporal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    hurst_method: str,
    acf_max_lag: int = 12,
) -> dict[str, dict]:
    """Compute temporal dependence metrics per site.

    Metrics: lag1_acf, lag2_acf, hurst_exponent, acf_rmse.
    """
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()

        try:
            obs_H = float(compute_hurst_exponent(obs.values, method=hurst_method)["H"])
        except Exception:
            obs_H = np.nan

        # Compute observed ACF up to acf_max_lag
        obs_acf = np.array(
            [
                float(obs.autocorr(lag=k)) if len(obs) > k else np.nan
                for k in range(1, acf_max_lag + 1)
            ]
        )

        syn_acf1: list[float] = []
        syn_acf2: list[float] = []
        syn_H: list[float] = []
        syn_acf_rmse: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            syn_acf1.append(float(s.autocorr(lag=1)))
            syn_acf2.append(float(s.autocorr(lag=2)))
            try:
                syn_H.append(
                    float(compute_hurst_exponent(s.values, method=hurst_method)["H"])
                )
            except Exception:
                pass

            # ACF RMSE across lags
            syn_acf = np.array(
                [
                    float(s.autocorr(lag=k)) if len(s) > k else np.nan
                    for k in range(1, acf_max_lag + 1)
                ]
            )
            valid = np.isfinite(obs_acf) & np.isfinite(syn_acf)
            if valid.sum() > 0:
                rmse = float(np.sqrt(np.mean((obs_acf[valid] - syn_acf[valid]) ** 2)))
                syn_acf_rmse.append(rmse)

        site_results = {}
        for name, obs_val, syn_vals in [
            ("lag1_acf", float(obs.autocorr(lag=1)), syn_acf1),
            ("lag2_acf", float(obs.autocorr(lag=2)), syn_acf2),
            ("hurst_exponent", obs_H, syn_H),
        ]:
            entry = _metric_entry(obs_val, syn_vals)
            if entry is not None:
                site_results[name] = entry

        # ACF RMSE: ideal = 0
        if syn_acf_rmse:
            arr = np.array(syn_acf_rmse)
            site_results["acf_rmse"] = {
                "observed": 0.0,
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(np.median(arr)),
            }

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Spatial metrics (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Drought metrics (extended)
# ---------------------------------------------------------------------------


def _compute_drought_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    threshold: Optional[float],
) -> dict[str, dict]:
    """Compute drought duration, severity, and frequency metrics per site.

    Metrics: mean_drought_duration, mean_drought_severity, drought_frequency,
    max_drought_duration, max_drought_severity.
    """
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values
        thresh = float(np.percentile(obs, 20)) if threshold is None else threshold

        obs_durations, obs_severities = _extract_droughts(obs, thresh)
        obs_mean_dur = float(np.mean(obs_durations)) if obs_durations else 0.0
        obs_mean_sev = float(np.mean(obs_severities)) if obs_severities else 0.0
        obs_max_dur = float(np.max(obs_durations)) if obs_durations else 0.0
        obs_max_sev = float(np.max(obs_severities)) if obs_severities else 0.0
        obs_freq = len(obs_durations) / len(obs) if len(obs) > 0 else 0.0

        syn_mean_dur: list[float] = []
        syn_mean_sev: list[float] = []
        syn_max_dur: list[float] = []
        syn_max_sev: list[float] = []
        syn_freq: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            dur, sev = _extract_droughts(s, thresh)
            syn_mean_dur.append(float(np.mean(dur)) if dur else 0.0)
            syn_mean_sev.append(float(np.mean(sev)) if sev else 0.0)
            syn_max_dur.append(float(np.max(dur)) if dur else 0.0)
            syn_max_sev.append(float(np.max(sev)) if sev else 0.0)
            syn_freq.append(len(dur) / len(s) if len(s) > 0 else 0.0)

        site_results = {}
        for name, obs_val, syn_vals in [
            ("mean_drought_duration", obs_mean_dur, syn_mean_dur),
            ("mean_drought_severity", obs_mean_sev, syn_mean_sev),
            ("max_drought_duration", obs_max_dur, syn_max_dur),
            ("max_drought_severity", obs_max_sev, syn_max_sev),
            ("drought_frequency", obs_freq, syn_freq),
        ]:
            entry = _metric_entry(obs_val, syn_vals)
            if entry is not None:
                site_results[name] = entry

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Spectral metrics (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Seasonal metrics (NEW)
# ---------------------------------------------------------------------------


def _compute_seasonal_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute per-month statistics comparison per site.

    For each of the 12 calendar months, compares mean, std, and skewness
    between observed and synthetic. Also reports the median Wilcoxon
    rank-sum p-value across months as a distributional equality summary.

    Metrics per site: monthly_mean_bias, monthly_std_bias,
    monthly_skewness_bias, monthly_wilcoxon_pvalue.
    """
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()
        if not hasattr(obs.index, "month"):
            continue

        # Observed monthly stats
        obs_by_month = obs.groupby(obs.index.month)
        obs_means = obs_by_month.mean()
        obs_stds = obs_by_month.std()
        obs_skews = obs_by_month.apply(lambda x: _skewness(x.values))

        # Synthetic: pool all realizations then group by month
        syn_values = []
        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            syn_values.append(df[site].dropna())
        if not syn_values:
            continue
        syn_all = pd.concat(syn_values)
        if not hasattr(syn_all.index, "month"):
            continue

        syn_by_month = syn_all.groupby(syn_all.index.month)
        syn_means = syn_by_month.mean()
        syn_stds = syn_by_month.std()
        syn_skews = syn_by_month.apply(lambda x: _skewness(x.values))

        # Compute biases across months
        months = sorted(set(obs_means.index) & set(syn_means.index))
        if not months:
            continue

        mean_biases = []
        std_biases = []
        skew_biases = []
        wilcoxon_pvals = []

        for m in months:
            obs_m = obs_means.get(m, np.nan)
            syn_m = syn_means.get(m, np.nan)
            if abs(obs_m) > 1e-10:
                mean_biases.append((syn_m - obs_m) / abs(obs_m))
            obs_s = obs_stds.get(m, np.nan)
            syn_s = syn_stds.get(m, np.nan)
            if abs(obs_s) > 1e-10:
                std_biases.append((syn_s - obs_s) / abs(obs_s))
            obs_sk = obs_skews.get(m, np.nan)
            syn_sk = syn_skews.get(m, np.nan)
            if np.isfinite(obs_sk) and abs(obs_sk) > 1e-10:
                skew_biases.append((syn_sk - obs_sk) / abs(obs_sk))

            # Wilcoxon rank-sum test per month
            try:
                obs_month_vals = obs[obs.index.month == m].values
                syn_month_vals = syn_all[syn_all.index.month == m].values
                if len(obs_month_vals) > 2 and len(syn_month_vals) > 2:
                    _, p = sp_stats.ranksums(obs_month_vals, syn_month_vals)
                    wilcoxon_pvals.append(float(p))
            except Exception:
                pass

        site_results = {}

        # Mean relative bias: ideal = 0
        if mean_biases:
            rmb = float(np.mean(np.abs(mean_biases)))
            site_results["monthly_mean_bias"] = {
                "observed": 0.0,
                "synthetic_median": rmb,
                "synthetic_p10": float(np.percentile(np.abs(mean_biases), 10)),
                "synthetic_p90": float(np.percentile(np.abs(mean_biases), 90)),
                "relative_error": rmb,
            }

        # Std relative bias: ideal = 0
        if std_biases:
            rsb = float(np.mean(np.abs(std_biases)))
            site_results["monthly_std_bias"] = {
                "observed": 0.0,
                "synthetic_median": rsb,
                "synthetic_p10": float(np.percentile(np.abs(std_biases), 10)),
                "synthetic_p90": float(np.percentile(np.abs(std_biases), 90)),
                "relative_error": rsb,
            }

        # Skewness relative bias: ideal = 0
        if skew_biases:
            rskb = float(np.mean(np.abs(skew_biases)))
            site_results["monthly_skewness_bias"] = {
                "observed": 0.0,
                "synthetic_median": rskb,
                "synthetic_p10": float(np.percentile(np.abs(skew_biases), 10)),
                "synthetic_p90": float(np.percentile(np.abs(skew_biases), 90)),
                "relative_error": rskb,
            }

        # Median Wilcoxon p-value across months: ideal = 1.0 (no significant differences)
        if wilcoxon_pvals:
            arr = np.array(wilcoxon_pvals)
            site_results["monthly_wilcoxon_pvalue"] = {
                "observed": 1.0,
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(np.median(arr) - 1.0),
            }

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Annual metrics (NEW)
# ---------------------------------------------------------------------------


def _compute_annual_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute annual aggregate statistics per site.

    Metrics: annual_mean, annual_variance, annual_skewness, annual_lag1_acf,
    variance_ratio (annual variance of synthetic / annual variance of observed).
    """
    results = {}

    for site in sites:
        obs = Q_obs[site].dropna()
        obs_annual = obs.resample("YS").sum().dropna()
        if len(obs_annual) < 5:
            continue

        obs_stats = {
            "annual_mean": float(obs_annual.mean()),
            "annual_variance": float(obs_annual.var(ddof=1)),
            "annual_skewness": _skewness(obs_annual.values),
            "annual_lag1_acf": (
                float(obs_annual.autocorr(lag=1)) if len(obs_annual) > 1 else np.nan
            ),
        }
        obs_var = obs_stats["annual_variance"]

        syn_stats: dict[str, list[float]] = {k: [] for k in obs_stats}
        syn_var_ratio: list[float] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna()
            s_annual = s.resample("YS").sum().dropna()
            if len(s_annual) < 3:
                continue
            syn_stats["annual_mean"].append(float(s_annual.mean()))
            syn_stats["annual_variance"].append(float(s_annual.var(ddof=1)))
            syn_stats["annual_skewness"].append(_skewness(s_annual.values))
            if len(s_annual) > 1:
                syn_stats["annual_lag1_acf"].append(float(s_annual.autocorr(lag=1)))

            # Variance ratio: synthetic / observed (ideal = 1.0)
            s_var = float(s_annual.var(ddof=1))
            if abs(obs_var) > 1e-10:
                syn_var_ratio.append(s_var / obs_var)

        site_results = {}
        for metric_name, obs_val in obs_stats.items():
            entry = _metric_entry(obs_val, syn_stats[metric_name])
            if entry is not None:
                site_results[metric_name] = entry

        # Variance ratio: ideal = 1.0
        if syn_var_ratio:
            arr = np.array(syn_var_ratio)
            site_results["variance_ratio"] = {
                "observed": 1.0,
                "synthetic_median": float(np.median(arr)),
                "synthetic_p10": float(np.percentile(arr, 10)),
                "synthetic_p90": float(np.percentile(arr, 90)),
                "relative_error": float(np.median(arr) - 1.0),
            }

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Flow duration curve metrics (NEW)
# ---------------------------------------------------------------------------


def _compute_fdc_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
) -> dict[str, dict]:
    """Compute flow duration curve comparison metrics per site.

    Metrics: fdc_rmse (log-space RMSE of FDC), fdc_bias_q10, fdc_bias_q50,
    fdc_bias_q90, fdc_envelope_coverage (fraction of observed exceedance
    probabilities where the observed flow falls within the ensemble range).
    """
    # Common exceedance probability grid
    probs = np.linspace(0.001, 0.999, 200)

    results = {}

    for site in sites:
        obs = Q_obs[site].dropna().values
        if len(obs) < 10:
            continue
        obs_quantiles = np.quantile(obs, probs)

        syn_fdc_rmse: list[float] = []
        syn_bias_q10: list[float] = []
        syn_bias_q50: list[float] = []
        syn_bias_q90: list[float] = []
        syn_quantile_arrays: list[np.ndarray] = []

        for df in ensemble.data_by_realization.values():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            if len(s) < 10:
                continue
            syn_q = np.quantile(s, probs)
            syn_quantile_arrays.append(syn_q)

            # Log-space RMSE of FDC
            obs_log = np.log(np.clip(obs_quantiles, 1e-6, None))
            syn_log = np.log(np.clip(syn_q, 1e-6, None))
            valid = np.isfinite(obs_log) & np.isfinite(syn_log)
            if valid.sum() > 0:
                syn_fdc_rmse.append(
                    float(np.sqrt(np.mean((obs_log[valid] - syn_log[valid]) ** 2)))
                )

            # Bias at key exceedances
            obs_q10 = float(np.quantile(obs, 0.10))
            obs_q50 = float(np.quantile(obs, 0.50))
            obs_q90 = float(np.quantile(obs, 0.90))
            syn_q10 = float(np.quantile(s, 0.10))
            syn_q50 = float(np.quantile(s, 0.50))
            syn_q90 = float(np.quantile(s, 0.90))

            if abs(obs_q10) > 1e-10:
                syn_bias_q10.append((syn_q10 - obs_q10) / abs(obs_q10))
            if abs(obs_q50) > 1e-10:
                syn_bias_q50.append((syn_q50 - obs_q50) / abs(obs_q50))
            if abs(obs_q90) > 1e-10:
                syn_bias_q90.append((syn_q90 - obs_q90) / abs(obs_q90))

        if not syn_fdc_rmse:
            continue

        site_results = {}

        # FDC RMSE: ideal = 0
        arr = np.array(syn_fdc_rmse)
        site_results["fdc_rmse"] = {
            "observed": 0.0,
            "synthetic_median": float(np.median(arr)),
            "synthetic_p10": float(np.percentile(arr, 10)),
            "synthetic_p90": float(np.percentile(arr, 90)),
            "relative_error": float(np.median(arr)),
        }

        # Bias at key exceedances
        for name, vals in [
            ("fdc_bias_q10", syn_bias_q10),
            ("fdc_bias_q50", syn_bias_q50),
            ("fdc_bias_q90", syn_bias_q90),
        ]:
            if vals:
                a = np.array(vals)
                site_results[name] = {
                    "observed": 0.0,
                    "synthetic_median": float(np.median(a)),
                    "synthetic_p10": float(np.percentile(a, 10)),
                    "synthetic_p90": float(np.percentile(a, 90)),
                    "relative_error": float(np.median(np.abs(a))),
                }

        # Ensemble envelope coverage: fraction of observed quantiles that
        # fall within the min-max range of synthetic quantiles
        if len(syn_quantile_arrays) >= 2:
            stacked = np.stack(syn_quantile_arrays, axis=0)
            env_min = np.min(stacked, axis=0)
            env_max = np.max(stacked, axis=0)
            coverage = float(
                np.mean((obs_quantiles >= env_min) & (obs_quantiles <= env_max))
            )
            site_results["fdc_envelope_coverage"] = {
                "observed": 1.0,
                "synthetic_median": coverage,
                "synthetic_p10": coverage,
                "synthetic_p90": coverage,
                "relative_error": float(coverage - 1.0),
            }

        results[site] = site_results

    return results


# ---------------------------------------------------------------------------
# Summary scores
# ---------------------------------------------------------------------------


def _compute_summary_scores(result: ValidationResult) -> dict[str, float]:
    """Compute aggregate summary scores across all metric categories."""
    all_rel_errors = []

    for category in [
        result.marginal,
        result.temporal,
        result.drought,
        result.spectral,
        result.seasonal,
        result.annual,
        result.fdc,
    ]:
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
