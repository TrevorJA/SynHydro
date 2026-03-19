"""
Bootstrap confidence intervals and method comparison tools.

Provides statistical testing infrastructure for comparing synthetic
generation methods. Operates on per-realization metric values extracted
from ensembles, enabling fast bootstrap resampling without re-running
the full validation pipeline.

References
----------
Efron, B. and Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
Chapman & Hall/CRC.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.core.validation._helpers import (
    _skewness,
    _extract_droughts,
    _metric_entry,
)
from synhydro.core.statistics import compute_hurst_exponent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-realization metric extraction
# ---------------------------------------------------------------------------

# Metrics computed per realization. Each is a function:
#   f(realization_series, observed_series) -> float
_REALIZATION_METRICS = {
    "mean": lambda s, o: float(np.mean(s)),
    "std": lambda s, o: float(np.std(s, ddof=1)),
    "skewness": lambda s, o: _skewness(s),
    "cv": lambda s, o: (
        float(np.std(s, ddof=1) / np.mean(s)) if abs(np.mean(s)) > 1e-10 else np.nan
    ),
    "lag1_acf": lambda s, o: (
        float(pd.Series(s).autocorr(lag=1)) if len(s) > 1 else np.nan
    ),
    "lag2_acf": lambda s, o: (
        float(pd.Series(s).autocorr(lag=2)) if len(s) > 2 else np.nan
    ),
    "annual_variance": lambda s, o: _annual_variance(s),
    "p10": lambda s, o: float(np.percentile(s, 10)),
    "p50": lambda s, o: float(np.percentile(s, 50)),
    "p90": lambda s, o: float(np.percentile(s, 90)),
}


def _annual_variance(values: np.ndarray) -> float:
    """Compute variance of annual sums from a monthly/daily series."""
    # Approximate: treat every 12 values as one year
    n = len(values)
    if n < 24:
        return np.nan
    n_years = n // 12
    annual_sums = np.array(
        [np.sum(values[i * 12 : (i + 1) * 12]) for i in range(n_years)]
    )
    return float(np.var(annual_sums, ddof=1))


def compute_realization_metrics(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract per-realization metric values from an ensemble.

    For each realization, site, and metric, computes a single scalar value.
    The result is a tidy DataFrame suitable for bootstrap resampling.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble.
    Q_obs : pd.DataFrame
        Observed streamflow with DatetimeIndex.
    sites : list of str, optional
        Sites to evaluate. If None, uses all shared sites.
    metrics : list of str, optional
        Metric names to compute. If None, uses all available metrics.
        Available: mean, std, skewness, cv, lag1_acf, lag2_acf,
        annual_variance, p10, p50, p90.

    Returns
    -------
    pd.DataFrame
        Columns: realization, site, metric, value, observed.
        One row per (realization, site, metric) combination.
    """
    if sites is None:
        ens_sites = ensemble.site_names
        sites = [s for s in ens_sites if s in Q_obs.columns]
    if metrics is None:
        metrics = list(_REALIZATION_METRICS.keys())

    unknown = set(metrics) - set(_REALIZATION_METRICS.keys())
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. "
            f"Available: {sorted(_REALIZATION_METRICS.keys())}"
        )

    rows = []
    for site in sites:
        obs = Q_obs[site].dropna().values
        if len(obs) < 10:
            continue

        # Pre-compute observed values for each metric
        obs_vals = {}
        for metric_name in metrics:
            fn = _REALIZATION_METRICS[metric_name]
            obs_vals[metric_name] = fn(obs, obs)

        for real_id, df in ensemble.data_by_realization.items():
            if site not in df.columns:
                continue
            s = df[site].dropna().values
            if len(s) < 10:
                continue
            for metric_name in metrics:
                fn = _REALIZATION_METRICS[metric_name]
                val = fn(s, obs)
                rows.append(
                    {
                        "realization": real_id,
                        "site": site,
                        "metric": metric_name,
                        "value": val,
                        "observed": obs_vals[metric_name],
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_metric_ci(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute bootstrap confidence intervals on validation metrics.

    For each metric, resamples ensemble realizations with replacement
    and recomputes the median metric value. The CI reflects uncertainty
    due to finite ensemble size.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble.
    Q_obs : pd.DataFrame
        Observed streamflow.
    sites : list of str, optional
        Sites to evaluate.
    metrics : list of str, optional
        Metrics to bootstrap. See ``compute_realization_metrics``.
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: site, metric, observed, estimate, ci_lower, ci_upper,
        relative_error, re_ci_lower, re_ci_upper.
    """
    rng = np.random.default_rng(seed)

    # Get per-realization values
    df = compute_realization_metrics(ensemble, Q_obs, sites, metrics)
    if df.empty:
        return pd.DataFrame()

    alpha = (1 - confidence_level) / 2
    results = []

    for (site, metric), group in df.groupby(["site", "metric"]):
        values = group["value"].values
        obs_val = group["observed"].iloc[0]
        n = len(values)
        if n < 3:
            continue

        # Bootstrap: resample realization values, compute median each time
        boot_medians = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_medians[b] = np.nanmedian(sample)

        estimate = float(np.nanmedian(values))
        ci_lower = float(np.nanpercentile(boot_medians, 100 * alpha))
        ci_upper = float(np.nanpercentile(boot_medians, 100 * (1 - alpha)))

        # Relative errors
        if abs(obs_val) > 1e-10:
            re = (estimate - obs_val) / abs(obs_val)
            re_lower = (ci_lower - obs_val) / abs(obs_val)
            re_upper = (ci_upper - obs_val) / abs(obs_val)
        else:
            re = re_lower = re_upper = np.nan

        results.append(
            {
                "site": site,
                "metric": metric,
                "observed": float(obs_val),
                "estimate": estimate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "relative_error": float(re),
                "re_ci_lower": float(re_lower),
                "re_ci_upper": float(re_upper),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------


def compare_methods(
    ensemble_a: Ensemble,
    ensemble_b: Ensemble,
    Q_obs: pd.DataFrame,
    sites: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Paired bootstrap test comparing two generation methods.

    For each metric, computes per-realization absolute errors for both
    methods, then bootstrap-resamples the difference (|error_A| - |error_B|).
    If the CI of the difference excludes zero, the methods are significantly
    different for that metric.

    Parameters
    ----------
    ensemble_a : Ensemble
        First method's ensemble.
    ensemble_b : Ensemble
        Second method's ensemble.
    Q_obs : pd.DataFrame
        Observed streamflow.
    sites : list of str, optional
        Sites to compare.
    metrics : list of str, optional
        Metrics to compare. See ``compute_realization_metrics``.
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    confidence_level : float, default=0.95
        Confidence level for intervals.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: site, metric, method_a_median, method_b_median,
        diff_estimate (positive means A has larger error), diff_ci_lower,
        diff_ci_upper, significant (True if CI excludes zero),
        better_method ('A', 'B', or 'neither').
    """
    rng = np.random.default_rng(seed)

    df_a = compute_realization_metrics(ensemble_a, Q_obs, sites, metrics)
    df_b = compute_realization_metrics(ensemble_b, Q_obs, sites, metrics)

    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    alpha = (1 - confidence_level) / 2
    results = []

    # Find common (site, metric) pairs
    keys_a = set(zip(df_a["site"], df_a["metric"]))
    keys_b = set(zip(df_b["site"], df_b["metric"]))
    common_keys = keys_a & keys_b

    for site, metric in sorted(common_keys):
        group_a = df_a[(df_a["site"] == site) & (df_a["metric"] == metric)]
        group_b = df_b[(df_b["site"] == site) & (df_b["metric"] == metric)]

        vals_a = group_a["value"].values
        vals_b = group_b["value"].values
        obs_val = group_a["observed"].iloc[0]

        if len(vals_a) < 3 or len(vals_b) < 3:
            continue

        # Per-realization absolute errors
        errors_a = np.abs(vals_a - obs_val)
        errors_b = np.abs(vals_b - obs_val)

        median_a = float(np.nanmedian(vals_a))
        median_b = float(np.nanmedian(vals_b))
        mae_a = float(np.nanmean(errors_a))
        mae_b = float(np.nanmean(errors_b))

        # Bootstrap the difference in MAE: positive means A worse
        n_a = len(errors_a)
        n_b = len(errors_b)
        boot_diffs = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            sample_a = rng.choice(errors_a, size=n_a, replace=True)
            sample_b = rng.choice(errors_b, size=n_b, replace=True)
            boot_diffs[b] = np.nanmean(sample_a) - np.nanmean(sample_b)

        diff_est = mae_a - mae_b
        diff_lower = float(np.nanpercentile(boot_diffs, 100 * alpha))
        diff_upper = float(np.nanpercentile(boot_diffs, 100 * (1 - alpha)))

        # Significance: CI excludes zero
        significant = (diff_lower > 0) or (diff_upper < 0)
        if significant:
            better = "B" if diff_est > 0 else "A"
        else:
            better = "neither"

        results.append(
            {
                "site": site,
                "metric": metric,
                "observed": float(obs_val),
                "method_a_median": median_a,
                "method_b_median": median_b,
                "method_a_mae": mae_a,
                "method_b_mae": mae_b,
                "diff_estimate": float(diff_est),
                "diff_ci_lower": diff_lower,
                "diff_ci_upper": diff_upper,
                "significant": significant,
                "better_method": better,
            }
        )

    return pd.DataFrame(results)
