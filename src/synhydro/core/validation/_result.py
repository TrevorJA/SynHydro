"""
ValidationResult container for ensemble validation output.
"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationResult:
    """
    Container for ensemble validation results.

    Attributes
    ----------
    marginal : dict[str, dict]
        Per-site marginal statistics (mean, std, skewness, kurtosis, cv,
        percentiles, KS test p-value).
    temporal : dict[str, dict]
        Per-site temporal statistics (lag-1/2 ACF, Hurst exponent, ACF RMSE).
    spatial : dict[str, float]
        Cross-site correlation preservation metrics.
    drought : dict[str, dict]
        Per-site drought duration, severity, and frequency (mean and max).
    spectral : dict[str, dict]
        Per-site spectral comparison metrics.
    seasonal : dict[str, dict]
        Per-site monthly statistics (mean bias, std bias, skewness bias,
        Wilcoxon p-values per month).
    annual : dict[str, dict]
        Per-site annual aggregate statistics (mean, variance, skewness,
        lag-1 ACF, cross-scale variance ratio).
    fdc : dict[str, dict]
        Per-site flow duration curve metrics (RMSE, bias at key exceedances,
        ensemble envelope coverage).
    lmoments : dict[str, dict]
        Per-site L-moment ratio metrics (L-CV, L-skewness, L-kurtosis).
    extremes : dict[str, dict]
        Per-site extreme value metrics (annual max statistics, GEV return
        period quantiles, 7-day minimum flow statistics).
    crps : dict[str, dict]
        Per-site CRPS (Continuous Ranked Probability Score) and CRPSS
        (skill score vs climatology).
    ssi_drought : dict[str, dict]
        Per-site SSI-based drought metrics (duration, severity, frequency)
        using SSI < -1 threshold.
    summary : dict[str, float]
        Aggregate summary scores across all metric categories.
    """

    marginal: dict[str, dict] = field(default_factory=dict)
    temporal: dict[str, dict] = field(default_factory=dict)
    spatial: dict[str, float] = field(default_factory=dict)
    drought: dict[str, dict] = field(default_factory=dict)
    spectral: dict[str, dict] = field(default_factory=dict)
    seasonal: dict[str, dict] = field(default_factory=dict)
    annual: dict[str, dict] = field(default_factory=dict)
    fdc: dict[str, dict] = field(default_factory=dict)
    lmoments: dict[str, dict] = field(default_factory=dict)
    extremes: dict[str, dict] = field(default_factory=dict)
    crps: dict[str, dict] = field(default_factory=dict)
    ssi_drought: dict[str, dict] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten per-site metric results into a tidy DataFrame.

        Includes all per-site metric categories. Spatial metrics
        (site-pair level) are excluded.

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
            ("drought", self.drought),
            ("spectral", self.spectral),
            ("seasonal", self.seasonal),
            ("annual", self.annual),
            ("fdc", self.fdc),
            ("lmoments", self.lmoments),
            ("extremes", self.extremes),
            ("crps", self.crps),
            ("ssi_drought", self.ssi_drought),
        ]:
            for site, metrics in site_metrics.items():
                for metric_name, values in metrics.items():
                    if isinstance(values, dict) and "observed" in values:
                        rows.append(
                            {
                                "category": category,
                                "metric": metric_name,
                                "site": site,
                                "observed": values["observed"],
                                "synthetic_median": values.get("synthetic_median"),
                                "synthetic_p10": values.get("synthetic_p10"),
                                "synthetic_p90": values.get("synthetic_p90"),
                                "relative_error": values.get("relative_error"),
                            }
                        )
        return pd.DataFrame(rows)
