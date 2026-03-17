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
        Per-site marginal statistics (mean, std, skewness, cv, percentiles).
    temporal : dict[str, dict]
        Per-site temporal statistics (lag-1 ACF, lag-2 ACF, Hurst exponent).
    spatial : dict[str, float]
        Cross-site correlation preservation metrics.
    drought : dict[str, dict]
        Per-site drought duration, severity, and frequency statistics.
    spectral : dict[str, dict]
        Per-site spectral comparison metrics.
    summary : dict[str, float]
        Aggregate summary scores across all metric categories.
    """

    marginal: dict[str, dict] = field(default_factory=dict)
    temporal: dict[str, dict] = field(default_factory=dict)
    spatial: dict[str, float] = field(default_factory=dict)
    drought: dict[str, dict] = field(default_factory=dict)
    spectral: dict[str, dict] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten per-site metric results into a tidy DataFrame.

        Includes marginal, temporal, drought, and spectral categories.
        Spatial metrics (site-pair level) are excluded.

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
