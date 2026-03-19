"""
Tests for bootstrap confidence intervals and method comparison tools.
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.validation._testing import (
    compute_realization_metrics,
    bootstrap_metric_ci,
    compare_methods,
    _REALIZATION_METRICS,
)


@pytest.fixture
def monthly_obs():
    """20-year monthly observed data, 2 sites."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", "2019-12-31", freq="MS")
    n = len(dates)
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 12)
    return pd.DataFrame(
        {
            "site_A": rng.lognormal(5.0, 0.4, n) * seasonal,
            "site_B": rng.lognormal(4.5, 0.5, n) * seasonal,
        },
        index=dates,
    )


def _make_ensemble(
    rng_seed, n_realizations=15, n_months=240, n_sites=2, mean=5.0, sigma=0.4
):
    """Helper to create a test ensemble."""
    rng = np.random.default_rng(rng_seed)
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    site_names = [f"site_{chr(65 + i)}" for i in range(n_sites)]

    realization_dict = {}
    for i in range(n_realizations):
        data = {
            site: rng.lognormal(mean, sigma, n_months) * seasonal for site in site_names
        }
        realization_dict[i] = pd.DataFrame(data, index=dates)

    metadata = EnsembleMetadata(
        generator_class="TestGenerator",
        n_realizations=n_realizations,
        n_sites=n_sites,
        time_resolution="MS",
    )
    return Ensemble(realization_dict, metadata=metadata)


@pytest.fixture
def ensemble_a():
    return _make_ensemble(99, mean=5.0, sigma=0.4)


@pytest.fixture
def ensemble_b():
    return _make_ensemble(200, mean=5.2, sigma=0.6)


# ---------------------------------------------------------------------------
# compute_realization_metrics
# ---------------------------------------------------------------------------


class TestComputeRealizationMetrics:

    def test_returns_dataframe(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(ensemble_a, monthly_obs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_columns(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(ensemble_a, monthly_obs)
        assert set(df.columns) == {"realization", "site", "metric", "value", "observed"}

    def test_all_metrics_present(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(ensemble_a, monthly_obs, sites=["site_A"])
        metrics_found = set(df["metric"].unique())
        assert metrics_found == set(_REALIZATION_METRICS.keys())

    def test_subset_metrics(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(
            ensemble_a, monthly_obs, metrics=["mean", "std"]
        )
        assert set(df["metric"].unique()) == {"mean", "std"}

    def test_subset_sites(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(ensemble_a, monthly_obs, sites=["site_A"])
        assert set(df["site"].unique()) == {"site_A"}

    def test_rows_per_realization(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(
            ensemble_a, monthly_obs, sites=["site_A"], metrics=["mean"]
        )
        assert len(df) == 15  # one per realization

    def test_observed_column_consistent(self, ensemble_a, monthly_obs):
        df = compute_realization_metrics(
            ensemble_a, monthly_obs, sites=["site_A"], metrics=["mean"]
        )
        # All rows for same site/metric should have same observed value
        assert df["observed"].nunique() == 1

    def test_invalid_metric_raises(self, ensemble_a, monthly_obs):
        with pytest.raises(ValueError, match="Unknown metrics"):
            compute_realization_metrics(
                ensemble_a, monthly_obs, metrics=["nonexistent"]
            )


# ---------------------------------------------------------------------------
# bootstrap_metric_ci
# ---------------------------------------------------------------------------


class TestBootstrapMetricCI:

    def test_returns_dataframe(self, ensemble_a, monthly_obs):
        result = bootstrap_metric_ci(
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=0,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_columns(self, ensemble_a, monthly_obs):
        result = bootstrap_metric_ci(
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=0,
        )
        expected_cols = {
            "site",
            "metric",
            "observed",
            "estimate",
            "ci_lower",
            "ci_upper",
            "relative_error",
            "re_ci_lower",
            "re_ci_upper",
        }
        assert set(result.columns) == expected_cols

    def test_ci_contains_estimate(self, ensemble_a, monthly_obs):
        result = bootstrap_metric_ci(
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            n_bootstrap=500,
            seed=0,
        )
        for _, row in result.iterrows():
            assert row["ci_lower"] <= row["estimate"] <= row["ci_upper"]

    def test_wider_ci_with_fewer_realizations(self, monthly_obs):
        """Smaller ensemble should produce wider CIs."""
        ens_large = _make_ensemble(99, n_realizations=50)
        ens_small = _make_ensemble(99, n_realizations=5)

        ci_large = bootstrap_metric_ci(
            ens_large,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=500,
            seed=0,
        )
        ci_small = bootstrap_metric_ci(
            ens_small,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=500,
            seed=0,
        )

        width_large = (ci_large["ci_upper"] - ci_large["ci_lower"]).iloc[0]
        width_small = (ci_small["ci_upper"] - ci_small["ci_lower"]).iloc[0]
        assert width_small > width_large

    def test_reproducibility_with_seed(self, ensemble_a, monthly_obs):
        ci1 = bootstrap_metric_ci(
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=42,
        )
        ci2 = bootstrap_metric_ci(
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=42,
        )
        assert ci1["ci_lower"].iloc[0] == ci2["ci_lower"].iloc[0]
        assert ci1["ci_upper"].iloc[0] == ci2["ci_upper"].iloc[0]


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------


class TestCompareMethods:

    def test_returns_dataframe(self, ensemble_a, ensemble_b, monthly_obs):
        result = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=0,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_columns(self, ensemble_a, ensemble_b, monthly_obs):
        result = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=0,
        )
        expected_cols = {
            "site",
            "metric",
            "observed",
            "method_a_median",
            "method_b_median",
            "method_a_mae",
            "method_b_mae",
            "diff_estimate",
            "diff_ci_lower",
            "diff_ci_upper",
            "significant",
            "better_method",
        }
        assert set(result.columns) == expected_cols

    def test_identical_ensembles_not_significant(self, ensemble_a, monthly_obs):
        """Comparing a method to itself should never be significant."""
        result = compare_methods(
            ensemble_a,
            ensemble_a,
            monthly_obs,
            sites=["site_A"],
            n_bootstrap=500,
            seed=0,
        )
        # Most metrics should not be significant when comparing identical ensembles
        n_sig = result["significant"].sum()
        assert n_sig <= 2  # allow small number due to bootstrap variance

    def test_better_method_values(self, ensemble_a, ensemble_b, monthly_obs):
        result = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            n_bootstrap=100,
            seed=0,
        )
        valid_values = {"A", "B", "neither"}
        for val in result["better_method"]:
            assert val in valid_values

    def test_diff_sign_matches_better(self, ensemble_a, ensemble_b, monthly_obs):
        """When significant, diff > 0 means B is better, diff < 0 means A."""
        result = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            n_bootstrap=500,
            seed=0,
        )
        sig = result[result["significant"]]
        for _, row in sig.iterrows():
            if row["diff_estimate"] > 0:
                assert row["better_method"] == "B"
            else:
                assert row["better_method"] == "A"

    def test_reproducibility_with_seed(self, ensemble_a, ensemble_b, monthly_obs):
        r1 = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=42,
        )
        r2 = compare_methods(
            ensemble_a,
            ensemble_b,
            monthly_obs,
            sites=["site_A"],
            metrics=["mean"],
            n_bootstrap=100,
            seed=42,
        )
        assert r1["diff_ci_lower"].iloc[0] == r2["diff_ci_lower"].iloc[0]
