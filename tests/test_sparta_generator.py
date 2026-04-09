"""Tests for the SPARTAGenerator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.methods.generation.parametric.sparta import SPARTAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_multisite():
    """Synthetic monthly data: 30 years, 3 correlated sites with seasonality."""
    rng = np.random.default_rng(42)
    n_years = 30
    n_months = n_years * 12
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")

    from scipy.stats import gamma as gamma_dist, norm

    # Generate correlated Gaussian with month-to-month persistence
    z = np.zeros((n_months, 3))
    z[0] = rng.standard_normal(3)
    corr_mat = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]])
    L = np.linalg.cholesky(corr_mat)
    for t in range(1, n_months):
        w = L @ rng.standard_normal(3)
        z[t] = 0.5 * z[t - 1] + np.sqrt(1 - 0.25) * w

    # Map to gamma with seasonal means
    months = dates.month
    data = {}
    for s_idx, (site, shape, base_scale) in enumerate(
        [("siteA", 3, 100), ("siteB", 5, 80), ("siteC", 2, 150)]
    ):
        # Season-varying scale
        seasonal_scale = base_scale * (1 + 0.5 * np.sin(2 * np.pi * (months - 4) / 12))
        u = norm.cdf(z[:, s_idx])
        data[site] = gamma_dist.ppf(u, a=shape, scale=seasonal_scale)

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def monthly_single_site(monthly_multisite):
    """Single-site monthly data."""
    return monthly_multisite[["siteA"]]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestSPARTAInit:
    def test_default_params(self):
        gen = SPARTAGenerator()
        assert gen.nataf_method == "GH"
        assert gen.nataf_poly_deg == 6

    def test_custom_params(self):
        gen = SPARTAGenerator(nataf_method="MC", nataf_n_eval=11)
        assert gen.nataf_method == "MC"
        assert gen.nataf_n_eval == 11


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestSPARTAPreprocessing:
    def test_preprocessed_flag(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed

    def test_sites_stored(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen._n_sites == 3
        assert list(gen._sites) == ["siteA", "siteB", "siteC"]

    def test_monthly_data_shape(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen._Q_monthly.shape == (360, 3)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestSPARTAFit:
    def test_fitted_flag(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen.is_fitted

    def test_marginal_params_populated(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        # 12 months * 1 site = 12 entries
        assert len(gen._marginal_params) == 12
        for m in range(1, 13):
            assert (m, 0) in gen._marginal_params

    def test_equiv_auto_shape(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen._equiv_auto.shape == (1, 12)

    def test_multisite_fit(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert gen._equiv_auto.shape == (3, 12)
        assert len(gen._equiv_cross) == 12
        assert len(gen._A_s) == 12
        assert len(gen._B_s) == 12

    def test_fitted_params_returned(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        fp = gen._compute_fitted_params()
        assert fp.n_sites_ == 1
        assert fp.sample_size_ == 360


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestSPARTAGenerate:
    def test_generate_shape_univariate(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=2, n_years=20)
        assert isinstance(ens, Ensemble)
        assert len(ens.data_by_realization) == 2
        df = ens.data_by_realization[0]
        assert df.shape == (240, 1)

    def test_generate_shape_multivariate(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=10)
        df = ens.data_by_realization[0]
        assert df.shape == (120, 3)

    def test_generate_default_length(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 360  # matches observed

    def test_seed_reproducibility(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0],
            ens2.data_by_realization[0],
        )

    def test_different_seeds_differ(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=1)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=2)
        assert not np.allclose(
            ens1.data_by_realization[0].values,
            ens2.data_by_realization[0].values,
        )

    def test_output_has_datetime_index(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_years=5)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_output_columns(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["siteA", "siteB", "siteC"]

    def test_positive_values(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_years=50, seed=42)
        df = ens.data_by_realization[0]
        assert (df.values > 0).mean() > 0.95

    def test_n_timesteps(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_timesteps=100)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 100


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


class TestSPARTAStateValidation:
    def test_generate_before_fit_raises(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_single_site)
        with pytest.raises(Exception):
            gen.generate()

    def test_fit_auto_preprocesses(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen.is_preprocessed
        assert gen.is_fitted
