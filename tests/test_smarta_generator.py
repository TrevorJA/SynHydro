"""Tests for the SMARTAGenerator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.methods.generation.parametric.smarta import SMARTAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def annual_multisite():
    """Synthetic annual data: 80 years, 3 correlated sites."""
    rng = np.random.default_rng(42)
    n_years = 80
    dates = pd.date_range("1940-01-01", periods=n_years, freq="YS")

    # Correlated gamma-distributed flows
    z = rng.multivariate_normal(
        [0, 0, 0],
        [[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]],
        size=n_years,
    )
    from scipy.stats import gamma as gamma_dist, norm

    u = norm.cdf(z)
    site_a = gamma_dist.ppf(u[:, 0], a=3, scale=100)
    site_b = gamma_dist.ppf(u[:, 1], a=5, scale=80)
    site_c = gamma_dist.ppf(u[:, 2], a=2, scale=150)

    return pd.DataFrame(
        {"siteA": site_a, "siteB": site_b, "siteC": site_c},
        index=dates,
    )


@pytest.fixture
def annual_single_site(annual_multisite):
    """Single-site annual data."""
    return annual_multisite[["siteA"]]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestSMARTAInit:
    def test_default_params(self):
        gen = SMARTAGenerator()
        assert gen.acf_model == "cas"
        assert gen.sma_order == 512
        assert gen.nataf_method == "GH"

    def test_custom_sma_order(self):
        gen = SMARTAGenerator(sma_order=64)
        assert gen.sma_order == 64

    def test_stores_kwargs(self):
        gen = SMARTAGenerator(nataf_method="MC", nataf_n_eval=11)
        assert gen.nataf_method == "MC"
        assert gen.nataf_n_eval == 11


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestSMARTAPreprocessing:
    def test_preprocessed_flag(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen.is_preprocessed

    def test_sites_stored(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen._n_sites == 3
        assert list(gen._sites) == ["siteA", "siteB", "siteC"]

    def test_annual_data_shape(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen._Q_annual.shape == (80, 3)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestSMARTAFit:
    def test_fitted_flag(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen.is_fitted

    def test_marginal_params_populated(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert len(gen._marginal_params) == 3
        for s_idx in range(3):
            assert "dist" in gen._marginal_params[s_idx]

    def test_sma_weights_shape(self, annual_multisite):
        q = 64
        gen = SMARTAGenerator(sma_order=q)
        gen.fit(annual_multisite)
        assert len(gen._sma_weights) == 3
        for w in gen._sma_weights:
            assert len(w) == 2 * q + 1

    def test_b_tilde_shape(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen._B_tilde.shape == (3, 3)

    def test_cas_params_stored(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert len(gen._cas_params) == 3

    def test_single_site_fit(self, annual_single_site):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_single_site)
        assert gen.is_fitted
        assert gen._B_tilde.shape == (1, 1)

    def test_fitted_params_returned(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        fp = gen._compute_fitted_params()
        assert fp.n_sites_ == 3
        assert fp.sample_size_ == 80


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestSMARTAGenerate:
    def test_generate_shape(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=2, n_years=50)
        assert isinstance(ens, Ensemble)
        assert len(ens.data_by_realization) == 2
        df = ens.data_by_realization[0]
        assert df.shape == (50, 3)

    def test_generate_default_length(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 80  # matches observed length

    def test_seed_reproducibility(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=30, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=30, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0],
            ens2.data_by_realization[0],
        )

    def test_different_seeds_differ(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=30, seed=1)
        ens2 = gen.generate(n_realizations=1, n_years=30, seed=2)
        assert not np.allclose(
            ens1.data_by_realization[0].values,
            ens2.data_by_realization[0].values,
        )

    def test_output_has_correct_columns(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=20)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["siteA", "siteB", "siteC"]

    def test_output_has_datetime_index(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=20)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_single_site_generate(self, annual_single_site):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_single_site)
        ens = gen.generate(n_realizations=1, n_years=30)
        df = ens.data_by_realization[0]
        assert df.shape == (30, 1)

    def test_positive_values(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=100, seed=42)
        df = ens.data_by_realization[0]
        # Most values should be positive (gamma/lognorm marginals)
        assert (df.values > 0).mean() > 0.95


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


class TestSMARTAStateValidation:
    def test_generate_before_fit_raises(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        with pytest.raises(Exception):
            gen.generate()

    def test_fit_auto_preprocesses(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen.is_preprocessed
        assert gen.is_fitted
