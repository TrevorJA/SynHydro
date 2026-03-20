"""Tests for the ARFIMA(p,d,q) Generator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.methods.generation.parametric.arfima import ARFIMAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_series():
    """20 years of monthly single-site data with seasonal pattern."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", periods=240, freq="MS")
    seasonal = 200 + 100 * np.sin(2 * np.pi * np.arange(240) / 12)
    noise = rng.gamma(2, 30, 240)
    vals = seasonal + noise
    return pd.Series(vals, index=dates, name="site_1")


@pytest.fixture
def annual_series():
    """50 years of annual data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1960-01-01", periods=50, freq="YS")
    vals = rng.gamma(3, 500, 50)
    return pd.Series(vals, index=dates, name="site_1")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestARFIMAInit:
    def test_default_params(self):
        gen = ARFIMAGenerator()
        assert gen.p == 1
        assert gen.q == 0
        assert gen.d_method == "whittle"
        assert gen.auto_order is False
        assert gen.is_fitted is False

    def test_custom_params(self):
        gen = ARFIMAGenerator(p=2, q=1, d_method="gph", auto_order=True)
        assert gen.p == 2
        assert gen.q == 1
        assert gen.d_method == "gph"
        assert gen.auto_order is True

    def test_stores_algorithm_params(self):
        gen = ARFIMAGenerator(p=2, q=1, auto_order=True)
        params = gen.init_params.algorithm_params
        assert params["p"] == 2
        assert params["q"] == 1
        assert params["auto_order"] is True


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestARFIMAPreprocessing:
    def test_preprocessing_monthly(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.preprocessing(monthly_series)
        assert gen.is_preprocessed
        assert gen._is_monthly is True

    def test_preprocessing_annual(self, annual_series):
        gen = ARFIMAGenerator(deseasonalize=False)
        gen.preprocessing(annual_series)
        assert gen.is_preprocessed
        assert gen._is_monthly is False

    def test_preprocessing_deseasonalization(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.preprocessing(monthly_series)
        assert gen.seasonal_params is not None
        assert "means" in gen.seasonal_params
        assert "stds" in gen.seasonal_params

    def test_preprocessing_dataframe_single_column(self, monthly_series):
        df = monthly_series.to_frame()
        gen = ARFIMAGenerator()
        gen.preprocessing(df)
        assert gen.is_preprocessed


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestARFIMAFit:
    def test_fit_basic(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        assert gen.is_fitted
        assert 0.01 <= gen.d <= 0.49

    def test_fit_with_q_obs(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        assert gen.is_fitted

    def test_fit_without_preprocessing_raises(self):
        gen = ARFIMAGenerator()
        with pytest.raises(Exception):
            gen.fit()

    def test_fit_d_estimation_whittle(self, monthly_series):
        gen = ARFIMAGenerator(d_method="whittle")
        gen.fit(monthly_series)
        assert 0.01 <= gen.d <= 0.49

    def test_fit_d_estimation_gph(self, monthly_series):
        gen = ARFIMAGenerator(d_method="gph")
        gen.fit(monthly_series)
        assert 0.01 <= gen.d <= 0.49

    def test_fit_d_estimation_rs(self, monthly_series):
        gen = ARFIMAGenerator(d_method="rs")
        gen.fit(monthly_series)
        assert 0.01 <= gen.d <= 0.49

    def test_fit_ar_only(self, monthly_series):
        gen = ARFIMAGenerator(p=1, q=0)
        gen.fit(monthly_series)
        assert len(gen.phi) == 1
        assert len(gen.theta) == 0

    def test_fit_arma_q1(self, monthly_series):
        gen = ARFIMAGenerator(p=1, q=1)
        gen.fit(monthly_series)
        assert len(gen.phi) == 1
        assert len(gen.theta) == 1
        assert gen.sigma_eps_sq > 0

    def test_fit_arma_q2(self, monthly_series):
        gen = ARFIMAGenerator(p=1, q=2)
        gen.fit(monthly_series)
        assert len(gen.phi) == 1
        assert len(gen.theta) == 2

    def test_fit_creates_fitted_params(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        fp = gen.get_fitted_params()
        assert fp is not None
        assert fp["n_parameters_"] > 0

    def test_fit_auto_order_bic(self, monthly_series):
        gen = ARFIMAGenerator(auto_order=True)
        gen.fit(monthly_series)
        assert gen.is_fitted
        # BIC should have selected some order
        assert gen.p >= 0
        assert gen.q >= 0
        assert gen.p <= 2
        assert gen.q <= 2

    def test_fit_annual(self, annual_series):
        gen = ARFIMAGenerator(deseasonalize=False)
        gen.fit(annual_series)
        assert gen.is_fitted
        assert 0.01 <= gen.d <= 0.49


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestARFIMAGeneration:
    def test_generate_basic_monthly(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_basic_annual(self, annual_series):
        gen = ARFIMAGenerator(deseasonalize=False)
        gen.fit(annual_series)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_shape_monthly(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape == (60, 1)  # 5 * 12 months, 1 site

    def test_generate_shape_annual(self, annual_series):
        gen = ARFIMAGenerator(deseasonalize=False)
        gen.fit(annual_series)
        ens = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape == (10, 1)

    def test_generate_non_negative(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=5, n_years=20, seed=42)
        for r, df in ens.data_by_realization.items():
            assert (df.values >= 0).all(), f"Realization {r} has negative values"

    def test_generate_reproducible_with_seed(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0], ens2.data_by_realization[0]
        )

    def test_generate_without_fit_raises(self):
        gen = ARFIMAGenerator()
        with pytest.raises(Exception):
            gen.generate(n_realizations=1, n_years=5)

    def test_generate_with_ma_component(self, monthly_series):
        gen = ARFIMAGenerator(p=1, q=1)
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=3, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 3
        for df in ens.data_by_realization.values():
            assert (df.values >= 0).all()

    def test_generate_has_datetime_index(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_with_auto_order(self, monthly_series):
        gen = ARFIMAGenerator(auto_order=True)
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestARFIMAStatisticalProperties:
    def test_mean_preserved(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=30, n_years=20, seed=42)

        obs_mean = monthly_series.mean()
        syn_means = [df.values.mean() for df in ens.data_by_realization.values()]
        ensemble_mean = np.mean(syn_means)
        ratio = ensemble_mean / obs_mean
        assert (
            0.5 < ratio < 2.0
        ), f"Mean not preserved: obs={obs_mean:.1f}, syn={ensemble_mean:.1f}"

    def test_std_preserved(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=30, n_years=20, seed=42)

        obs_std = monthly_series.std()
        syn_stds = [df.values.std() for df in ens.data_by_realization.values()]
        ensemble_std = np.mean(syn_stds)
        ratio = ensemble_std / obs_std
        assert (
            0.3 < ratio < 3.0
        ), f"Std not preserved: obs={obs_std:.1f}, syn={ensemble_std:.1f}"


# ---------------------------------------------------------------------------
# CSS residuals
# ---------------------------------------------------------------------------


class TestCSSResiduals:
    def test_pure_ar_residuals(self):
        # Known AR(1) process: W_t = 0.5 * W_{t-1} + eps_t
        rng = np.random.default_rng(42)
        n = 200
        eps_true = rng.normal(0, 1, n)
        W = np.zeros(n)
        for t in range(1, n):
            W[t] = 0.5 * W[t - 1] + eps_true[t]

        phi = np.array([0.5])
        theta = np.array([])
        eps_recovered = ARFIMAGenerator._compute_css_residuals(W, phi, theta)
        # After burn-in, residuals should match true innovations
        np.testing.assert_allclose(eps_recovered[1:], eps_true[1:], atol=1e-10)

    def test_empty_ar_ma(self):
        W = np.array([1.0, 2.0, 3.0])
        eps = ARFIMAGenerator._compute_css_residuals(W, np.array([]), np.array([]))
        np.testing.assert_array_equal(eps, W)

    def test_arma_residuals_shape(self):
        rng = np.random.default_rng(42)
        W = rng.normal(0, 1, 100)
        phi = np.array([0.3])
        theta = np.array([0.2])
        eps = ARFIMAGenerator._compute_css_residuals(W, phi, theta)
        assert len(eps) == 100


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestARFIMASerialization:
    def test_pickle_save_load(self, monthly_series, tmp_path):
        gen = ARFIMAGenerator(p=1, q=1)
        gen.fit(monthly_series)

        filepath = tmp_path / "arfima_gen.pkl"
        gen.save(str(filepath))

        loaded = ARFIMAGenerator.load(str(filepath))
        assert loaded.is_fitted
        assert loaded.p == 1
        assert loaded.q == 1
        assert len(loaded.theta) == 1

    def test_pickle_generate_after_load(self, monthly_series, tmp_path):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)

        filepath = tmp_path / "arfima_gen.pkl"
        gen.save(str(filepath))
        loaded = ARFIMAGenerator.load(str(filepath))

        ens = loaded.generate(n_realizations=1, n_years=5, seed=42)
        assert len(ens.data_by_realization) == 1
