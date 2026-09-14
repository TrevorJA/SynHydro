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
def monthly_series_high_cv():
    """20 years of high-CV, high-skew monthly data (drought-prone basin).

    Realistic streamflow regime: lognormal-distributed flow with strong
    seasonal multiplier (winter low / spring peak). CV ~ 1.5, skew > 4.
    This is the regime where the pre-fix truncation bug was visible.
    """
    rng = np.random.default_rng(11)
    dates = pd.date_range("1990-01-01", periods=240, freq="MS")
    months = dates.month
    season = np.array([0.5, 0.4, 0.9, 2.0, 4.0, 3.0, 1.2, 0.4, 0.2, 0.3, 0.5, 0.4])
    vals = 50 * season[months - 1] * np.exp(rng.normal(0, 1.0, 240))
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
        assert gen.d_method == "mle"
        assert gen.auto_order is False
        assert gen.order_criterion == "aic"
        assert gen.backcast_length == 30
        assert gen.d_bounds == (-0.49, 0.49)
        assert gen.is_fitted is False

    def test_invalid_options_raise(self):
        with pytest.raises(ValueError):
            ARFIMAGenerator(d_method="nope")
        with pytest.raises(ValueError):
            ARFIMAGenerator(order_criterion="hqic")
        with pytest.raises(ValueError):
            ARFIMAGenerator(d_bounds=(0.3, 0.1))

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
        assert params["order_criterion"] == "aic"
        assert params["backcast_length"] == 30


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
        gen = ARFIMAGenerator()
        gen.preprocessing(annual_series)
        assert gen.is_preprocessed
        assert gen._is_monthly is False

    def test_preprocessing_stedinger_transform_fitted(self, monthly_series):
        """Preprocessing must fit the shifted-lognormal and z-score transforms."""
        gen = ARFIMAGenerator()
        gen.preprocessing(monthly_series)
        assert gen.log_transform.is_fitted
        assert gen.scaler.is_fitted

        tau = gen.log_transform.params_["tau"]
        # Per-month tau: DataFrame indexed by month (1-12), one column per site.
        assert tau.shape == (12, 1)
        assert (tau.values >= 0).all(), "Stedinger tau must be non-negative"

        # Per-month z-score statistics from the StandardScaler stage.
        assert "mean" in gen.scaler.params_
        assert "std" in gen.scaler.params_

    def test_preprocessing_q_norm_is_standardized(self, monthly_series):
        """After both transforms, Q_norm should be approximately mean=0, std=1 by month."""
        gen = ARFIMAGenerator()
        gen.preprocessing(monthly_series)
        by_month_mean = gen.Q_norm.groupby(gen.Q_norm.index.month).mean()
        by_month_std = gen.Q_norm.groupby(gen.Q_norm.index.month).std()
        assert np.allclose(by_month_mean.values, 0.0, atol=1e-8)
        assert np.allclose(by_month_std.values, 1.0, atol=1e-8)

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
        assert -0.49 <= gen.d <= 0.49

    def test_fit_d_bounds_respected(self, monthly_series):
        gen = ARFIMAGenerator(d_bounds=(0.01, 0.49))
        gen.fit(monthly_series)
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

    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_fit_auto_order(self, monthly_series, criterion):
        gen = ARFIMAGenerator(auto_order=True, order_criterion=criterion)
        gen.fit(monthly_series)
        assert gen.is_fitted
        assert 0 <= gen.p <= 2
        assert 0 <= gen.q <= 2
        assert len(gen.phi) == gen.p
        assert len(gen.theta) == gen.q
        assert (gen.p, gen.q) in gen._joint_fit_cache

    def test_fit_auto_order_two_stage(self, monthly_series):
        gen = ARFIMAGenerator(auto_order=True, d_method="whittle")
        gen.fit(monthly_series)
        assert 0 <= gen.p <= 2
        assert 0 <= gen.q <= 2

    def test_fit_annual(self, annual_series):
        gen = ARFIMAGenerator()
        gen.fit(annual_series)
        assert gen.is_fitted
        assert -0.49 <= gen.d <= 0.49


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
        gen = ARFIMAGenerator()
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
        gen = ARFIMAGenerator()
        gen.fit(annual_series)
        ens = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape == (10, 1)

    def test_generate_strictly_positive(self, monthly_series):
        """The Stedinger back-transform guarantees Q = tau + exp(Y) > tau >= 0,
        so every synthetic value must be strictly positive."""
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=5, n_years=20, seed=42)
        for r, df in ens.data_by_realization.items():
            assert (df.values > 0).all(), f"Realization {r} has non-positive values"

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
            assert (df.values > 0).all()

    def test_no_truncation_at_zero_high_cv(self, monthly_series_high_cv):
        """Regression test for the np.maximum(X, 0) truncation bug.

        Before the Stedinger-transform fix, ~15% of synthetic values on a
        realistic high-CV basin were clipped to exact zero. With the shifted-
        lognormal transformation, no value should land at or near zero.
        """
        gen = ARFIMAGenerator(p=1, q=0)
        gen.fit(monthly_series_high_cv)
        ens = gen.generate(n_realizations=5, n_years=20, seed=789)
        arr = np.concatenate(
            [df.values.ravel() for df in ens.data_by_realization.values()]
        )
        n_zero = int((arr == 0).sum())
        n_near_zero = int((arr < 1e-6).sum())
        assert n_zero == 0, f"Found {n_zero} exact-zero values (truncation bug)"
        assert n_near_zero / arr.size < 0.001, (
            f"Found {n_near_zero}/{arr.size} near-zero values "
            f"({100 * n_near_zero / arr.size:.2f}%)"
        )

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

    def test_generate_long_annual_run_second_resolution(self, annual_series):
        """Annual output beyond year 2262 uses a datetime64[s] index."""
        gen = ARFIMAGenerator()
        gen.fit(annual_series)
        ens = gen.generate(n_realizations=1, n_years=5000, seed=3)
        df = ens.data_by_realization[0]
        assert len(df) == 5000
        assert df.index.dtype == "datetime64[s]"
        assert df.index[0] == pd.Timestamp("2010-01-01")
        assert df.index[-1].year == 2010 + 4999
        assert (df.values > 0).all()

    def test_generate_monthly_index_continues_record(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        df = gen.generate(n_realizations=1, n_years=2, seed=3).data_by_realization[0]
        expected = pd.date_range("2010-01-01", periods=24, freq="MS")
        assert (df.index.values == expected.values).all()


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

    def test_mean_preserved_high_cv(self, monthly_series_high_cv):
        """Mean must be preserved on high-CV data -- the pre-fix truncation
        biased the mean upward via the np.maximum(X, 0) clip."""
        gen = ARFIMAGenerator(p=1, q=0)
        gen.fit(monthly_series_high_cv)
        ens = gen.generate(n_realizations=30, n_years=20, seed=42)

        obs_mean = monthly_series_high_cv.mean()
        syn_means = [df.values.mean() for df in ens.data_by_realization.values()]
        ensemble_mean = np.mean(syn_means)
        ratio = ensemble_mean / obs_mean
        assert (
            0.85 < ratio < 1.20
        ), f"Mean not preserved on high-CV data: obs={obs_mean:.1f}, syn={ensemble_mean:.1f}, ratio={ratio:.3f}"


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


# ---------------------------------------------------------------------------
# Whittle estimator and burn-in (AUDIT.md follow-up item 6)
# ---------------------------------------------------------------------------


def _exact_arfima_0d0(d: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate an exact ARFIMA(0,d,0) series via Cholesky of Hosking's ACF."""
    from scipy.linalg import cholesky, toeplitz

    rho = np.ones(n)
    for k in range(1, n):
        rho[k] = rho[k - 1] * (k - 1 + d) / (k - d)
    L = cholesky(toeplitz(rho), lower=True)
    return L @ rng.standard_normal(n)


class TestWhittleEstimator:
    @pytest.mark.parametrize("d_true", [0.2, 0.35])
    def test_profile_whittle_unbiased(self, d_true):
        """Mean d_hat over exact ARFIMA(0,d,0) replicates is close to d.

        The non-profiled objective (scale fixed at 1) gave ~0.24 / ~0.40
        for d = 0.2 / 0.35 on z-scored input.
        """
        rng = np.random.default_rng(123)
        n = 600
        idx = pd.date_range("1950-01-01", periods=n, freq="MS")
        d_hats = []
        for _ in range(12):
            x = _exact_arfima_0d0(d_true, n, rng)
            q = pd.Series(np.exp(x), index=idx, name="site_1")
            gen = ARFIMAGenerator(p=0, q=0, d_method="whittle")
            gen.fit(q)
            d_hats.append(gen.d)
        assert abs(np.mean(d_hats) - d_true) < 0.03

    def test_profile_whittle_scale_invariant(self):
        """The profile form does not depend on the scale of Q_norm."""
        rng = np.random.default_rng(5)
        x = _exact_arfima_0d0(0.3, 400, rng)
        gen = ARFIMAGenerator(p=0, q=0)
        gen.Q_norm = pd.Series(x)
        d_unit = gen._whittle_estimator()
        gen.Q_norm = pd.Series(7.0 * x)
        d_scaled = gen._whittle_estimator()
        assert abs(d_unit - d_scaled) < 1e-4


class TestBurnIn:
    def test_burn_in_at_least_truncation_lag(self):
        gen = ARFIMAGenerator(truncation_lag=250)
        assert gen._burn_in_length() >= 250

    def test_first_step_has_full_variance(self):
        """Without burn-in, X_0 = eps_0 has variance sigma_eps^2 instead of
        sigma_eps^2 * sum(psi_k^2); with burn-in the first output step is
        already in the (truncated) stationary regime.

        Checks var(X_0) / var(X_12) (same calendar month, so the per-month
        scaling cancels); without burn-in this ratio is ~0.6 at d = 0.35.
        """
        rng = np.random.default_rng(21)
        n = 480
        idx = pd.date_range("1960-01-01", periods=n, freq="MS")
        x = _exact_arfima_0d0(0.35, n, rng)
        series = pd.Series(np.exp(x), index=idx, name="site_1")
        gen = ARFIMAGenerator(p=0, q=0)
        gen.fit(series)
        assert gen.d > 0.25
        tau = gen.log_transform.params_["tau"]
        month0 = gen.Q_obs.index[-1].month % 12 + 1
        tau0 = float(tau.loc[month0, "site_1"])
        q = np.array(
            [
                gen._generate_single(13, rng=np.random.default_rng(s)).values
                for s in range(500)
            ]
        )
        v0 = np.var(np.log(q[:, 0] - tau0))
        v12 = np.var(np.log(q[:, 12] - tau0))
        assert v0 / v12 == pytest.approx(1.0, abs=0.15)

    def test_seed_reproducible_with_burn_in(self, monthly_series):
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        a = gen.generate(n_realizations=2, n_years=5, seed=9)
        b = gen.generate(n_realizations=2, n_years=5, seed=9)
        for i in range(2):
            np.testing.assert_allclose(
                a.data_by_realization[i].values, b.data_by_realization[i].values
            )


# ---------------------------------------------------------------------------
# Joint approximate maximum likelihood (Hosking 1984, Sec. 4.2)
# ---------------------------------------------------------------------------


def _arma_acov(phi, theta, nlags, length=4000):
    """Autocovariance of a unit-variance ARMA via its psi weights."""
    from scipy.signal import lfilter

    impulse = np.zeros(length)
    impulse[0] = 1.0
    psi = lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], impulse)
    return np.array([np.sum(psi[: length - k] * psi[k:]) for k in range(nlags)])


def _exact_arfima(d, phi, theta, n, rng, tail=2000):
    """Exact ARFIMA(p,d,q) sample: Hosking (1984) Eq. 3 convolution + Cholesky."""
    from scipy.linalg import cholesky, toeplitz
    from scipy.special import gamma

    gw = _arma_acov(np.atleast_1d(phi), np.atleast_1d(theta), tail)
    gx = np.zeros(n + tail)
    gx[0] = gamma(1 - 2 * d) / gamma(1 - d) ** 2
    for k in range(1, n + tail):
        gx[k] = gx[k - 1] * (k - 1 + d) / (k - d)
    js = np.arange(-tail + 1, tail)
    g = np.array([np.sum(gw[np.abs(js)] * gx[np.abs(k + js)]) for k in range(n)])
    L = cholesky(toeplitz(g), lower=True)
    return L @ rng.standard_normal(n)


class TestJointEstimator:
    def test_backcast_uses_ar_structure(self):
        """Backcasts of an AR(1) series extrapolate x_0 with the AR coefficient."""
        rng = np.random.default_rng(0)
        x = _exact_arfima(0.0, [0.8], [], 500, rng)
        x = x - x.mean()
        gen = ARFIMAGenerator(backcast_length=30)
        pre = gen._backcast_presample(x)
        assert pre.shape == (30,)
        # the last backcast value is x_{-1}, which should be close to 0.8 * x_0
        assert abs(pre[-1] - 0.8 * x[0]) < 0.6
        gen0 = ARFIMAGenerator(backcast_length=0)
        assert gen0._backcast_presample(x).shape == (0,)

    def test_backcast_length_capped_for_short_records(self):
        gen = ARFIMAGenerator(backcast_length=30)
        x = np.random.default_rng(1).normal(size=40)
        assert gen._backcast_presample(x).shape == (10,)

    def test_fractional_difference_matches_truncated_filter(self):
        """With M = 0 and no truncation, Eq. 9 reduces to Eq. 8."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=80)
        d = 0.3
        gen = ARFIMAGenerator(truncation_lag=200)
        gen.Q_norm = pd.Series(x)
        gen.pi_coeffs = gen._compute_fractional_diff_coefficients(d)
        w_eq8 = gen._apply_fractional_differencing().values
        w_eq9 = gen._fractional_difference_backcast(x, d, len(x))
        np.testing.assert_allclose(w_eq9, w_eq8, atol=1e-10)

    def test_arma_innovations_match_recursion(self):
        rng = np.random.default_rng(3)
        w = rng.normal(size=50)
        phi = np.array([0.5, -0.2])
        theta = np.array([0.3])
        np.testing.assert_allclose(
            ARFIMAGenerator._arma_innovations(w, phi, theta),
            ARFIMAGenerator._compute_css_residuals(w, phi, theta),
            atol=1e-12,
        )

    def test_arma_admissibility(self):
        ok = ARFIMAGenerator._arma_is_admissible
        assert ok(np.array([0.5]), np.array([]))
        assert ok(np.array([]), np.array([-0.9]))
        assert not ok(np.array([1.2]), np.array([]))
        assert not ok(np.array([0.9, 0.9]), np.array([]))

    @pytest.mark.parametrize(
        "p, q, d_true, coef",
        [(1, 0, 0.35, -0.3), (0, 1, 0.2, 0.5), (0, 1, 0.35, -0.3)],
    )
    def test_joint_removes_two_stage_contamination(self, p, q, d_true, coef):
        """Joint estimates are close to truth where the two-stage Whittle
        estimate of d is badly contaminated by the ARMA part.

        Two-stage (Whittle then ARMA) at n = 600, 30 reps:
            ARFIMA(1,0.35,0) phi=-0.3:   d_hat 0.17, phi_hat -0.13
            ARFIMA(0,0.20,1) theta=0.5:  d_hat 0.47, theta_hat 0.30
            ARFIMA(0,0.35,1) theta=-0.3: d_hat 0.15, theta_hat -0.09
        Joint approximate ML:
            d_hat 0.33 / 0.19 / 0.32, coef -0.29 / 0.49 / -0.28
        """
        rng = np.random.default_rng(77)
        n = 600
        idx = pd.date_range("1950-01-01", periods=n, freq="MS")
        d_joint, c_joint, d_two = [], [], []
        for _ in range(8):
            x = _exact_arfima(d_true, [coef] if p else [], [coef] if q else [], n, rng)
            series = pd.Series(np.exp(x), index=idx, name="site_1")
            gen = ARFIMAGenerator(p=p, q=q)
            gen.fit(series)
            d_joint.append(gen.d)
            c_joint.append(gen.phi[0] if p else gen.theta[0])
            two = ARFIMAGenerator(p=p, q=q, d_method="whittle")
            two.fit(series)
            d_two.append(two.d)
        assert abs(np.mean(d_joint) - d_true) < 0.06
        assert abs(np.mean(c_joint) - coef) < 0.08
        assert abs(np.mean(d_two) - d_true) > 0.1

    def test_joint_agrees_with_profile_whittle_when_white(self):
        """For ARFIMA(0,d,0) the joint CSS estimate agrees with Whittle."""
        rng = np.random.default_rng(11)
        n = 600
        idx = pd.date_range("1950-01-01", periods=n, freq="MS")
        diffs = []
        for _ in range(6):
            x = _exact_arfima_0d0(0.3, n, rng)
            series = pd.Series(np.exp(x), index=idx, name="site_1")
            a = ARFIMAGenerator(p=0, q=0)
            a.fit(series)
            b = ARFIMAGenerator(p=0, q=0, d_method="whittle")
            b.fit(series)
            diffs.append(a.d - b.d)
        assert abs(np.mean(diffs)) < 0.05

    def test_joint_fit_sets_expected_attributes(self, monthly_series):
        gen = ARFIMAGenerator(p=1, q=1)
        gen.fit(monthly_series)
        assert len(gen.W) == len(monthly_series)
        assert len(gen.phi) == 1 and len(gen.theta) == 1
        assert gen.sigma_eps_sq > 0
        assert gen._arma_is_admissible(gen.phi, gen.theta)
        assert len(gen.pi_coeffs) == gen.truncation_lag + 1

    def test_joint_fit_deterministic(self, monthly_series):
        a = ARFIMAGenerator()
        a.fit(monthly_series)
        b = ARFIMAGenerator()
        b.fit(monthly_series)
        assert a.d == b.d
        np.testing.assert_array_equal(a.phi, b.phi)


# ---------------------------------------------------------------------------
# Generated output: fitted d, Hosking ACF and moments
# ---------------------------------------------------------------------------


def _hosking_acf(d: float, nlags: int) -> np.ndarray:
    """Theoretical ACF of ARFIMA(0,d,0) at lags 0..nlags (Hosking, 1981):
    rho(k) = prod_{j=1..k} (j - 1 + d) / (j - d)."""
    rho = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        rho[k] = rho[k - 1] * (k - 1 + d) / (k - d)
    return rho


def _acf_about_zero(x: np.ndarray, lags) -> np.ndarray:
    """Biased (1/n) sample ACF taken about the known process mean of zero."""
    denom = np.sum(x * x)
    return np.array([np.sum(x[:-k] * x[k:]) / denom for k in lags])


class TestARFIMAGeneratedOutput:
    """Statistical regression tests on the output of ``generate``.

    The estimator tests above check that the fit recovers d from exact
    ARFIMA samples.  These check the other direction: that flows produced
    by the generator carry the fitted d, Hosking's (1981) ACF and the
    fitted Gaussian-domain variance.

    Input is an exact ARFIMA(0, 0.3, 0) sample of 600 months, exponentiated
    to positive flows as in ``TestWhittleEstimator``.  The generator uses
    the profile Whittle estimator (d_fit = 0.279 for this seed) and an
    inverse filter of K = 300 lags, which retains about 98 percent of the
    process variance at this d (about 97 percent at the default K = 100).
    Six 150-year realizations are generated once per class.
    """

    LAGS = (1, 2, 5, 10, 20)

    @pytest.fixture(scope="class")
    def fitted(self):
        rng = np.random.default_rng(0)
        n = 600
        idx = pd.date_range("1950-01-01", periods=n, freq="MS")
        series = pd.Series(
            np.exp(_exact_arfima_0d0(0.3, n, rng)), index=idx, name="site_1"
        )
        gen = ARFIMAGenerator(p=0, q=0, d_method="whittle", truncation_lag=300)
        gen.fit(series)
        ens = gen.generate(n_realizations=6, n_years=150, seed=0)
        return series, gen, ens

    @staticmethod
    def _to_gaussian(gen: ARFIMAGenerator, flows: pd.Series) -> np.ndarray:
        """Map generated flows back to the domain of the ARFIMA process.

        Applies the forward transforms of ``preprocessing`` (per-month
        Stedinger log, then per-month z-score), which invert the last two
        steps of ``_generate_single`` exactly.
        """
        y = gen.scaler.transform(gen.log_transform.transform(flows))
        return np.asarray(y.values, dtype=float)

    def test_generated_series_recovers_d(self, fitted):
        """Refitting Whittle on each realization recovers the fitted d.

        Measured (seed 0): d_fit = 0.2791, d_hat = [0.286, 0.288, 0.278,
        0.320, 0.281, 0.295], mean 0.2915 (bias +0.012, max deviation
        0.041).  The bias is not a truncation effect: it is +0.016 at
        K = 100 and +0.010 at K = 1000.  The asymptotic Whittle standard
        error at n = 1800 is sqrt(6 / pi^2 / n) = 0.018 per realization.
        """
        _, gen, ens = fitted
        d_fit = gen.d
        d_hats = []
        for df in ens.data_by_realization.values():
            refit = ARFIMAGenerator(p=0, q=0, d_method="whittle")
            refit.fit(df.iloc[:, 0])
            d_hats.append(refit.d)
        d_hats = np.array(d_hats)
        assert abs(d_hats.mean() - d_fit) < 0.05, (
            f"mean d_hat {d_hats.mean():.4f} vs d_fit {d_fit:.4f}; "
            f"d_hat = {np.round(d_hats, 4).tolist()}"
        )
        assert np.all(
            np.abs(d_hats - d_fit) < 0.12
        ), f"d_hat = {np.round(d_hats, 4).tolist()} vs d_fit {d_fit:.4f}"

    def test_generated_acf_matches_hosking(self, fitted):
        """Ensemble-mean ACF of the Gaussian-domain output matches Hosking.

        The ARFIMA process X_t = sum_k psi_k eps_{t-k} is zero-mean by
        construction, so the ACF is taken about zero.  The usual sample ACF
        about the sample mean is biased low by about var(xbar) / gamma(0)
        for a long-memory series (var(xbar) / gamma(0) is 0.013 here at
        n = 1800 and K = 300, and 0.046 for the untruncated process; Beran,
        1994), an estimator artifact unrelated to the generator.  The K = 300
        truncation lowers the exact ACF by 0.009 (lag 1) to 0.013 (lag 20).

        Measured (seed 0), lags (1, 2, 5, 10, 20):
            Hosking exact  0.387 0.288 0.193 0.142 0.105
            ensemble mean  0.393 0.298 0.180 0.146 0.109
        A short-memory process with the same lag-1 correlation would have
        rho(20) = 0.39**20 ~ 1e-8, so the lag-20 floor of half the Hosking
        value cannot be met without long memory.
        """
        _, gen, ens = fitted
        lags = list(self.LAGS)
        acf = np.mean(
            [
                _acf_about_zero(self._to_gaussian(gen, df.iloc[:, 0]), lags)
                for df in ens.data_by_realization.values()
            ],
            axis=0,
        )
        theory = _hosking_acf(gen.d, max(lags))[lags]
        assert np.max(np.abs(acf - theory)) < 0.05, (
            f"lags {lags}: ensemble ACF {np.round(acf, 4).tolist()} vs "
            f"Hosking {np.round(theory, 4).tolist()}"
        )
        assert acf[-1] > 0.5 * theory[-1], (
            f"lag-20 ACF {acf[-1]:.4f} below half the Hosking value "
            f"{theory[-1]:.4f}: long memory not carried into the output"
        )

    def test_generated_variance_matches_fit(self, fitted):
        """Pooled Gaussian-domain output has the fitted mean and variance.

        var(X) = sigma_eps^2 * sum_{k<=K} psi_k^2, which for the fitted
        sigma_eps^2 should reproduce var(Q_norm) = 1 up to the 2 percent
        truncation deficit; this is the check on sigma_eps^2 and the filter
        gain that the (normalized) ACF test cannot make.  The pooled mean
        over 6 x 1800 values has standard deviation sqrt(v / 6) = 0.046,
        with v = var(xbar) / gamma(0) = 0.013 for the K = 300 truncated
        process (0.046 untruncated), so 0.25 is over 5 sigma.
        Measured: mean +0.03, std 1.02.
        """
        _, gen, ens = fitted
        x = np.concatenate(
            [
                self._to_gaussian(gen, df.iloc[:, 0])
                for df in ens.data_by_realization.values()
            ]
        )
        assert abs(x.mean()) < 0.25, f"pooled Gaussian-domain mean {x.mean():+.4f}"
        assert abs(x.std() - 1.0) < 0.10, f"pooled Gaussian-domain std {x.std():.4f}"

    def test_generated_moments_match_input(self, monthly_series):
        """Pooled flow mean and std match the input to 10 / 20 percent.

        Pooled version of ``TestARFIMAStatisticalProperties`` (same fixture,
        generator and ensemble size; those tests allow factors of 2 to 3 per
        realization).  Measured: mean ratio 1.0001, std ratio 1.004.

        The long-memory ensemble above is not used here: its input is a
        sigma = 1 lognormal, whose sample std at n = 600 has a sampling
        standard deviation of 25 to 30 percent (kurtosis above 100, plus
        long memory), so its measured pooled ratios of 1.08 (mean) and
        1.28 (std) say nothing about the generator.
        """
        gen = ARFIMAGenerator()
        gen.fit(monthly_series)
        ens = gen.generate(n_realizations=30, n_years=20, seed=42)
        arr = np.concatenate(
            [df.values.ravel() for df in ens.data_by_realization.values()]
        )
        mean_ratio = arr.mean() / monthly_series.mean()
        std_ratio = arr.std() / monthly_series.std()
        assert abs(mean_ratio - 1.0) < 0.10, f"pooled mean ratio {mean_ratio:.4f}"
        assert abs(std_ratio - 1.0) < 0.20, f"pooled std ratio {std_ratio:.4f}"
