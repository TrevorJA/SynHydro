"""Tests for the SMARTAGenerator."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

from synhydro.core.ensemble import Ensemble
from synhydro.core.nataf import hurst_acf
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


# ---------------------------------------------------------------------------
# Long runs (beyond the pandas nanosecond year-2262 limit)
# ---------------------------------------------------------------------------


class TestSMARTALongRuns:
    def test_generate_1000_years(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=1000, seed=0)
        df = ens.data_by_realization[0]
        assert df.shape == (1000, 3)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].year == 1940
        assert df.index[-1].year == 1940 + 999
        assert df.index.is_monotonic_increasing
        assert len(set(df.index.year)) == 1000
        assert np.all(np.isfinite(df.values))
        assert ens.metadata.time_period == ("1940-01-01", "2939-01-01")

    def test_long_run_to_hdf5(self, annual_multisite, tmp_path):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=2, n_years=1200, seed=0)
        fn = tmp_path / "smarta_long.h5"
        ens.to_hdf5(str(fn))
        assert fn.exists()

    def test_short_run_index_values_unchanged(self, annual_multisite):
        """Calendar values match the observed index; dtype is datetime64[s]."""
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        df = gen.generate(n_years=80, seed=0).data_by_realization[0]
        assert (df.index.values == annual_multisite.index.values).all()
        assert df.index.dtype == np.dtype("datetime64[s]")

    def test_long_index_matches_date_range_values(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        long_idx = gen.generate(n_years=1000, seed=0).data_by_realization[0].index
        short_ref = pd.date_range("1940-01-01", periods=300, freq="YS")
        assert (long_idx[:300].values == short_ref.values).all()
        assert long_idx[-1].year == 1940 + 999


# ---------------------------------------------------------------------------
# Hurst fallback warning
# ---------------------------------------------------------------------------


class TestSMARTAHurstFallback:
    def test_warns_when_beta_not_lrd(self, annual_multisite, caplog):
        import logging

        gen = SMARTAGenerator(sma_order=64, acf_model="hurst")
        with caplog.at_level(logging.WARNING):
            gen.fit(annual_multisite)
        # i.i.d. gamma input has no LRD, so the CAS fit gives beta <= 1
        # for at least one site and the H=0.6 fallback must be announced.
        fallback_sites = [i for i, (H, _) in gen._cas_params.items() if H == 0.6]
        assert fallback_sites
        assert any(
            "Falling back to the default H=0.6" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Non-PD innovation covariance repair
# ---------------------------------------------------------------------------


class TestSMARTAInnovationRepair:
    def test_repair_preserves_diagonal_of_g_tilde(
        self, annual_multisite, monkeypatch, caplog
    ):
        """The G_tilde fallback must not rescale the diagonal to one.

        Forces the repair path by making the first Cholesky call fail and
        scales the SMA weights so that diag(G_tilde) = 1 / sum(a^2) != 1.
        The old repair_correlation_matrix() call would have returned a unit
        diagonal; the diagonal of B B^T must equal diag(G_tilde) instead.
        """
        import logging

        import synhydro.methods.generation.parametric.smarta as smarta_mod

        orig_chol = np.linalg.cholesky
        calls = {"n": 0}

        def failing_cholesky(x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced")
            return orig_chol(x)

        orig_sma = smarta_mod.sma_weights_fft
        monkeypatch.setattr(
            smarta_mod, "sma_weights_fft", lambda acf: orig_sma(acf) * 1.1
        )
        monkeypatch.setattr(np.linalg, "cholesky", failing_cholesky)

        gen = SMARTAGenerator(sma_order=64)
        captured = {}
        orig_repair = gen._repair_innovation_covariance

        def capturing_repair(G):
            captured["G"] = G.copy()
            return orig_repair(G)

        gen._repair_innovation_covariance = capturing_repair

        with caplog.at_level(logging.WARNING):
            gen.fit(annual_multisite)

        assert calls["n"] == 2
        assert any("not positive-definite" in r.message for r in caplog.records)
        G = captured["G"]
        expected_diag = np.full(3, 1.0 / 1.1**2)
        np.testing.assert_allclose(np.diag(G), expected_diag, rtol=1e-10)
        BBt = gen._B_tilde @ gen._B_tilde.T
        np.testing.assert_allclose(np.diag(BBt), np.diag(G), rtol=1e-10)
        # G_tilde was PD to begin with, so the repair must be a no-op
        np.testing.assert_allclose(BBt, G, atol=1e-12)

    def test_repair_fixes_indefinite_matrix_and_keeps_diagonal(self):
        gen = SMARTAGenerator()
        G = np.array([[0.8, 0.9, 0.9], [0.9, 0.8, -0.9], [0.9, -0.9, 0.8]])
        assert np.linalg.eigvalsh(G).min() < 0
        G_rep = gen._repair_innovation_covariance(G)
        np.testing.assert_allclose(np.diag(G_rep), np.diag(G), rtol=1e-12)
        np.linalg.cholesky(G_rep)


# ---------------------------------------------------------------------------
# Statistical reproduction of the published SMARTA properties
# ---------------------------------------------------------------------------


def _fgn_gamma_multisite(
    n_years=120,
    H=0.75,
    rho=((1.0, 0.7, 0.5), (0.7, 1.0, 0.6), (0.5, 0.6, 1.0)),
    shapes=(3.0, 5.0, 2.0),
    scales=(100.0, 60.0, 150.0),
    seed=0,
):
    """Deterministic 3-site annual series with fGn persistence and gamma margins.

    ``Z = L_t @ W @ L_s.T`` with ``W`` i.i.d. N(0, 1): ``L_t`` is the Cholesky
    factor of the Toeplitz matrix of ``hurst_acf(H)`` so every column has the
    fGn autocorrelation, and ``L_s`` is the Cholesky factor of ``rho`` so the
    columns have lag-0 correlation ``rho``. A Gaussian copula then maps each
    column onto its gamma marginal.
    """
    rho = np.asarray(rho, dtype=float)
    n_sites = rho.shape[0]
    rng = np.random.default_rng(seed)
    L_t = np.linalg.cholesky(toeplitz(hurst_acf(H, n_years - 1)))
    L_s = np.linalg.cholesky(rho)
    W = rng.standard_normal((n_years, n_sites))
    Z = L_t @ W @ L_s.T
    X = np.empty_like(Z)
    for i in range(n_sites):
        X[:, i] = gamma_dist.ppf(norm.cdf(Z[:, i]), a=shapes[i], scale=scales[i])
    dates = pd.date_range("1900-01-01", periods=n_years, freq="YS")
    return pd.DataFrame(X, index=dates, columns=["siteA", "siteB", "siteC"])


def _frozen_marginal(params):
    """Build the scipy frozen distribution described by a ``_marginal_params`` entry."""
    if params["dist"] == "gamma":
        return gamma_dist(a=params["shape"], loc=params["loc"], scale=params["scale"])
    if params["dist"] == "lognorm":
        return lognorm(s=params["s"], loc=params["loc"], scale=params["scale"])
    raise ValueError(f"Unexpected marginal distribution: {params['dist']}")


_N_REALIZATIONS = 40
_N_YEARS_SYN = 1000


@pytest.fixture(scope="module")
def smarta_fgn():
    """SMARTA fitted to the fGn/gamma fixture plus a 40 x 1000-year ensemble."""
    gen = SMARTAGenerator(sma_order=128, nataf_method="GH")
    gen.fit(_fgn_gamma_multisite())
    ens = gen.generate(n_realizations=_N_REALIZATIONS, n_years=_N_YEARS_SYN, seed=1)
    reals = [ens.data_by_realization[r] for r in range(_N_REALIZATIONS)]
    return gen, reals


class TestSMARTAStatisticalReproduction:
    """The generator must reproduce what SMARTA is built to preserve
    (Tsoukalas et al., 2018): the target autocorrelation function, the lag-0
    cross-site correlation, and the fitted marginal distribution.

    All sample statistics use the generator's own biased ACF estimator
    (``SMARTAGenerator._empirical_acf``) on both the observed and synthetic
    sides so that estimator bias cancels rather than masquerading as error.
    """

    def test_generated_acf_matches_target(self, smarta_fgn):
        gen, reals = smarta_fgn
        max_lag = 10
        for s in range(gen._n_sites):
            acf_syn = np.mean(
                [
                    SMARTAGenerator._empirical_acf(df.iloc[:, s].values, max_lag)
                    for df in reals
                ],
                axis=0,
            )
            target = gen._target_acf[s][: max_lag + 1]
            # The biased sample ACF of a long-memory series sits below the
            # true ACF by roughly sum(rho)/n at every lag (about 0.02 for the
            # LRD site here), so a small uniform shortfall is expected.
            assert np.abs(acf_syn[1:] - target[1:]).max() < 0.05

    def test_generated_acf_consistent_with_observed(self, smarta_fgn):
        """The CAS fit smooths a noisy 120-year sample ACF (per-lag sampling
        SE about 0.1), so the observed ACF is compared where the comparison
        is statistically meaningful: at lag 1, on the lags 1-5 average, and
        lag by lag against the model's own sampling band."""
        gen, reals = smarta_fgn
        max_lag = 5
        n_obs = len(gen._Q_annual)
        n_chunks = _N_YEARS_SYN // n_obs
        for s in range(gen._n_sites):
            acf_syn = np.mean(
                [
                    SMARTAGenerator._empirical_acf(df.iloc[:, s].values, max_lag)
                    for df in reals
                ],
                axis=0,
            )
            acf_obs = SMARTAGenerator._empirical_acf(
                gen._Q_annual.iloc[:, s].values, max_lag
            )
            # Lag 1 is the best-determined sample lag and anchors the CAS fit.
            assert abs(acf_syn[1] - acf_obs[1]) < 0.05
            # A least-squares fit balances residuals over lags, so the mean
            # over lags 1-5 is preserved even though single lags are not.
            assert abs(acf_syn[1:].mean() - acf_obs[1:].mean()) < 0.10
            # Each observed lag must lie within 3 sampling SEs of the model,
            # SE estimated from n_obs-year chunks of the synthetic ensemble.
            chunks = np.array(
                [
                    SMARTAGenerator._empirical_acf(
                        df.iloc[:, s].values[c * n_obs : (c + 1) * n_obs], max_lag
                    )[1:]
                    for df in reals
                    for c in range(n_chunks)
                ]
            )
            se = chunks.std(axis=0)
            assert np.all(np.abs(acf_obs[1:] - acf_syn[1:]) < 3.0 * se)

    def test_generated_cross_correlation_matches_observed(self, smarta_fgn):
        gen, reals = smarta_fgn
        corr_syn = np.mean([np.corrcoef(df.values.T) for df in reals], axis=0)
        corr_obs = np.corrcoef(gen._Q_annual.values.T)
        off_diag = ~np.eye(gen._n_sites, dtype=bool)
        assert np.abs(corr_syn - corr_obs)[off_diag].max() < 0.05

    def test_generated_marginal_moments_match_fitted(self, smarta_fgn):
        gen, reals = smarta_fgn
        pooled = np.vstack([df.values for df in reals])
        Q_obs = gen._Q_annual.values
        for s in range(gen._n_sites):
            dist = _frozen_marginal(gen._marginal_params[s])
            syn_mean = pooled[:, s].mean()
            syn_std = pooled[:, s].std()
            # Against the fitted marginal the generator samples from.
            assert abs(syn_mean / dist.mean() - 1.0) < 0.03
            assert abs(syn_std / dist.std() - 1.0) < 0.08
            # Looser, against the observed record the marginal was fitted to.
            assert abs(syn_mean / Q_obs[:, s].mean() - 1.0) < 0.10
            assert abs(syn_std / Q_obs[:, s].std() - 1.0) < 0.20
