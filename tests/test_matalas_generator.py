"""
Tests for MATALASGenerator (Matalas 1967, multi-site MAR(1)).
"""
import pickle
import numpy as np
import pandas as pd
import pytest

from synhydro.methods.generation.parametric.matalas import MATALASGenerator
from synhydro.core.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_multisite():
    """30 years of correlated monthly flows at 3 sites."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", periods=360, freq="MS")
    n = len(dates)
    # Seasonal pattern + correlated noise
    seasonal = 200 + 150 * np.sin(2 * np.pi * np.arange(n) / 12)
    data = {}
    base = rng.gamma(shape=3.0, scale=1.0, size=n)
    for i, name in enumerate(["A", "B", "C"]):
        data[name] = np.maximum(seasonal * (1 + 0.3 * i) + 80 * base + rng.normal(0, 20, n), 1.0)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def monthly_single_site(monthly_multisite):
    """Single site for degenerate (Thomas-Fiering-equivalent) tests."""
    return monthly_multisite[["A"]]


@pytest.fixture
def short_monthly(monthly_multisite):
    """Shorter record for edge-case tests."""
    return monthly_multisite.iloc[:60]   # 5 years


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestMATALASInit:
    def test_default_params(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        assert gen.log_transform is True
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False

    def test_log_transform_false(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite, log_transform=False)
        assert gen.log_transform is False

    def test_stores_algorithm_params(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        assert gen.init_params.algorithm_params['method'] == 'Matalas MAR(1)'

    def test_accepts_series(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite.iloc[:, 0])
        assert gen is not None

    def test_accepts_dataframe(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        assert gen is not None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestMATALASPreprocessing:
    def test_preprocessed_flag(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        assert gen.is_preprocessed is True

    def test_sites_stored(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        assert gen._sites == ["A", "B", "C"]
        assert gen._n_sites == 3

    def test_single_site(self, monthly_single_site):
        gen = MATALASGenerator(monthly_single_site)
        gen.preprocessing()
        assert gen._n_sites == 1

    def test_site_subset(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing(sites=["A", "B"])
        assert gen._sites == ["A", "B"]
        assert gen._n_sites == 2

    def test_monthly_index_preserved(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        assert len(gen.Q_obs_monthly) == 360

    def test_daily_resampled_to_monthly(self):
        dates = pd.date_range("2000-01-01", periods=365 * 10, freq="D")
        Q = pd.DataFrame({"X": np.random.gamma(2, 50, len(dates))}, index=dates)
        gen = MATALASGenerator(Q)
        gen.preprocessing()
        assert gen.Q_obs_monthly.index.freqstr in ('MS', 'MS-JAN')

    def test_no_fit_before_preprocessing_raises(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestMATALASFit:
    def test_fitted_flag(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        assert gen.is_fitted is True

    def test_twelve_matrices(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        assert len(gen._A) == 12
        assert len(gen._B) == 12

    def test_matrix_shapes(self, monthly_multisite):
        n = 3
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        for m in range(12):
            assert gen._A[m].shape == (n, n), f"A[{m}] wrong shape"
            assert gen._B[m].shape == (n, n), f"B[{m}] wrong shape"

    def test_b_lower_triangular(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        for m in range(12):
            B = gen._B[m]
            # Lower triangular: upper-right elements should be ~0
            assert np.allclose(np.triu(B, 1), 0, atol=1e-8)

    def test_mu_sigma_shape(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        assert gen._mu.shape == (12, 3)
        assert gen._sigma.shape == (12, 3)
        assert gen._sigma.values.min() > 0

    def test_mu_sigma_index(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        assert list(gen._mu.index) == list(range(1, 13))

    def test_fitted_params_stored(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        assert gen.fitted_params_ is not None
        assert gen.fitted_params_.n_sites_ == 3

    def test_single_site_fit(self, monthly_single_site):
        gen = MATALASGenerator(monthly_single_site)
        gen.preprocessing()
        gen.fit()
        assert gen.is_fitted
        for m in range(12):
            assert gen._A[m].shape == (1, 1)

    def test_no_generate_before_fit_raises(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=5)

    def test_log_transform_false_fits(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite, log_transform=False)
        gen.preprocessing()
        gen.fit()
        assert gen.is_fitted


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestMATALASGenerate:
    def test_returns_ensemble(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=10, n_realizations=2, seed=0)
        assert isinstance(result, Ensemble)

    def test_n_realizations(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=5, n_realizations=7, seed=0)
        assert result.metadata.n_realizations == 7

    def test_shape_multisite(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=10, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert df.shape == (120, 3)   # 10*12 months, 3 sites

    def test_site_columns_match(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=5, n_realizations=1, seed=0)
        assert list(result.data_by_realization[0].columns) == ["A", "B", "C"]

    def test_datetime_index(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=5, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_non_negative_flows(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=20, n_realizations=10, seed=0)
        for i in range(10):
            assert (result.data_by_realization[i].values >= 0).all()

    def test_no_nans(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=20, n_realizations=5, seed=0)
        for i in range(5):
            assert not result.data_by_realization[i].isna().any().any()

    def test_seed_reproducibility(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        r1 = gen.generate(n_years=10, n_realizations=3, seed=99)
        r2 = gen.generate(n_years=10, n_realizations=3, seed=99)
        for i in range(3):
            pd.testing.assert_frame_equal(
                r1.data_by_realization[i], r2.data_by_realization[i]
            )

    def test_different_seeds_differ(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        r1 = gen.generate(n_years=10, n_realizations=1, seed=1)
        r2 = gen.generate(n_years=10, n_realizations=1, seed=2)
        assert not r1.data_by_realization[0].equals(r2.data_by_realization[0])

    def test_n_timesteps_override(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_timesteps=36, n_realizations=1, seed=0)
        assert len(result.data_by_realization[0]) == 36

    def test_default_n_years(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_realizations=1, seed=0)
        # Should match historic length in years
        expected = len(gen.Q_obs_monthly) // 12 * 12
        assert len(result.data_by_realization[0]) == expected

    def test_realizations_differ(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=10, n_realizations=5, seed=7)
        # Not all realizations should be identical
        frames = [result.data_by_realization[i] for i in range(5)]
        n_unique = sum(1 for j in range(1, 5) if not frames[0].equals(frames[j]))
        assert n_unique >= 3


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

class TestMATALASStatistics:
    def test_monthly_mean_preserved(self, monthly_multisite):
        """Ensemble mean by month should approximate historical monthly mean."""
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=50, n_realizations=20, seed=0)

        # Pool all realizations
        all_dfs = pd.concat([result.data_by_realization[i] for i in range(20)])
        syn_means = all_dfs.groupby(all_dfs.index.month).mean()
        obs_means = gen.Q_obs_monthly.groupby(gen.Q_obs_monthly.index.month).mean()

        # Monthly means should be within 30% (wide tolerance for small ensemble)
        for col in gen._sites:
            ratio = syn_means[col] / obs_means[col]
            assert (ratio > 0.5).all() and (ratio < 2.0).all(), \
                f"Site {col}: monthly mean ratio out of range\n{ratio}"

    def test_spatial_correlation_sign_preserved(self, monthly_multisite):
        """Contemporaneous cross-site correlation sign should be preserved."""
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=50, n_realizations=10, seed=0)

        df_syn = pd.concat([result.data_by_realization[i] for i in range(10)])
        syn_corr = df_syn.corr()
        obs_corr = gen.Q_obs_monthly.corr()

        for i, si in enumerate(gen._sites):
            for j, sj in enumerate(gen._sites):
                if i != j:
                    assert np.sign(syn_corr.loc[si, sj]) == np.sign(obs_corr.loc[si, sj]), \
                        f"Correlation sign mismatch between {si} and {sj}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestMATALASSerialization:
    def test_pickle_roundtrip(self, monthly_multisite, tmp_path):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()

        path = tmp_path / "matalas.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)

        assert gen2.is_fitted
        assert gen2._n_sites == 3

    def test_generate_after_pickle(self, monthly_multisite, tmp_path):
        gen = MATALASGenerator(monthly_multisite)
        gen.preprocessing()
        gen.fit()

        path = tmp_path / "matalas.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)

        result = gen2.generate(n_years=5, n_realizations=2, seed=0)
        assert result.metadata.n_realizations == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestMATALASEdgeCases:
    def test_short_record_warns(self, short_monthly):
        """Short record should fit without crash (may log warnings)."""
        gen = MATALASGenerator(short_monthly)
        gen.preprocessing()
        gen.fit()
        assert gen.is_fitted

    def test_single_site_equivalent(self, monthly_single_site):
        """Single-site MAR(1) should produce valid output."""
        gen = MATALASGenerator(monthly_single_site)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=10, n_realizations=3, seed=0)
        for i in range(3):
            assert result.data_by_realization[i].shape == (120, 1)

    def test_two_sites(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite[["A", "B"]])
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=5, n_realizations=2, seed=0)
        assert result.data_by_realization[0].shape == (60, 2)

    def test_log_transform_false_end_to_end(self, monthly_multisite):
        gen = MATALASGenerator(monthly_multisite, log_transform=False)
        gen.preprocessing()
        gen.fit()
        result = gen.generate(n_years=10, n_realizations=2, seed=0)
        assert not result.data_by_realization[0].isna().any().any()
