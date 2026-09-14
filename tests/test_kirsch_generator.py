"""
Tests for Kirsch hybrid bootstrap streamflow generator.

Core round-trip, fit, generation, and paper-conformance tests are parametrized
across monthly and weekly fixtures. Tests that depend on daily input remain
monthly-only because daily-to-weekly aggregation is opt-in (requires
``timestep='weekly'``).
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.core.ensemble import Ensemble
from synhydro.utils import load_example_data


AGGREGATED_FIXTURES = ("sample_monthly_dataframe", "sample_weekly_dataframe")


class TestKirschGeneratorInitialization:
    """Tests for KirschGenerator initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters (no Q_obs at init)."""
        gen = KirschGenerator()
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        gen = KirschGenerator(
            generate_using_log_flow=True, matrix_repair_method="nearest", debug=True
        )
        assert gen.debug is True


class TestKirschGeneratorPreprocessing:
    """Tests for KirschGenerator preprocessing."""

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Daily input aggregates to monthly by default."""
        gen = KirschGenerator()
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q")
        assert hasattr(gen, "Qm")
        assert gen.n_sites == 3
        assert gen.Qm.shape[1] == 3
        assert gen.output_frequency == "MS"
        assert gen.n_periods_per_year == 12

    def test_preprocessing_daily_with_weekly_timestep(self, sample_daily_dataframe):
        """Daily input aggregates to weekly when timestep='weekly'."""
        gen = KirschGenerator()
        gen.preprocessing(sample_daily_dataframe, timestep="weekly")

        assert gen.is_preprocessed is True
        assert gen.output_frequency == "W-SUN"
        assert gen.n_periods_per_year == 52

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_preprocessing_aggregated_dataframe(self, fixture_name, request):
        """Pre-aggregated monthly/weekly input is auto-detected."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.preprocessing(df)

        assert gen.is_preprocessed is True
        if fixture_name == "sample_monthly_dataframe":
            assert gen.output_frequency == "MS"
            assert gen.n_periods_per_year == 12
        else:
            assert gen.output_frequency == "W-SUN"
            assert gen.n_periods_per_year == 52

    def test_preprocessing_contradictory_timestep_raises(
        self, sample_monthly_dataframe
    ):
        """Asking for weekly on monthly input must raise."""
        gen = KirschGenerator()
        with pytest.raises(ValueError, match="weekly"):
            gen.preprocessing(sample_monthly_dataframe, timestep="weekly")

    def test_preprocessing_with_log_transform(self, sample_daily_dataframe):
        """Log transformation is applied without error."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True

    def test_preprocessing_invalid_input(self):
        """Invalid input type raises TypeError during validation."""
        gen = KirschGenerator()
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])


class TestKirschGeneratorFit:
    """Tests for KirschGenerator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Fit on a single-site DataFrame derived from a Series."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        assert gen.is_fitted is True
        assert hasattr(gen, "mean_period")
        assert hasattr(gen, "std_period")
        assert hasattr(gen, "Z_h")
        assert len(gen.mean_period) == 12
        assert len(gen.std_period) == 12

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_multiple_sites(self, fixture_name, request):
        """Fit produces per-period mean/std with the right shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert gen.is_fitted is True
        n_per = gen.n_periods_per_year
        assert gen.mean_period.shape == (n_per, 3)
        assert gen.std_period.shape == (n_per, 3)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_creates_cholesky_decomposition(self, fixture_name, request):
        """Fit populates per-site Cholesky factors."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert hasattr(gen, "U_site")
        assert isinstance(gen.U_site, dict)
        assert len(gen.U_site) == gen.n_sites
        for s in range(gen.n_sites):
            assert gen.U_site[s].shape == (
                gen.n_periods_per_year,
                gen.n_periods_per_year,
            )

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_stores_correlation_matrices(self, fixture_name, request):
        """Fit stores Z_h with the expected (n_years, n_per, n_sites) shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert hasattr(gen, "Z_h")
        assert gen.Z_h.shape[1] == gen.n_periods_per_year
        assert gen.Z_h.shape[2] == gen.n_sites


class TestKirschGeneratorGenerate:
    """Tests for KirschGenerator generation."""

    def test_generate_single_realization_series(self, sample_daily_series):
        """Single-realization round-trip on monthly-from-daily input."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        assert isinstance(result.data_by_realization[0], pd.DataFrame)
        assert len(result.data_by_realization[0]) == gen.n_periods_per_year

    def test_generate_multiple_realizations_series(self, sample_daily_series):
        """Multiple realizations on monthly-from-daily input."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=5, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5
        for i in range(5):
            assert i in result.realization_ids
            assert isinstance(result.data_by_realization[i], pd.DataFrame)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_single_realization_dataframe(self, fixture_name, request):
        """Single realization has shape (n_periods_per_year, n_sites)."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        out = result.data_by_realization[0]
        assert out.shape[1] == 3
        assert len(out) == gen.n_periods_per_year

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_multiple_realizations_dataframe(self, fixture_name, request):
        """Multiple realizations each have n_sites columns and the right length."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=3, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3
        for r in range(3):
            assert r in result.realization_ids
            out = result.data_by_realization[r]
            assert out.shape[1] == 3
            assert len(out) == gen.n_periods_per_year

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_preserves_period_statistics(self, fixture_name, request):
        """Generated flows are finite and non-negative."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=20, n_years=5, seed=0)

        assert isinstance(result, Ensemble)
        for r in range(20):
            d = result.data_by_realization[r]
            assert not d.isna().any().any()
            assert (d >= 0).all().all()

    def test_generate_with_log_flow(self, sample_daily_series):
        """Log-flow generation produces finite output."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        d = result.data_by_realization[0]
        assert not d.isna().any().any()

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_as_array(self, fixture_name, request):
        """generate_single_series returns an array of shape (n_per*n_years, n_sites)."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate_single_series(n_years=2, as_array=True)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2 * gen.n_periods_per_year, 3)

    def test_generate_default_index_anchors_after_record(
        self, sample_monthly_dataframe
    ):
        """Default output index starts January 1 of the year after the record."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        result = gen.generate(n_realizations=1, n_years=2, seed=0)

        idx = result.data_by_realization[0].index
        assert idx[0] == pd.Timestamp("2021-01-01")
        assert len(idx) == 24

    def test_generate_start_year_anchors_index(self, sample_monthly_dataframe):
        """start_year anchors the output index at January 1 of that year, with
        month labels running Jan..Dec within each synthetic year."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        result = gen.generate(n_realizations=2, n_years=3, seed=0, start_year=1945)

        for r in range(2):
            idx = result.data_by_realization[r].index
            assert idx[0] == pd.Timestamp("1945-01-01")
            assert list(idx.month) == list(range(1, 13)) * 3
            assert list(np.unique(idx.year)) == [1945, 1946, 1947]

    def test_generate_start_year_relabels_only(self, sample_monthly_dataframe):
        """start_year changes labels, never values (same seed, same content)."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        default = gen.generate(n_realizations=1, n_years=2, seed=7)
        anchored = gen.generate(n_realizations=1, n_years=2, seed=7, start_year=1945)

        np.testing.assert_array_equal(
            default.data_by_realization[0].to_numpy(),
            anchored.data_by_realization[0].to_numpy(),
        )


class TestKirschGeneratorSaveLoad:
    """Tests for KirschGenerator save and load."""

    def test_save_and_load(self, sample_daily_dataframe, tmp_path):
        """Save then load reproduces the same output shape."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(sample_daily_dataframe)

        original_result = gen.generate(n_realizations=1, n_years=1)

        save_path = tmp_path / "kirsch_gen.pkl"
        gen.save(str(save_path))

        loaded_gen = KirschGenerator.load(str(save_path))

        assert loaded_gen.is_preprocessed is True
        assert loaded_gen.is_fitted is True
        assert loaded_gen.n_sites == 3

        loaded_result = loaded_gen.generate(n_realizations=1, n_years=1)

        assert (
            loaded_result.data_by_realization[0].shape
            == original_result.data_by_realization[0].shape
        )


class TestKirschGeneratorMethods:
    """Tests for KirschGenerator internal methods."""

    def test_repair_and_cholesky(self, sample_daily_dataframe):
        """_repair_and_cholesky is exercised by fit without error."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        assert hasattr(gen, "U_site")

    def test_bootstrap_indices_generation(self, sample_daily_series):
        """Bootstrap index path produces output."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=2, n_years=1)
        assert result is not None

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_destandardize_flows(self, fixture_name, request):
        """Destandardization keeps output non-negative."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert (result.data_by_realization[0] >= 0).all().all()


class TestKirschPaperConformance:
    """Tests verifying conformance with Kirsch et al. (2013), p. 6.

    The paper specifies that X_prime is a deterministic half-year shift of X,
    not an independent bootstrap. These tests guard against regression to
    the pre-fix behavior where ``generate_single_series`` drew a second
    bootstrap and ``generate_from_indices`` did a shared-index lookup.
    """

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_derive_X_prime_is_deterministic_shift(self, fixture_name, request):
        """X_prime row i = [second-half of X year i, first-half of X year i+1]."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(42)
        n_years = 5
        n_per = gen.n_periods_per_year
        half = n_per // 2
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)
        X = gen._create_bootstrap_tensor(M)
        X_prime = gen._derive_X_prime(X)

        assert X_prime.shape == (n_years + 1, n_per, gen.n_sites)
        np.testing.assert_allclose(X_prime[:n_years, :half], X[:n_years, half:])
        np.testing.assert_allclose(X_prime[:n_years, half:], X[1 : n_years + 1, :half])

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_derive_X_prime_rejects_wrong_period_count(self, fixture_name, request):
        """_derive_X_prime validates its input shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)
        bad_X = np.zeros((3, gen.n_periods_per_year - 2, gen.n_sites))
        with pytest.raises(ValueError, match="expected"):
            gen._derive_X_prime(bad_X)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_entry_points_agree_on_cross_period_correlation(
        self, fixture_name, request
    ):
        """generate() and generate_from_residuals() must agree on cross-period
        correlation. Pre-fix, generate() drew an independent bootstrap for
        X_prime and diverged."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        n_years = 10
        n_realizations = 50
        n_per = gen.n_periods_per_year

        ens_a = gen.generate(n_realizations=n_realizations, n_years=n_years, seed=42)
        flows_a = [
            ens_a.data_by_realization[r].values
            for r in sorted(ens_a.data_by_realization)
        ]

        rng = np.random.default_rng(42)
        flows_c = []
        for _ in range(n_realizations):
            # Resample from gen.Y (the space the bootstrap draws from), with
            # the n_years + 1 buffer row consumed by the half-year shift.
            residuals = np.empty((n_years + 1, n_per, gen.n_sites))
            for m in range(n_per):
                for s in range(gen.n_sites):
                    residuals[:, m, s] = rng.choice(
                        gen.Y[:, m, s], size=n_years + 1, replace=True
                    )
            flows_c.append(gen.generate_from_residuals(residuals, n_years=n_years))

        def pool_corr(flows_list):
            mats = []
            for fl in flows_list:
                col0 = fl[:, 0]
                grid = col0[: (len(col0) // n_per) * n_per].reshape(-1, n_per)
                mats.append(np.corrcoef(grid, rowvar=False))
            return np.mean(mats, axis=0)

        diff = pool_corr(flows_a) - pool_corr(flows_c)
        # Frobenius norm scales with sqrt(n_per^2); allow more slack for weekly.
        threshold = 1.5 if n_per == 12 else 6.0
        frob = np.linalg.norm(diff, ord="fro")
        assert frob < threshold, (
            f"generate() and generate_from_residuals() disagree on cross-period "
            f"correlation: Frobenius {frob:.3f} (threshold {threshold})."
        )

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_from_indices_matches_single_series(self, fixture_name, request):
        """Given the same M, generate_from_indices and generate_single_series
        must produce identical output."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(123)
        n_years = 8
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)

        out_series = gen.generate_single_series(n_years, M=M, as_array=True)
        out_indices = gen.generate_from_indices(M, n_years=n_years, as_array=True)

        np.testing.assert_allclose(out_series, out_indices)


class TestKirschGenerateFromResiduals:
    """Tests for the ``generate_from_residuals`` entry point.

    The residual tensor plays the role of the bootstrap tensor X and must
    carry the same ``n_years + 1`` buffer row that ``generate_single_series``
    and ``generate_from_indices`` consume in the half-year shift.
    """

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_from_residuals_matches_generate_from_indices(
        self, fixture_name, request
    ):
        """Feeding the bootstrap tensor built from M into
        generate_from_residuals must reproduce generate_from_indices(M)
        exactly."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(7)
        n_years = 6
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)
        X = gen._create_bootstrap_tensor(M)

        np.testing.assert_allclose(
            gen.generate_from_residuals(X, n_years=n_years),
            gen.generate_from_indices(M, n_years=n_years),
        )

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_from_residuals_does_not_duplicate_year_halves(
        self, fixture_name, request
    ):
        """Regression guard: the old implementation padded the residual
        tensor by repeating its last row, which made the last two synthetic
        years share an identical second half."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(0)
        n_years = 5
        n_per = gen.n_periods_per_year
        half = n_per // 2
        residuals = rng.standard_normal((n_years + 1, n_per, gen.n_sites))

        out = gen.generate_from_residuals(residuals, n_years=n_years)
        assert out.shape == (n_years * n_per, gen.n_sites)

        years = out.reshape(n_years, n_per, gen.n_sites)
        assert not np.allclose(
            years[n_years - 2, half:, :], years[n_years - 1, half:, :]
        )

    def test_generate_from_residuals_rejects_wrong_shape(
        self, sample_monthly_dataframe
    ):
        """Exactly n_years rows (the old contract) and a wrong period count
        are both rejected."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        n_years = 4
        n_per = gen.n_periods_per_year

        with pytest.raises(ValueError, match=r"n_years \+ 1"):
            gen.generate_from_residuals(
                np.zeros((n_years, n_per, gen.n_sites)), n_years=n_years
            )

        with pytest.raises(ValueError):
            gen.generate_from_residuals(np.zeros((n_years + 1, n_per - 2, gen.n_sites)))


@pytest.fixture(scope="module")
def usgs_monthly_complete_years():
    """Packaged USGS monthly flows trimmed to complete calendar years."""
    Q = load_example_data("usgs_monthly_streamflow_cms")
    counts = Q.groupby(Q.index.year).size()
    complete = counts[counts == 12].index
    return Q.loc[Q.index.year.isin(complete)]


@pytest.fixture(scope="module")
def kirsch_usgs_ensemble(usgs_monthly_complete_years):
    """Default KirschGenerator fit to the USGS record plus a 200 x 40-year
    ensemble (seed 0), shared across the statistical reproduction tests.

    With 200 realizations the pooled sample is N = 8000 years per
    (month, site) cell, so the sampling standard error of a per-period mean
    is about 0.011 sigma and the 0.05 sigma tolerance below is roughly 4.5
    standard errors. At 50 realizations (N = 2000, standard error 0.022
    sigma) the maximum over the 48 cells exceeds 0.05 sigma through sampling
    noise alone.
    """
    gen = KirschGenerator()
    gen.fit(usgs_monthly_complete_years)
    ens = gen.generate(n_realizations=200, n_years=40, seed=0)
    return gen, ens


class TestKirschStatisticalReproduction:
    """Statistical regression tests against the packaged USGS monthly record.

    A default (log-flow, normal-score) KirschGenerator is fit to the complete
    calendar years of ``usgs_monthly_streamflow_cms``. All statistics are
    computed on log flows pooled over realizations and years.

    Correlation targets are taken in the normal-score space of ``gen.Y`` (the
    observed standardized log residuals after the forward normal-score
    transform), because that is the space in which the per-site Cholesky
    factors impose correlation. The inverse normal-score transform is a
    nonlinear marginal map, so Pearson correlation of the synthetic output in
    ``Z_h`` space differs from ``Corr(Z_h)`` by up to about 0.09 on this
    record even at very large ensemble sizes (the historical
    ``|Corr(Y) - Corr(Z_h)|`` itself reaches 0.13). Comparing in ``Z_h``
    space would test the marginal transform rather than the correlation
    structure.
    """

    @staticmethod
    def _pooled_log_flows(gen, ens):
        """Stack log flows as (n_realizations * n_years, 12, n_sites).

        Axis 1 is calendar month 1..12 (grouped via ``df.index.month``) and
        axis 2 follows ``gen._sites``, matching the row and column order of
        ``gen.mean_period`` and ``gen.std_period``.
        """
        n_per = gen.n_periods_per_year
        blocks = []
        for r in sorted(ens.data_by_realization):
            df = ens.data_by_realization[r][gen._sites]
            log_q = np.log(df.to_numpy())
            months = df.index.month
            block = np.empty((len(df) // n_per, n_per, gen.n_sites))
            for m in range(n_per):
                block[:, m, :] = log_q[months == m + 1]
            blocks.append(block)
        return np.concatenate(blocks, axis=0)

    @staticmethod
    def _normal_scores(gen, log_flows):
        """Standardize pooled log flows with the fitted per-period moments
        and map them into the normal-score space of ``gen.Y``."""
        z = (log_flows - gen.mean_period.to_numpy()) / gen.std_period.to_numpy()
        return gen._apply_normal_score_transform(z)

    def test_per_period_mean_and_std(self, kirsch_usgs_ensemble):
        """Per-month mean of synthetic log flows is within 0.05 sigma of the
        fitted mean, and per-month std is within 10 percent of the fitted
        std, at every site."""
        gen, ens = kirsch_usgs_ensemble
        assert list(gen.mean_period.index) == list(range(1, 13))

        log_flows = self._pooled_log_flows(gen, ens)
        mean_syn = log_flows.mean(axis=0)
        std_syn = log_flows.std(axis=0, ddof=1)
        mean_obs = gen.mean_period.to_numpy()
        std_obs = gen.std_period.to_numpy()

        mean_err = np.abs(mean_syn - mean_obs) / std_obs
        std_ratio = std_syn / std_obs

        assert mean_err.max() < 0.05, f"max |mean error| / sigma = {mean_err.max():.4f}"
        assert (
            std_ratio.min() > 0.9 and std_ratio.max() < 1.1
        ), f"std ratio range = [{std_ratio.min():.4f}, {std_ratio.max():.4f}]"

    def test_within_half_year_correlation(self, kirsch_usgs_ensemble):
        """Within each half-year block, the synthetic intra-annual correlation
        in normal-score space matches the historical Corr(Y).

        Only the two within-half blocks (months 1-6 x 1-6 and 7-12 x 7-12)
        are asserted. The lag-1 correlation across the mid-year seam (month 6
        to month 7) is only approximately reproduced because the two halves
        come from different Cholesky factors; this is an inherent limitation
        of the published method (see docs/algorithms/kirsch.md) and is
        deliberately not asserted here.
        """
        gen, ens = kirsch_usgs_ensemble
        y_syn = self._normal_scores(gen, self._pooled_log_flows(gen, ens))
        half = gen.n_periods_per_year // 2

        max_diff = 0.0
        for s in range(gen.n_sites):
            c_syn = np.corrcoef(y_syn[:, :, s].T)
            c_obs = np.corrcoef(gen.Y[:, :, s].T)
            diff = np.abs(c_syn - c_obs)
            max_diff = max(max_diff, diff[:half, :half].max(), diff[half:, half:].max())

        assert max_diff < 0.10, f"max within-half correlation error = {max_diff:.4f}"

    def test_cross_site_lag0_correlation(self, kirsch_usgs_ensemble):
        """Per month, the synthetic cross-site correlation in normal-score
        space matches the historical Corr(Y[:, m, :]) to within 0.12.

        The bootstrap index matrix is shared across sites, which is what
        carries cross-site dependence into the synthetic tensor. The per-site
        Cholesky mixing then blends the cross-site correlation of month m
        with that of the earlier months feeding column m, so preservation is
        approximate rather than exact: on this record the implied deviation
        reaches about 0.09 where the cross-site correlation of the smallest
        basin swings strongly between months. The 0.12 tolerance leaves
        room for sampling noise on top of that systematic deviation.
        """
        gen, ens = kirsch_usgs_ensemble
        y_syn = self._normal_scores(gen, self._pooled_log_flows(gen, ens))

        max_diff = 0.0
        for m in range(gen.n_periods_per_year):
            c_syn = np.corrcoef(y_syn[:, m, :].T)
            c_obs = np.corrcoef(gen.Y[:, m, :].T)
            max_diff = max(max_diff, np.abs(c_syn - c_obs).max())

        assert max_diff < 0.12, f"max cross-site correlation error = {max_diff:.4f}"
