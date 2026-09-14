"""
Tests for Phase Randomization Generator (Brunner et al. 2019).
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.phase_randomization import (
    PhaseRandomizationGenerator,
)
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_daily_series_long():
    """Generate a sample daily time series with at least 2 full years (730+ days)."""
    # Create exactly 3 years of data (no leap days will be handled by preprocessing)
    dates = pd.date_range(start="2010-01-01", end="2012-12-31", freq="D")
    np.random.seed(42)

    # Generate seasonal flow data
    n = len(dates)
    seasonal = 100 + 50 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.gamma(shape=2.0, scale=20.0, size=n)
    values = seasonal + noise

    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_daily_dataframe_long():
    """Generate a sample daily multi-site DataFrame with at least 2 full years."""
    dates = pd.date_range(start="2010-01-01", end="2012-12-31", freq="D")
    np.random.seed(42)

    n = len(dates)
    n_sites = 3
    data = {}

    for i in range(n_sites):
        # Generate seasonal flow data with noise
        seasonal = 100 + 50 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.gamma(shape=2.0, scale=20.0, size=n)
        data[f"site_{i+1}"] = seasonal + noise

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_daily_series_short():
    """Generate a short daily time series (less than 2 years) for error testing."""
    dates = pd.date_range(start="2010-01-01", end="2010-12-31", freq="D")
    np.random.seed(42)
    values = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture(scope="module")
def daily_ar1_seasonal_series():
    """Six years of daily flow with a seasonal cycle and AR(1) persistence.

    flow_t = 50 + 40 * sin(2 * pi * doy / 365) + 30 * exp(0.6 * y_t), where
    y_t is a unit-variance AR(1) process with phi = 0.85.  The date range
    includes 2004-02-29, which preprocessing removes, leaving 6 x 365 days.
    Shared (module scope) by the spectral/marginal regression tests; do not
    mutate it.
    """
    dates = pd.date_range(start="2001-01-01", end="2006-12-31", freq="D")
    rng = np.random.default_rng(7)
    n = len(dates)
    phi = 0.85

    innovations = rng.standard_normal(n) * np.sqrt(1.0 - phi**2)
    y = np.empty(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + innovations[t]

    doy = dates.dayofyear.values
    flow = 50.0 + 40.0 * np.sin(2.0 * np.pi * doy / 365.0) + 30.0 * np.exp(0.6 * y)
    return pd.Series(flow, index=dates, name="site_1")


@pytest.fixture(scope="module")
def ar1_kappa_generator(daily_ar1_seasonal_series):
    """Kappa-marginal generator fitted once on the AR(1) series (read-only)."""
    gen = PhaseRandomizationGenerator(marginal="kappa")
    gen.fit(daily_ar1_seasonal_series)
    return gen


@pytest.fixture(scope="module")
def ar1_kappa_ensemble(ar1_kappa_generator):
    """Two hundred realizations of the observed length from the shared generator.

    Generation costs about one second. The day-of-year marginal test pools
    n_realizations * n_years_ = 1200 values per day; with fewer realizations
    the maximum over its eight L-moment comparisons routinely reaches 2.5
    standard errors and the stated tolerances are not safe.
    """
    return ar1_kappa_generator.generate(n_realizations=200, seed=0)


def _sample_acf(x, lags):
    """Biased (1/n) sample autocorrelation of a mean-removed series at the given lags."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c0 = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / c0 for k in lags])


def _circular_autocovariance(x):
    """Circular autocovariance via the inverse FFT of the power spectrum."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    return np.real(np.fft.ifft(np.abs(np.fft.fft(x)) ** 2)) / len(x)


def _noleap_day_of_year(index):
    """Day-of-year (1-365) for a no-leap DatetimeIndex.

    Mirrors PhaseRandomizationGenerator._create_day_index: after Feb 29 is
    removed, dates past Feb 28 in a leap year have dayofyear one too large.
    """
    doy = index.dayofyear.values.copy()
    doy[index.is_leap_year & (doy > 59)] -= 1
    return doy


class TestPhaseRandomizationGeneratorInit:
    """Tests for PhaseRandomizationGenerator initialization."""

    def test_initialization_default_params(self, sample_daily_series_long):
        """Test initialization with default parameters."""
        gen = PhaseRandomizationGenerator()

        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False
        assert gen.marginal == "kappa"
        assert gen.win_h_length == 15

    def test_initialization_with_empirical_marginal(self, sample_daily_series_long):
        """Test initialization with empirical marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="empirical")

        assert gen.marginal == "empirical"

    def test_initialization_with_custom_window(self, sample_daily_series_long):
        """Test initialization with custom window length."""
        gen = PhaseRandomizationGenerator(win_h_length=20)

        assert gen.win_h_length == 20

    def test_initialization_invalid_marginal(self, sample_daily_series_long):
        """Test that invalid marginal raises ValueError."""
        with pytest.raises(ValueError, match="marginal must be"):
            PhaseRandomizationGenerator(marginal="invalid")

    def test_initialization_with_dataframe(self, sample_daily_dataframe_long):
        """Test initialization without data (data provided later)."""
        gen = PhaseRandomizationGenerator()

        assert gen.is_preprocessed is False

    def test_output_frequency(self, sample_daily_series_long):
        """Test that output frequency is daily."""
        gen = PhaseRandomizationGenerator()
        assert gen.output_frequency == "D"


class TestPhaseRandomizationGeneratorPreprocessing:
    """Tests for PhaseRandomizationGenerator preprocessing."""

    def test_preprocessing_basic(self, sample_daily_series_long):
        """Test basic preprocessing."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q_obs_")
        assert hasattr(gen, "day_index_")
        assert hasattr(gen, "n_years_")

    def test_preprocessing_removes_leap_days(self, sample_daily_series_long):
        """Test that preprocessing removes February 29."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        # Data length should be multiple of 365
        assert len(gen.Q_obs_) % 365 == 0

    def test_preprocessing_day_index_range(self, sample_daily_series_long):
        """Test that day index is in range 1-365."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        assert gen.day_index_.min() >= 1
        assert gen.day_index_.max() <= 365

    def test_preprocessing_minimum_data_requirement(self, sample_daily_series_short):
        """Test that preprocessing fails with insufficient data."""
        gen = PhaseRandomizationGenerator()

        with pytest.raises(ValueError, match="At least 730 days"):
            gen.preprocessing(sample_daily_series_short)

    def test_preprocessing_multisite_raises(self, sample_daily_dataframe_long):
        """Test that multi-site data raises ValueError."""
        gen = PhaseRandomizationGenerator()
        with pytest.raises(ValueError, match="univariate"):
            gen.preprocessing(sample_daily_dataframe_long)


class TestPhaseRandomizationGeneratorFit:
    """Tests for PhaseRandomizationGenerator fitting."""

    def test_fit_kappa_marginal(self, sample_daily_series_long):
        """Test fitting with kappa marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        assert gen.is_fitted is True
        assert hasattr(gen, "par_day_")
        assert hasattr(gen, "norm_")
        assert hasattr(gen, "modulus_")
        assert hasattr(gen, "phases_")

    def test_fit_empirical_marginal(self, sample_daily_series_long):
        """Test fitting with empirical marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="empirical")
        gen.fit(sample_daily_series_long)

        assert gen.is_fitted is True
        # Empirical marginal doesn't fit kappa params
        assert gen.par_day_ == {}

    def test_fit_kappa_params_structure(self, sample_daily_series_long):
        """Test that kappa parameters have correct structure."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        # Should have parameters for most days
        assert len(gen.par_day_) > 0

        # Check structure of a valid parameter set
        for day, params in gen.par_day_.items():
            if params is not None:
                assert "xi" in params
                assert "alfa" in params
                assert "k" in params
                assert "h" in params
                break

    def test_fit_normal_score_transform(self, sample_daily_series_long):
        """Test that normal score transform is applied."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Normalized data should have zero mean approximately
        assert gen.norm_ is not None
        assert len(gen.norm_) == len(gen.Q_obs_)

    def test_fit_fft_computation(self, sample_daily_series_long):
        """Test that FFT is computed correctly."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # FFT should have same length as data
        assert len(gen.modulus_) == len(gen.Q_obs_)
        assert len(gen.phases_) == len(gen.Q_obs_)

        # Modulus should be non-negative
        assert np.all(gen.modulus_ >= 0)

        # Phases should be in [-pi, pi]
        assert np.all(gen.phases_ >= -np.pi)
        assert np.all(gen.phases_ <= np.pi)

    def test_fit_without_preprocessing_raises(self, sample_daily_series_long):
        """Test that fit without preprocessing raises error."""
        gen = PhaseRandomizationGenerator()

        with pytest.raises(Exception):  # Will raise due to validation
            gen.fit()

    def test_fit_creates_fitted_params(self, sample_daily_series_long):
        """Test that fit creates FittedParams object."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        assert hasattr(gen, "fitted_params_")
        assert gen.fitted_params_.n_sites_ == 1


class TestPhaseRandomizationGeneratorGenerate:
    """Tests for PhaseRandomizationGenerator generation."""

    def test_generate_single_realization(self, sample_daily_series_long):
        """Test generating a single realization."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert len(result.realization_ids) == 1

    def test_generate_multiple_realizations(self, sample_daily_series_long):
        """Test generating multiple realizations."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=5, seed=42)

        assert isinstance(result, Ensemble)
        assert len(result.realization_ids) == 5

    def test_generate_reproducibility(self, sample_daily_series_long):
        """Test that seed produces reproducible results."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result1 = gen.generate(n_realizations=1, seed=42)
        result2 = gen.generate(n_realizations=1, seed=42)

        # Same seed should produce same results
        np.testing.assert_array_almost_equal(
            result1.data_by_realization[0].values, result2.data_by_realization[0].values
        )

    def test_generate_different_seeds(self, sample_daily_series_long):
        """Test that different seeds produce different results."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result1 = gen.generate(n_realizations=1, seed=42)
        result2 = gen.generate(n_realizations=1, seed=123)

        # Different seeds should produce different results
        assert not np.allclose(
            result1.data_by_realization[0].values, result2.data_by_realization[0].values
        )

    def test_generate_output_length_default(self, sample_daily_series_long):
        """Test that output length equals observed length when n_years is not given."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert len(result.data_by_realization[0]) == len(gen.Q_obs_)

    def test_generate_n_years(self, sample_daily_series_long):
        """Test that n_years produces the correct output length."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        n_years = 10
        result = gen.generate(n_realizations=2, n_years=n_years, seed=42)

        expected_len = n_years * 365
        for r in result.realization_ids:
            assert len(result.data_by_realization[r]) == expected_len

    def test_generate_n_years_longer_than_obs(self, sample_daily_series_long):
        """Test n_years greater than the observed record length."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        n_years = 50
        result = gen.generate(n_realizations=1, n_years=n_years, seed=42)

        expected_len = n_years * 365
        assert len(result.data_by_realization[0]) == expected_len

    def test_generate_n_years_noleap_index(self, sample_daily_series_long):
        """Test that output index contains no Feb 29 entries."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, n_years=5, seed=42)
        idx = result.data_by_realization[0].index
        feb29 = idx[(idx.month == 2) & (idx.day == 29)]
        assert len(feb29) == 0

    def test_generate_non_negative(self, sample_daily_series_long):
        """Test that generated flows are non-negative."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=10, seed=42)

        for r in result.realization_ids:
            assert (result.data_by_realization[r].values >= 0).all()

    def test_generate_kappa_marginal(self, sample_daily_series_long):
        """Test generation with kappa marginal."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert not result.data_by_realization[0].isna().any().any()

    def test_generate_empirical_marginal(self, sample_daily_series_long):
        """Test generation with empirical marginal."""
        gen = PhaseRandomizationGenerator(marginal="empirical")
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert not result.data_by_realization[0].isna().any().any()

    def test_generate_without_fit_raises(self, sample_daily_series_long):
        """Test that generate without fit raises error."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        with pytest.raises(Exception):  # Will raise due to validation
            gen.generate(n_realizations=1)


class TestLMomentsComputation:
    """Tests for L-moments computation."""

    def test_lmoments_basic(self, sample_daily_series_long):
        """Test basic L-moments computation."""
        gen = PhaseRandomizationGenerator()

        # Generate test data
        np.random.seed(42)
        data = np.random.gamma(shape=2.0, scale=50.0, size=100)

        lmom = gen._compute_lmoments(data)

        assert "l1" in lmom
        assert "l2" in lmom
        assert "lcv" in lmom
        assert "lca" in lmom
        assert "lkur" in lmom

    def test_lmoments_l1_is_mean(self, sample_daily_series_long):
        """Test that L1 is approximately the sample mean."""
        gen = PhaseRandomizationGenerator()

        np.random.seed(42)
        data = np.random.gamma(shape=2.0, scale=50.0, size=1000)

        lmom = gen._compute_lmoments(data)

        # L1 should be the mean
        np.testing.assert_almost_equal(lmom["l1"], np.mean(data), decimal=5)

    def test_lmoments_insufficient_data(self, sample_daily_series_long):
        """Test that L-moments computation fails with insufficient data."""
        gen = PhaseRandomizationGenerator()

        data = np.array([1.0, 2.0, 3.0])  # Only 3 observations

        with pytest.raises(ValueError, match="at least 4 observations"):
            gen._compute_lmoments(data)


class TestKappaDistribution:
    """Tests for kappa distribution functions."""

    def test_invF_kappa_basic(self, sample_daily_series_long):
        """Test inverse kappa CDF basic functionality."""
        gen = PhaseRandomizationGenerator()

        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0, alfa=1, k=0.5, h=0.5)

        # Output should be finite
        assert np.all(np.isfinite(x))

        # Values should be monotonically increasing with F
        assert x[0] < x[1] < x[2]

    def test_invF_kappa_gev_case(self, sample_daily_series_long):
        """Test inverse kappa CDF when h=0 (GEV case)."""
        gen = PhaseRandomizationGenerator()

        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0, alfa=1, k=0.5, h=0)

        assert np.all(np.isfinite(x))

    def test_rand_kappa_basic(self, sample_daily_series_long):
        """Test random kappa generation."""
        gen = PhaseRandomizationGenerator()

        np.random.seed(42)
        samples = gen._rand_kappa(n=1000, xi=0, alfa=1, k=0.5, h=0.5)

        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))


class TestStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_mean_within_tolerance(self, sample_daily_series_long):
        """Test that generated mean is within tolerance of observed."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate many realizations
        result = gen.generate(n_realizations=50, seed=42)

        obs_mean = gen.Q_obs_.mean()

        # Compute ensemble mean
        ensemble_means = [
            result.data_by_realization[r].values.mean() for r in result.realization_ids
        ]
        sim_mean = np.mean(ensemble_means)

        # Should be within 20% of observed mean
        relative_error = abs(sim_mean - obs_mean) / obs_mean
        assert relative_error < 0.2

    def test_std_within_tolerance(self, sample_daily_series_long):
        """Test that generated std is within tolerance of observed."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate many realizations
        result = gen.generate(n_realizations=50, seed=42)

        obs_std = gen.Q_obs_.std()

        # Compute ensemble std
        ensemble_stds = [
            result.data_by_realization[r].values.std() for r in result.realization_ids
        ]
        sim_std = np.mean(ensemble_stds)

        # Should be within 30% of observed std
        relative_error = abs(sim_std - obs_std) / obs_std
        assert relative_error < 0.3


class TestPhaseRandomizationGeneratorSaveLoad:
    """Tests for PhaseRandomizationGenerator save and load."""

    def test_save_and_load(self, sample_daily_series_long, tmp_path):
        """Test saving and loading generator."""
        gen = PhaseRandomizationGenerator(marginal="kappa", win_h_length=15)
        gen.fit(sample_daily_series_long)

        # Save
        save_path = tmp_path / "phase_rand_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = PhaseRandomizationGenerator.load(str(save_path))

        assert loaded_gen.is_preprocessed is True
        assert loaded_gen.is_fitted is True
        assert loaded_gen.marginal == "kappa"
        assert loaded_gen.win_h_length == 15

    def test_load_and_generate(self, sample_daily_series_long, tmp_path):
        """Test that loaded generator can generate."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate before saving
        original_result = gen.generate(n_realizations=1, seed=42)

        # Save and load
        save_path = tmp_path / "phase_rand_gen.pkl"
        gen.save(str(save_path))
        loaded_gen = PhaseRandomizationGenerator.load(str(save_path))

        # Generate from loaded
        loaded_result = loaded_gen.generate(n_realizations=1, seed=42)

        # Results should have same shape
        assert (
            original_result.data_by_realization[0].shape
            == loaded_result.data_by_realization[0].shape
        )


class TestWindowDays:
    """Tests for window days computation."""

    def test_get_window_days_middle(self, sample_daily_series_long):
        """Test window days for middle of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(100)  # Day 100

        # Should include day 100 and days within +/-15
        assert 100 in window
        assert len(window) == 31  # 15 before + 15 after + target

    def test_get_window_days_wrap_around_start(self, sample_daily_series_long):
        """Test window days wraps around at start of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(5)  # Day 5

        # Should wrap around to include days from end of previous year
        assert 5 in window
        assert 365 in window or 364 in window  # Should include some Dec days

    def test_get_window_days_wrap_around_end(self, sample_daily_series_long):
        """Test window days wraps around at end of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(360)  # Day 360

        # Should wrap around to include days from start of next year
        assert 360 in window
        assert 1 in window or 5 in window  # Should include some Jan days


class TestPhaseRandomizationSpectralAndMarginal:
    """Regression tests for the published properties of Brunner et al. (2019).

    1. Phase randomization preserves the amplitude spectrum (and therefore the
       circular autocovariance) of the normal-score series exactly.
    2. The autocorrelation is carried through the kappa back-transform into
       the generated flows.
    3. The fitted day-of-year kappa marginal reproduces the L-moments of the
       31-day observed window it was fitted to.
    4. The generated day-of-year marginal matches that observed window.
    """

    LAGS = (1, 2, 5, 10, 30)
    DAYS = (15, 100, 200, 300)

    def test_phase_randomize_preserves_amplitude_spectrum(
        self, sample_daily_series_long
    ):
        """|FFT| of the phase-randomized series equals modulus_ exactly."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # _compute_fft() takes the FFT of norm_ as-is (no detrending or
        # centring), so norm_ is the correct reference series.
        ts = gen._phase_randomize(rng=np.random.default_rng(0))

        assert ts.shape == gen.norm_.shape
        np.testing.assert_allclose(
            np.abs(np.fft.fft(ts)), gen.modulus_, rtol=1e-8, atol=1e-8
        )

        # Phases really changed, while the DC component (mean) is untouched.
        assert not np.allclose(ts, gen.norm_)
        assert np.isclose(ts.mean(), gen.norm_.mean(), atol=1e-10)

        # Wiener-Khinchin: an identical power spectrum implies an identical
        # circular autocovariance, i.e. the dependence structure of the
        # normal-score series is preserved exactly in the Gaussian domain.
        np.testing.assert_allclose(
            _circular_autocovariance(ts),
            _circular_autocovariance(gen.norm_),
            atol=1e-10,
        )

    def test_generated_acf_matches_observed(
        self, ar1_kappa_generator, ar1_kappa_ensemble
    ):
        """Autocorrelation of the AR(1) fixture survives the back-transform."""
        gen = ar1_kappa_generator
        ens = ar1_kappa_ensemble
        lags = self.LAGS

        obs_acf = _sample_acf(gen.Q_obs_, lags)
        sim_acf = np.mean(
            [
                _sample_acf(ens.data_by_realization[r].iloc[:, 0].values, lags)
                for r in ens.realization_ids
            ],
            axis=0,
        )

        # Reference: the observed normal scores (observed rank sequence, no
        # phase randomization) pushed through the same back-transform.  This
        # isolates what phase randomization must reproduce from the loss that
        # _back_transform itself introduces by drawing an independent kappa
        # sample of size n_years_ for every day and assigning it by rank.
        rng = np.random.default_rng(1)
        ref_acf = np.mean(
            [
                _sample_acf(gen._back_transform(gen.norm_, rng=rng), lags)
                for _ in range(20)
            ],
            axis=0,
        )

        # Genuine persistence: a seasonal cycle with white-noise anomalies
        # gives a lag-1 flow autocorrelation of about 0.58 for this fixture.
        assert sim_acf[0] > 0.7

        # Phase randomization carries the dependence through as well as the
        # observed rank sequence itself does.  The residual (about 0.035 at
        # lag 1) is the loss from re-ranking a continuous Gaussian series
        # within days of only n_years_ = 6 values; the ensemble-mean standard
        # error is below 0.01, so 0.06 is that bias plus three standard errors.
        assert np.max(np.abs(sim_acf - ref_acf)) < 0.06

        # Direct comparison with the observed flows.  The rank-based
        # back-transform only ever dilutes dependence (independent per-day
        # kappa levels), so the synthetic ACF sits below the observed one.
        # With n_years_ = 6 the measured lag-1 gap is 0.18 (0.12 / 0.08 / 0.05
        # at 12 / 30 / 60 years), so a tight bound is not attainable at this
        # record length; the 0.25 bound documents the current behaviour.
        assert np.all(obs_acf - sim_acf > -0.05)
        assert np.max(np.abs(obs_acf - sim_acf)) < 0.25

    def test_kappa_marginal_lmoments_roundtrip(self, ar1_kappa_generator):
        """Kappa samples reproduce the L-moments of the fitted 31-day window."""
        gen = ar1_kappa_generator
        rng = np.random.default_rng(3)
        n_checked = 0

        for day in self.DAYS:
            params = gen.par_day_.get(day)
            if params is None:
                # _fit_kappa_params returns None when the Nelder-Mead
                # objective exceeds 0.1 and no neighbour could be copied in;
                # there is nothing to round-trip for such a day.
                continue

            # Same window selection as _fit_kappa_distributions: all years of
            # the 2 * win_h_length + 1 circular window days, unweighted.
            window = gen.Q_obs_[np.isin(gen.day_index_, gen._get_window_days(day))]
            assert len(window) == (2 * gen.win_h_length + 1) * gen.n_years_
            lmom_obs = gen._compute_lmoments(window)

            refit = gen._fit_kappa_params(lmom_obs)
            if refit is None:
                # Fitting failed for this day, so par_day_ holds a neighbouring
                # day's parameters (fallback in _fit_kappa_distributions) and
                # the round-trip is not expected to hold.
                continue
            keys = ("xi", "alfa", "k", "h")
            np.testing.assert_allclose(
                [refit[k] for k in keys], [params[k] for k in keys], rtol=1e-6
            )

            sample = gen._rand_kappa(20000, rng=rng, **params)
            lmom_sim = gen._compute_lmoments(sample)

            rel_l1 = abs(lmom_sim["l1"] - lmom_obs["l1"]) / lmom_obs["l1"]
            rel_l2 = abs(lmom_sim["l2"] - lmom_obs["l2"]) / lmom_obs["l2"]
            abs_lca = abs(lmom_sim["lca"] - lmom_obs["lca"])

            assert rel_l1 < 0.03
            assert rel_l2 < 0.05
            assert abs_lca < 0.05
            n_checked += 1

        assert n_checked > 0

    def test_generated_day_of_year_marginal(
        self, ar1_kappa_generator, ar1_kappa_ensemble
    ):
        """Pooled generated values per day match the observed window marginal."""
        gen = ar1_kappa_generator
        ens = ar1_kappa_ensemble

        # generate() builds the output index from 2000-01-01 (a leap year)
        # with Feb 29 removed, so raw dayofyear is one too large after Feb 28
        # in leap years.  Apply the generator's own adjustment and confirm the
        # positions map to the same day-of-year that _back_transform used.
        idx = ens.data_by_realization[0].index
        doy = _noleap_day_of_year(idx)
        np.testing.assert_array_equal(doy, gen.day_index_)

        for day in self.DAYS:
            pooled = np.concatenate(
                [
                    ens.data_by_realization[r].iloc[:, 0].values[doy == day]
                    for r in ens.realization_ids
                ]
            )
            assert len(pooled) == len(ens.realization_ids) * gen.n_years_

            window = gen.Q_obs_[np.isin(gen.day_index_, gen._get_window_days(day))]
            lmom_obs = gen._compute_lmoments(window)
            lmom_sim = gen._compute_lmoments(pooled)

            rel_l1 = abs(lmom_sim["l1"] - lmom_obs["l1"]) / lmom_obs["l1"]
            rel_l2 = abs(lmom_sim["l2"] - lmom_obs["l2"]) / lmom_obs["l2"]

            # Measured (seed 0, 1200 values per day): rel l1 <= 0.021 and
            # rel l2 <= 0.050 across the four days; the relative standard
            # errors are about 0.012 (l1) and 0.02 (l2).
            assert rel_l1 < 0.05
            assert rel_l2 < 0.10
