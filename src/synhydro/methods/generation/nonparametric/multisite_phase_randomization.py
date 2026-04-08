"""
Multisite Wavelet Phase Randomization Generator for synthetic streamflow generation.

Implements the Brunner and Gilleland (2020) wavelet-based multisite phase
randomization method (PRSim.wave) for generating synthetic daily streamflow at
multiple sites simultaneously. Spatial correlation is preserved by sharing a
single set of random phases -- derived from a white-noise CWT -- across all sites.

References
----------
Brunner, M.I., and Gilleland, E. (2020). Stochastic simulation of streamflow
and spatial extremes: a continuous, wavelet-based approach. Hydrology and Earth
System Sciences, 24, 3967-3982. https://doi.org/10.5194/hess-24-3967-2020
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pywt
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import norm as _norm

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


class MultisitePhaseRandomizationGenerator(Generator):
    """
    Multisite wavelet phase randomization generator (Brunner and Gilleland, 2020).

    Generates synthetic daily streamflow at multiple sites using a shared
    wavelet (CWT) phase structure. Each site's power spectrum (CWT amplitude) is
    preserved from the observed record, while spatial correlation is maintained
    by applying identical random phases -- drawn from a single white-noise CWT --
    to all sites simultaneously.

    Attributes
    ----------
    par_day_ : dict of dict
        Fitted kappa distribution parameters for each site and day of year.
        Keyed by site name, then day-of-year integer (1-365). Each leaf entry
        contains {'xi', 'alfa', 'k', 'h'}.
    cwt_amplitudes_ : dict of np.ndarray
        Per-site CWT amplitude spectra of shape (n_scales, N). Keyed by site name.
    norm_ : dict of np.ndarray
        Per-site normal-score transformed series of length N. Keyed by site name.
    scales_ : np.ndarray
        CWT scales used, shape (n_scales,).
    delta_j_ : float
        Log-scale spacing (constant for geometrically spaced scales).

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.nonparametric import (
    ...     MultisitePhaseRandomizationGenerator,
    ... )
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>> gen = MultisitePhaseRandomizationGenerator()
    >>> gen.preprocessing(Q_daily)
    >>> gen.fit()
    >>> ensemble = gen.generate(n_realizations=100, seed=42)

    Notes
    -----
    - Requires at least 2 years (730 days) of daily data per site.
    - February 29 observations are removed before fitting.
    - After leap-day removal, the record length must be a multiple of 365.
    - All sites must share the same DatetimeIndex.
    - The generator produces realizations of the same length as the observed record
      unless n_years is specified.
    """

    supports_multisite: bool = True
    supported_frequencies: tuple = ("D",)

    def __init__(
        self,
        *,
        wavelet: str = "cmor1.5-1.0",
        n_scales: int = 100,
        win_h_length: int = 15,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the MultisitePhaseRandomizationGenerator.

        Parameters
        ----------
        wavelet : str, default='cmor1.5-1.0'
            PyWavelets continuous wavelet identifier. The complex Morlet wavelet
            'cmor1.5-1.0' (bandwidth 1.5, center frequency 1.0) is recommended.
        n_scales : int, default=100
            Number of CWT scales, spaced log-uniformly from 2 to N/8 where N is
            the record length in days.
        win_h_length : int, default=15
            Half-window length (days) for per-day-of-year kappa fitting. Values
            within +-win_h_length days of each target day are pooled, giving a
            total window of 2*win_h_length+1 days.
        name : str, optional
            Name identifier for this generator instance.
        debug : bool, default=False
            Enable debug-level logging.
        **kwargs : dict
            Additional keyword arguments (currently unused).
        """
        super().__init__(name=name, debug=debug)

        self.wavelet = wavelet
        self.n_scales = n_scales
        self.win_h_length = win_h_length

        self.init_params.algorithm_params = {
            "wavelet": wavelet,
            "n_scales": n_scales,
            "win_h_length": win_h_length,
        }

        self.par_day_: Dict[str, Dict[int, Optional[Dict[str, float]]]] = {}
        self.cwt_amplitudes_: Dict[str, np.ndarray] = {}
        self.norm_: Dict[str, np.ndarray] = {}
        self.scales_: Optional[np.ndarray] = None
        self.delta_j_: Optional[float] = None

    @property
    def output_frequency(self) -> str:
        """Wavelet phase randomization generates daily output."""
        return "D"

    def preprocessing(
        self,
        Q_obs: "pd.DataFrame",
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Preprocess observed multisite daily streamflow data.

        Validates input, removes leap days, and creates per-site day-of-year
        indices. After leap-day removal, the record length must be a multiple
        of 365.

        Parameters
        ----------
        Q_obs : pd.DataFrame or pd.Series
            Observed daily streamflow with DatetimeIndex. A DataFrame with one
            column per site is required for multisite generation. A Series is
            accepted and treated as a single-site case.
        sites : list of str, optional
            Subset of columns to use. If None, all columns are used.
        **kwargs : dict
            Additional preprocessing parameters (currently unused).

        Raises
        ------
        ValueError
            If data has fewer than 730 days after leap-day removal, or if the
            length after removal is not a multiple of 365.
        """
        Q = self._store_obs_data(Q_obs, sites=sites)

        # Remove February 29 across all sites
        leap_mask = (Q.index.month == 2) & (Q.index.day == 29)
        n_leap = leap_mask.sum()
        Q = Q[~leap_mask]

        if n_leap > 0:
            self.logger.info("Removed %d leap-day observations", n_leap)

        if len(Q) < 730:
            raise ValueError(
                f"At least 730 days (2 years) of data required after removing "
                f"leap days, got {len(Q)}"
            )

        if len(Q) % 365 != 0:
            raise ValueError(
                f"Record length must be a multiple of 365 after removing leap "
                f"days. Got {len(Q)} days. Some days may be missing."
            )

        self.day_index_ = self._create_day_index(Q.index)
        self.Q_obs_df_ = Q.copy()
        self.Q_obs_index_ = Q.index
        self.n_years_ = len(Q) // 365

        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d days (%d years), %d sites",
            len(Q),
            self.n_years_,
            len(self._sites),
        )

    def _create_day_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Create day-of-year index (1-365) accounting for leap-day removal.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Datetime index with leap days already removed.

        Returns
        -------
        np.ndarray
            Integer array of day-of-year values in the range [1, 365].
        """
        doy = index.dayofyear.values.copy()
        is_leap = index.is_leap_year
        after_feb28 = doy > 59
        doy[is_leap & after_feb28] -= 1
        return doy

    def fit(
        self,
        Q_obs: Optional["pd.DataFrame"] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the multisite wavelet phase randomization model.

        This method:
        1. Fits per-site, per-day-of-year kappa distributions using L-moments.
        2. Applies the normal score transform per site and day of year.
        3. Computes the CWT of each normal-score series and stores per-site
           amplitude spectra.

        Parameters
        ----------
        Q_obs : pd.DataFrame, optional
            If provided, calls preprocessing() automatically.
        sites : list of str, optional
            Passed to preprocessing() when Q_obs is provided.
        **kwargs : dict
            Additional fitting parameters (currently unused).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        N = len(self.Q_obs_df_)
        self.scales_ = np.geomspace(2, N / 8, self.n_scales)
        self.delta_j_ = float(np.log(self.scales_[1] / self.scales_[0]))

        for site in self._sites:
            self.logger.debug("Fitting site: %s", site)
            obs = self.Q_obs_df_[site].values

            # Step 1: fit kappa distributions
            self.par_day_[site] = self._fit_kappa_all_days(obs)

            # Step 2: normal score transform
            self.norm_[site] = self._normal_score_transform(obs)

            # Step 3: CWT and store amplitudes
            coefs, _ = pywt.cwt(
                self.norm_[site], self.scales_, self.wavelet, sampling_period=1.0
            )
            self.cwt_amplitudes_[site] = np.abs(coefs)  # (n_scales, N)

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()
        self.logger.info(
            "Fitting complete: %d sites, %d days, wavelet=%s",
            len(self._sites),
            N,
            self.wavelet,
        )

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Ensemble":
        """
        Generate synthetic multisite daily streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of independent synthetic realizations to generate.
        n_years : int, optional
            Target length of each realization in years (365-day years, no leap
            days). When provided, independent phase-randomized chunks are
            concatenated until the target length is reached and then trimmed.
            When None, the output length equals the observed record length.
        n_timesteps : int, optional
            Not used. Length is controlled via n_years.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional generation parameters (currently unused).

        Returns
        -------
        Ensemble
            Generated synthetic flows as an Ensemble object. Each realization
            is a DataFrame with shape (n_days, n_sites) and a no-leap
            DatetimeIndex.
        """
        self.validate_fit()

        rng = np.random.default_rng(seed)
        n_obs = len(self.Q_obs_df_)

        if n_years is not None:
            n_target = int(n_years) * 365
        else:
            n_target = n_obs

        n_chunks = max(1, int(np.ceil(n_target / n_obs)))
        self.logger.info(
            "Generating %d realizations of %d days (%d chunks of %d each)",
            n_realizations,
            n_target,
            n_chunks,
            n_obs,
        )

        out_index = self._build_noleap_index(n_target)
        realizations: Dict[int, pd.DataFrame] = {}

        for r in range(n_realizations):
            segments: List[np.ndarray] = []
            for _ in range(n_chunks):
                chunk = self._generate_chunk(rng=rng)  # (n_obs, n_sites)
                segments.append(chunk)
            simulated = np.concatenate(segments, axis=0)[:n_target, :]

            realizations[r] = pd.DataFrame(
                simulated, index=out_index, columns=self._sites
            )

        self.logger.info("Generated %d realizations", n_realizations)
        return Ensemble(realizations)

    def _generate_chunk(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate one chunk of synthetic multisite flows of length N.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Synthetic flows of shape (N, n_sites).
        """
        N = len(self.Q_obs_df_)

        # Draw shared phases from a single white-noise CWT
        noise = rng.standard_normal(N)
        noise_coefs, _ = pywt.cwt(
            noise, self.scales_, self.wavelet, sampling_period=1.0
        )
        shared_phases = np.angle(noise_coefs)  # (n_scales, N)

        syn_normal: Dict[str, np.ndarray] = {}
        for site in self._sites:
            amp = self.cwt_amplitudes_[site]  # (n_scales, N)
            syn_coefs = amp * np.exp(1j * shared_phases)

            # Approximate inverse CWT: sum Re(W(a,t)) / sqrt(a) * delta_j
            raw = (
                np.sum(np.real(syn_coefs) / self.scales_[:, None] ** 0.5, axis=0)
                * self.delta_j_
            )

            # Standardize to unit variance (per-realization normalization)
            std = raw.std()
            if std > 0:
                raw = raw / std
            syn_normal[site] = raw

        # Back-transform to original units per site
        n_sites = len(self._sites)
        out = np.zeros((N, n_sites))
        for col_idx, site in enumerate(self._sites):
            out[:, col_idx] = self._back_transform(syn_normal[site], site, rng=rng)

        return out

    def _back_transform(
        self, y_syn: np.ndarray, site: str, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Back-transform a synthetic normal-score series to original flow units.

        Uses rank-based mapping: for each day of year, the synthetic values are
        ranked and mapped to kappa quantiles drawn from the fitted marginal.

        Parameters
        ----------
        y_syn : np.ndarray
            Synthetic series in the normal-score domain, length N.
        site : str
            Site name whose fitted kappa parameters are used.
        rng : np.random.Generator
            Random number generator for kappa sampling.

        Returns
        -------
        np.ndarray
            Synthetic flows in original units, length N, non-negative.
        """
        simulated = np.zeros(len(y_syn))

        for d in range(1, 366):
            mask = self.day_index_ == d
            y_day = y_syn[mask]
            n = len(y_day)

            if n == 0:
                continue

            params = self.par_day_[site].get(d)
            if params is not None:
                kappa_sample = self._rand_kappa(n, rng=rng, **params)
                ranks = np.argsort(np.argsort(y_day))
                simulated[mask] = np.sort(kappa_sample)[ranks]
            else:
                # Fall back to empirical distribution for this day
                obs_day = self.Q_obs_df_[site].values[mask]
                ranks = np.argsort(np.argsort(y_day))
                simulated[mask] = np.sort(obs_day)[ranks]

            # Enforce non-negativity
            day_pos = np.where(mask)[0]
            neg_idx = day_pos[simulated[mask] < 0]
            if len(neg_idx) > 0:
                obs_day_vals = self.Q_obs_df_[site].values[mask]
                min_obs = obs_day_vals.min()
                if min_obs > 0:
                    simulated[neg_idx] = rng.uniform(0.0, min_obs, len(neg_idx))
                else:
                    simulated[neg_idx] = 0.0

        return simulated

    def _fit_kappa_all_days(
        self, obs: np.ndarray
    ) -> Dict[int, Optional[Dict[str, float]]]:
        """
        Fit kappa distribution parameters for each day of year (1-365).

        Parameters
        ----------
        obs : np.ndarray
            Observed streamflow values, length N (multiple of 365).

        Returns
        -------
        dict
            Mapping from day-of-year integer (1-365) to kappa parameter dict
            {'xi', 'alfa', 'k', 'h'} or None if fitting failed.
        """
        par_day: Dict[int, Optional[Dict[str, float]]] = {}

        for d in range(1, 366):
            window_days = self._get_window_days(d)
            mask = np.isin(self.day_index_, window_days)
            data_window = obs[mask]

            try:
                lmom = self._compute_lmoments(data_window)
                params = self._fit_kappa_params(lmom)
                if params is not None:
                    par_day[d] = params
                else:
                    par_day[d] = par_day.get(d - 1) if d > 1 else None
                    if par_day[d] is None:
                        self.logger.debug("Day %d: kappa fitting returned None", d)
            except Exception as exc:
                par_day[d] = par_day.get(d - 1) if d > 1 else None
                self.logger.debug("Day %d: kappa fitting failed: %s", d, exc)

        # Backward fill any remaining None values
        for d in range(364, 0, -1):
            if par_day.get(d) is None and par_day.get(d + 1) is not None:
                par_day[d] = par_day[d + 1].copy()

        return par_day

    def _normal_score_transform(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply the Van der Waerden normal score transform per day of year.

        Parameters
        ----------
        obs : np.ndarray
            Observed streamflow values aligned with self.day_index_.

        Returns
        -------
        np.ndarray
            Normal-score transformed series, same length as obs.
        """
        norm = np.zeros(len(obs))
        for d in range(1, 366):
            mask = self.day_index_ == d
            day_vals = obs[mask]
            n = len(day_vals)
            if n == 0:
                continue
            ranks = np.argsort(np.argsort(day_vals)) + 1
            norm[mask] = _norm.ppf(ranks / (n + 1.0))
        return norm

    def _get_window_days(self, day: int) -> List[int]:
        """
        Return the set of days within a circular window around a target day.

        Parameters
        ----------
        day : int
            Target day of year (1-365).

        Returns
        -------
        list of int
            Day-of-year values included in the window (1-365, circular).
        """
        window = {day}
        for w in range(1, self.win_h_length + 1):
            window.add((day - w - 1) % 365 + 1)
            window.add((day + w - 1) % 365 + 1)
        return list(window)

    def _compute_lmoments(self, x: np.ndarray) -> Dict[str, float]:
        """
        Compute sample L-moments via probability-weighted moments.

        Parameters
        ----------
        x : np.ndarray
            Sample data values.

        Returns
        -------
        dict
            L-moment ratios with keys 'l1', 'l2', 'lcv', 'lca', 'lkur'.

        Raises
        ------
        ValueError
            If fewer than 4 observations are provided.
        """
        x_sorted = np.sort(x)
        n = len(x_sorted)

        if n < 4:
            raise ValueError("Need at least 4 observations for L-moments")

        pp = np.arange(n, dtype=float)
        nn = n - 1

        p1 = pp / nn
        p2 = p1 * (pp - 1) / max(nn - 1, 1)
        p3 = p2 * (pp - 2) / max(nn - 2, 1)

        b0 = np.mean(x_sorted)
        b1 = np.mean(p1 * x_sorted)
        b2 = np.mean(p2 * x_sorted)
        b3 = np.mean(p3 * x_sorted)

        l1 = b0
        l2 = 2 * b1 - b0
        lcv = (2 * b1 / b0 - 1) if b0 != 0 else 0.0

        denom = 2 * b1 - b0
        if denom == 0:
            lca = 0.0
            lkur = 0.0
        else:
            lca = 2 * (3 * b2 - b0) / denom - 3
            lkur = 5 * (2 * (2 * b3 - 3 * b2) + b0) / denom + 6

        return {"l1": l1, "l2": l2, "lcv": lcv, "lca": lca, "lkur": lkur}

    def _fit_kappa_params(self, lmom: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Fit four-parameter kappa distribution via L-moments.

        Parameters
        ----------
        lmom : dict
            L-moment dictionary with keys 'l1', 'l2', 'lca', 'lkur'.

        Returns
        -------
        dict or None
            Dictionary with keys 'xi', 'alfa', 'k', 'h', or None if fitting
            fails or the fit quality is poor.
        """
        lambda1 = lmom["l1"]
        lambda2 = lmom["l2"]
        tau3 = lmom["lca"]
        tau4 = lmom["lkur"]

        def theoretical_tau(
            k: float, h: float
        ) -> Tuple[Optional[float], Optional[float]]:
            if h == 0:
                if k == 0:
                    k = 1e-100
                t3 = 2 * (1 - 3 ** (-k)) / (1 - 2 ** (-k)) - 3
                t4 = (
                    5 * (1 - 4 ** (-k)) - 10 * (1 - 3 ** (-k)) + 6 * (1 - 2 ** (-k))
                ) / (1 - 2 ** (-k))
            else:
                g = np.zeros(4)
                for r in range(1, 5):
                    try:
                        if h > 0:
                            g[r - 1] = (r * gamma(1 + k) * gamma(r / h)) / (
                                h ** (1 + k) * gamma(1 + k + r / h)
                            )
                        else:
                            g[r - 1] = (r * gamma(1 + k) * gamma(-k - r / h)) / (
                                (-h) ** (1 + k) * gamma(1 - r / h)
                            )
                    except (ValueError, ZeroDivisionError):
                        return None, None
                if g[0] - g[1] == 0:
                    return None, None
                t3 = (-g[0] + 3 * g[1] - 2 * g[2]) / (g[0] - g[1])
                t4 = -((-g[0] + 6 * g[1] - 10 * g[2] + 5 * g[3]) / (g[0] - g[1]))
            return t3, t4

        def objective(kh: np.ndarray) -> float:
            k, h = kh
            if (k < -1 and h >= 0) or (h < 0 and (k <= -1 or k >= -1 / h)):
                return 1e10
            t3, t4 = theoretical_tau(k, h)
            if t3 is None:
                return 1e10
            return (tau3 - t3) ** 2 + (tau4 - t4) ** 2

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    [1.0, 1.0],
                    method="Nelder-Mead",
                    options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
                )

            if result.fun > 0.1:
                return None

            k, h = result.x
            xi, alfa = self._compute_xi_alfa(lambda1, lambda2, k, h)

            if alfa is None or alfa <= 0:
                return None

            return {"xi": xi, "alfa": alfa, "k": k, "h": h}

        except Exception:
            return None

    def _compute_xi_alfa(
        self, lambda1: float, lambda2: float, k: float, h: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute kappa location (xi) and scale (alfa) from first two L-moments.

        Parameters
        ----------
        lambda1 : float
            First L-moment (mean).
        lambda2 : float
            Second L-moment (L-scale).
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.

        Returns
        -------
        tuple of (float or None, float or None)
            (xi, alfa) or (None, None) if computation fails.
        """
        try:
            if h == 0:
                if k == 0:
                    k = 1e-100
                alfa = (lambda2 * k) / ((1 - 2 ** (-k)) * gamma(1 + k))
                xi = lambda1 - alfa * (1 - gamma(1 + k)) / k
            else:
                g = np.zeros(2)
                for r in range(1, 3):
                    if h > 0:
                        g[r - 1] = (r * gamma(1 + k) * gamma(r / h)) / (
                            h ** (1 + k) * gamma(1 + k + r / h)
                        )
                    else:
                        g[r - 1] = (r * gamma(1 + k) * gamma(-k - r / h)) / (
                            (-h) ** (1 + k) * gamma(1 - r / h)
                        )
                if g[0] - g[1] == 0:
                    return None, None
                alfa = (lambda2 * k) / (g[0] - g[1])
                xi = lambda1 - alfa * (1 - g[0]) / k
            return xi, alfa
        except Exception:
            return None, None

    def _rand_kappa(
        self,
        n: int,
        xi: float,
        alfa: float,
        k: float,
        h: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw random samples from the four-parameter kappa distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        xi : float
            Location parameter.
        alfa : float
            Scale parameter.
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.
        rng : np.random.Generator, optional
            Random number generator. If None, a new default generator is used.

        Returns
        -------
        np.ndarray
            Array of n random samples from the kappa distribution.
        """
        if rng is None:
            rng = np.random.default_rng()
        F = rng.uniform(1e-10, 1 - 1e-10, n)
        return self._invF_kappa(F, xi, alfa, k, h)

    def _invF_kappa(
        self,
        F: np.ndarray,
        xi: float,
        alfa: float,
        k: float,
        h: float,
    ) -> np.ndarray:
        """
        Evaluate the quantile function of the four-parameter kappa distribution.

        Parameters
        ----------
        F : np.ndarray
            Cumulative probabilities in (0, 1).
        xi : float
            Location parameter.
        alfa : float
            Scale parameter.
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.

        Returns
        -------
        np.ndarray
            Quantile values corresponding to F.
        """
        if k == 0:
            k = 1e-100
        if h == 0:
            x = xi + alfa * (1 - (-np.log(F)) ** k) / k
        else:
            x = xi + (alfa / k) * (1 - ((1 - F**h) / h) ** k)
        return x

    @staticmethod
    def _build_noleap_index(n_days: int, start_year: int = 2000) -> pd.DatetimeIndex:
        """
        Build a no-leap daily DatetimeIndex of exactly n_days length.

        Parameters
        ----------
        n_days : int
            Required number of daily time steps.
        start_year : int, default=2000
            Calendar year for the first day of the index.

        Returns
        -------
        pd.DatetimeIndex
            Daily index with February 29 entries removed.
        """
        n_calendar = n_days + n_days // 365 + 10
        all_days = pd.date_range(
            start=f"{start_year}-01-01", periods=n_calendar, freq="D"
        )
        noleap = all_days[~((all_days.month == 2) & (all_days.day == 29))]
        return noleap[:n_days]

    def _compute_fitted_params(self) -> FittedParams:
        """
        Package fitted parameters for inspection and serialization.

        Returns
        -------
        FittedParams
            Dataclass containing fitted parameter metadata.
        """
        n_params = len(self._sites) * 365 * 4  # kappa: 4 params per day per site

        training_period = (
            str(self.Q_obs_index_[0].date()),
            str(self.Q_obs_index_[-1].date()),
        )

        return FittedParams(
            distributions_={
                "type": "kappa",
                "n_sites": len(self._sites),
                "sites": self._sites,
            },
            transformations_={
                "normal_score": True,
                "wavelet": self.wavelet,
                "n_scales": self.n_scales,
                "win_h_length": self.win_h_length,
            },
            fitted_models_={
                "cwt_amplitude_shape": (
                    self.n_scales,
                    len(self.Q_obs_df_),
                )
            },
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs_df_),
            n_sites_=len(self._sites),
            training_period_=training_period,
        )
