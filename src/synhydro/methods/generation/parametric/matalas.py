"""
Matalas (1967) multi-site lag-1 autoregressive model for monthly streamflow.

The classical MAR(1) model is the standard parametric multi-site baseline in
stochastic hydrology, appearing in virtually every comparison study as the
reference against which modern methods are evaluated.
"""
import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


def _nearest_psd(A: NDArray) -> NDArray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.

    Uses eigendecomposition to clip negative eigenvalues to a small positive
    value. Applied when residual covariance matrices become slightly non-PSD
    due to numerical errors.

    Parameters
    ----------
    A : NDArray
        Symmetric matrix of shape (n, n).

    Returns
    -------
    NDArray
        Nearest PSD matrix of same shape.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


class MATALASGenerator(Generator):
    """
    Matalas (1967) multi-site monthly lag-1 autoregressive (MAR(1)) model.

    The standard classical baseline for parametric multi-site stochastic
    generation. Extends the Thomas-Fiering univariate model to n sites using
    matrix autoregression, preserving contemporaneous cross-site correlations
    and lag-1 temporal structure at each site.

    For each monthly transition m → m+1, generates:

        Z(t+1) = A(m) · Z(t) + B(m) · ε(t+1)

    where Z are standardized flows across all sites, ε ~ N(0, I), and A, B
    are coefficient matrices fitted from observed cross-correlations.

    Parameters
    ----------
    Q_obs : pd.DataFrame or pd.Series
        Monthly streamflow with DatetimeIndex. If Series, treated as single
        site (equivalent to Thomas-Fiering). Columns are sites.
    log_transform : bool, default=True
        Apply log(Q + 1) transformation before standardization to reduce
        skewness and improve normality assumption.
    name : str, optional
        Name for this generator instance.
    debug : bool, default=False
        Enable debug logging.

    Notes
    -----
    The coefficient matrices are derived from the lag-0 and lag-1
    cross-correlation matrices of the standardized flows:

        A(m) = S₁(m) · S₀(m)⁻¹
        B(m) · B(m)ᵀ = S₀(m+1) - A(m) · S₀(m) · A(m)ᵀ

    where S₀(m) is the contemporaneous correlation matrix at month m and
    S₁(m) is the lag-1 cross-correlation between months m+1 and m.
    B(m) is the lower Cholesky factor of the residual covariance.

    Examples
    --------
    >>> gen = MATALASGenerator(Q_monthly)
    >>> gen.preprocessing()
    >>> gen.fit()
    >>> ensemble = gen.generate(n_years=100, n_realizations=50, seed=42)

    References
    ----------
    Matalas, N. C. (1967). Mathematical assessment of synthetic hydrology.
    Water Resources Research, 3(4), 937–945.

    Salas, J. D., Delleur, J. W., Yevjevich, V., & Lane, W. L. (1980).
    Applied Modeling of Hydrologic Time Series. Water Resources Publications.
    """

    def __init__(
        self,
        Q_obs,
        log_transform: bool = True,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(Q_obs=Q_obs, name=name, debug=debug)

        self.log_transform = log_transform

        self.init_params.algorithm_params = {
            'method': 'Matalas MAR(1)',
            'reference': 'Matalas (1967)',
        }
        self.init_params.transformation_params = {
            'log_transform': log_transform,
        }

        # Fitted attributes (set during fit)
        self._mu: Optional[pd.DataFrame] = None       # shape (12, n_sites)
        self._sigma: Optional[pd.DataFrame] = None    # shape (12, n_sites)
        self._A: Optional[List[NDArray]] = None        # list of 12 (n×n) matrices
        self._B: Optional[List[NDArray]] = None        # list of 12 (n×n) matrices

    @property
    def output_frequency(self) -> str:
        return 'MS'

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(self, sites: Optional[list] = None, **kwargs) -> None:
        """
        Validate input and resample to monthly frequency.

        Parameters
        ----------
        sites : list, optional
            Subset of site columns to use. Uses all columns if None.
        **kwargs : dict, optional
            Unused.
        """
        Q = self.validate_input_data(self._Q_obs_raw)

        if sites is not None:
            Q = Q[sites]

        self._sites = Q.columns.tolist()
        self._n_sites = len(self._sites)

        # Resample to monthly if needed (check resolution, not freq string)
        inferred = pd.infer_freq(Q.index[:min(30, len(Q))])
        if inferred is not None and (inferred.startswith('D') or inferred.startswith('W')):
            self.logger.info("Resampling from %s to monthly (sum)", inferred)
            Q = Q.resample('MS').sum()

        # Ensure positive values before log transform
        Q = Q.clip(lower=1e-6)

        self.Q_obs_monthly = Q
        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d months, %d sites",
            len(Q), self._n_sites,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> None:
        """
        Estimate MAR(1) coefficient matrices from observed monthly flows.

        For each of the 12 monthly transitions, computes lag-0 (S0) and
        lag-1 (S1) cross-correlation matrices then solves for A and B.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused.
        """
        self.validate_preprocessing()

        Q = self.Q_obs_monthly.copy()
        n_sites = self._n_sites

        # Optional log transform
        if self.log_transform:
            Q = np.log(Q + 1.0)

        # Monthly means and standard deviations (shape: 12 × n_sites)
        mu = np.zeros((12, n_sites))
        sigma = np.zeros((12, n_sites))
        for m in range(12):
            mask = Q.index.month == (m + 1)
            vals = Q.loc[mask].values
            mu[m] = vals.mean(axis=0)
            sigma[m] = vals.std(axis=0, ddof=1)
            sigma[m] = np.maximum(sigma[m], 1e-8)

        self._mu = pd.DataFrame(mu, index=range(1, 13), columns=self._sites)
        self._sigma = pd.DataFrame(sigma, index=range(1, 13), columns=self._sites)

        # Standardize: Z[t] = (Q[t] - mu[month]) / sigma[month]
        Z = Q.copy()
        for m in range(12):
            mask = Z.index.month == (m + 1)
            Z.loc[mask] = (Z.loc[mask].values - mu[m]) / sigma[m]

        # Fit 12 transition matrices A(m) and B(m), m=0..11
        # Transition m → (m+1) % 12
        self._A = []
        self._B = []

        for m in range(12):
            next_m = (m + 1) % 12

            # Collect Z(m, y) and Z(m+1, y) pairs
            # For Dec→Jan, pair Dec of year y with Jan of year y+1
            Z_curr = self._extract_month_vectors(Z, m + 1)       # shape (n_years, n_sites)
            Z_next = self._extract_month_vectors(Z, next_m + 1)  # shape (n_years, n_sites)

            # Align Dec→Jan across year boundary
            if m == 11:
                Z_curr = Z_curr[:-1]   # Dec years 0..T-2
                Z_next = Z_next[1:]    # Jan years 1..T-1
            else:
                # Same year — align by minimum length
                n = min(len(Z_curr), len(Z_next))
                Z_curr = Z_curr[:n]
                Z_next = Z_next[:n]

            n_obs = len(Z_curr)
            if n_obs < n_sites + 2:
                self.logger.warning(
                    "Month %d: only %d observations for %d sites; "
                    "A may be poorly conditioned",
                    m + 1, n_obs, n_sites,
                )

            # Lag-0 cross-correlation at month m (contemporaneous)
            S0 = (Z_curr.T @ Z_curr) / (n_obs - 1)

            # Lag-1 cross-correlation: corr(Z(m+1), Z(m))
            # Element [i, j] = sum_y z_i(m+1, y) * z_j(m, y)
            S1 = (Z_next.T @ Z_curr) / (n_obs - 1)

            # Lag-0 at month m+1 (needed for residual covariance)
            Z_next2 = self._extract_month_vectors(Z, next_m + 1)
            n2 = len(Z_next2)
            S0_next = (Z_next2.T @ Z_next2) / (n2 - 1)

            # A(m) = S1 · S0⁻¹
            try:
                A = S1 @ np.linalg.solve(S0, np.eye(n_sites))
            except np.linalg.LinAlgError:
                self.logger.warning("Singular S0 at month %d; using pseudo-inverse", m + 1)
                A = S1 @ np.linalg.pinv(S0)

            # Residual covariance: M = S0(m+1) - A · S0(m) · Aᵀ
            M = S0_next - A @ S0 @ A.T

            # Symmetrize and enforce PSD before Cholesky
            M = 0.5 * (M + M.T)
            M = _nearest_psd(M)

            try:
                B = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                # Fallback: diagonal noise
                self.logger.warning(
                    "Cholesky failed at month %d transition; using diagonal B", m + 1
                )
                B = np.diag(np.sqrt(np.maximum(np.diag(M), 1e-8)))

            self._A.append(A)
            self._B.append(B)

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()
        self.logger.info(
            "Fitting complete: 12 A/B matrices for %d sites", n_sites
        )

    def _extract_month_vectors(self, Z: pd.DataFrame, month: int) -> NDArray:
        """Return rows of Z for the given calendar month (1-indexed)."""
        mask = Z.index.month == month
        return Z.loc[mask].values

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_one(self, n_years: int) -> pd.DataFrame:
        """
        Generate a single realization of monthly flows across all sites.

        Parameters
        ----------
        n_years : int
            Number of years to simulate.

        Returns
        -------
        pd.DataFrame
            Synthetic monthly flows, shape (n_years*12, n_sites).
        """
        n_sites = self._n_sites
        n_steps = n_years * 12
        mu = self._mu.values    # (12, n_sites)
        sigma = self._sigma.values  # (12, n_sites)

        # Initialize: draw first month from marginal (standard normal)
        Z_prev = np.random.randn(n_sites)

        Z_all = np.zeros((n_steps, n_sites))
        Z_all[0] = Z_prev

        for t in range(1, n_steps):
            m_prev = (t - 1) % 12   # 0-indexed month of previous step
            eps = np.random.randn(n_sites)
            Z_curr = self._A[m_prev] @ Z_prev + self._B[m_prev] @ eps
            Z_all[t] = Z_curr
            Z_prev = Z_curr

        # Back-transform: Q = sigma(m) * Z + mu(m)  then invert log if needed
        Q_syn = np.zeros_like(Z_all)
        for t in range(n_steps):
            m = t % 12
            Q_syn[t] = Z_all[t] * sigma[m] + mu[m]

        if self.log_transform:
            Q_syn = np.expm1(Q_syn)

        # Enforce non-negative
        Q_syn = np.maximum(Q_syn, 0.0)

        # Build DataFrame with monthly DatetimeIndex
        start = pd.Timestamp(f"{self.Q_obs_monthly.index[0].year}-01-01")
        dates = pd.date_range(start=start, periods=n_steps, freq='MS')
        return pd.DataFrame(Q_syn, index=dates, columns=self._sites)

    def generate(
        self,
        n_years: Optional[int] = None,
        n_realizations: int = 1,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic monthly streamflows at all sites.

        Parameters
        ----------
        n_years : int, optional
            Years per realization. Defaults to length of historic record.
        n_realizations : int, default=1
            Number of independent synthetic sequences.
        n_timesteps : int, optional
            Total monthly timesteps; overrides n_years when provided.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict, optional
            Unused.

        Returns
        -------
        Ensemble
            Collection of synthetic realizations.
        """
        self.validate_fit()

        if seed is not None:
            np.random.seed(seed)

        if n_timesteps is not None:
            n_years = int(np.ceil(n_timesteps / 12))
        elif n_years is None:
            n_years = len(self.Q_obs_monthly) // 12

        if n_years <= 0:
            raise ValueError(f"n_years must be positive, got {n_years}")

        realizations = {}
        for i in range(n_realizations):
            df = self._generate_one(n_years)
            if n_timesteps is not None:
                df = df.iloc[:n_timesteps]
            realizations[i] = df

        self.logger.info(
            "Generated %d realizations of %d years × %d sites",
            n_realizations, n_years, self._n_sites,
        )
        return Ensemble(realizations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        training_period = (
            str(self.Q_obs_monthly.index[0].date()),
            str(self.Q_obs_monthly.index[-1].date()),
        )
        # Parameters: 12 transitions × (n² for A + n² for B) + 12×n means + 12×n stds
        n = self._n_sites
        n_params = 12 * (2 * n * n + 2 * n)

        return FittedParams(
            means_=self._mu.stack(),
            stds_=self._sigma.stack(),
            correlations_=None,
            distributions_={'type': 'Multivariate Normal with AR(1) structure'},
            transformations_={
                'log_transform': self.log_transform,
                'n_transition_matrices': 12,
            },
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs_monthly),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )
