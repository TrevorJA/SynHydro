"""
Kirsch Nonparametric Bootstrap Generator (Kirsch et al. 2013)

Generates synthetic multi-site monthly streamflow by bootstrapping standardized
residuals and imposing fitted correlation structure via Cholesky decomposition.
A cross-year shifted matrix preserves December-to-January correlations.

References
----------
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact
of alternative hydro-climate scenarios on transfer agreements: A practical
improvement for generating synthetic streamflows. Journal of Water Resources
Planning and Management, 139(4), 396-406.
https://doi.org/10.1061/(ASCE)WR.1943-5452.0000287
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm

from synhydro.core.base import Generator, GeneratorParams, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.statistics import (
    compute_monthly_statistics,
    repair_correlation_matrix,
)


class KirschGenerator(Generator):
    """
    Kirsch nonparametric bootstrap generator for monthly streamflow synthesis.

    Generates monthly synthetic flows using bootstrap resampling with
    correlation preservation via Cholesky decomposition.

    References
    ----------
    Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the
    impact of alternative hydro-climate scenarios on transfer agreements.
    Journal of Water Resources Planning and Management, 139(4), 396-406.
    """

    supports_multisite = True
    supported_frequencies = ("MS",)

    def __init__(
        self,
        *,
        generate_using_log_flow=True,
        matrix_repair_method="spectral",
        name=None,
        debug=False,
        **kwargs,
    ):
        """
        Initialize Kirsch generator.

        Parameters
        ----------
        generate_using_log_flow : bool, default=True
            If True, generates in log-space for better handling of skewed distributions.
        matrix_repair_method : str, default='spectral'
            Method for repairing non-positive-definite correlation matrices.
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        """
        super().__init__(name=name, debug=debug)

        # Store algorithm-specific parameters
        self.generate_using_log_flow = generate_using_log_flow
        self.matrix_repair_method = matrix_repair_method

        # Update init_params
        self.init_params.algorithm_params = {
            "method": "Kirsch",
            "generate_using_log_flow": generate_using_log_flow,
            "matrix_repair_method": matrix_repair_method,
        }

        self.n_months = 12
        self.U_site = {}
        self.U_prime_site = {}

    @property
    def output_frequency(self) -> str:
        """Kirsch generator produces monthly output."""
        return "MS"  # Month Start

    @property
    def Q_obs_monthly(self):
        """Get observed monthly data (alias for Qm for consistency with other generators)."""
        if not hasattr(self, "Qm"):
            raise AttributeError(
                "Generator must be preprocessed before accessing Q_obs_monthly"
            )

        # Convert back from log space if needed
        if self.generate_using_log_flow:
            data = np.exp(self.Qm)
        else:
            data = self.Qm

        # Convert MultiIndex (year, month) to DatetimeIndex for compatibility with plotting
        if isinstance(data.index, pd.MultiIndex):
            # Create DatetimeIndex from year/month MultiIndex
            dates = pd.to_datetime(
                [f"{year}-{month:02d}-01" for year, month in data.index]
            )
            data = data.copy()
            data.index = dates

        return data

    def _get_synthetic_index(self, n_years):
        """Generate DatetimeIndex for synthetic flows."""
        return pd.date_range(
            start=f"{self.Q.index.year.max() + 1}-01-01",
            periods=n_years * self.n_months,
            freq="MS",
        )

    def preprocessing(self, Q_obs, *, sites=None, timestep="monthly", **kwargs):
        """
        Preprocess observed data for Kirsch generation.

        Parameters
        ----------
        Q_obs : pd.DataFrame
            Observed historical flow data with DatetimeIndex.
        sites : list, optional
            Sites to use. If None, uses all sites.
        timestep : str, default='monthly'
            Currently only 'monthly' is supported.
        **kwargs
            Additional preprocessing parameters.
        """
        if timestep != "monthly":
            raise NotImplementedError("Currently only monthly timestep is supported.")

        Q = self._store_obs_data(Q_obs, sites)

        # Aggregate to monthly
        monthly = Q.groupby([Q.index.year, Q.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(
            monthly.index, names=["year", "month"]
        )
        self.Qm = monthly

        # Apply log transformation if requested
        if self.generate_using_log_flow:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))

        # Store for later use
        self.Q = Q  # Keep reference for synthetic index generation
        self.n_historic_years = Q.index.year.nunique()

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, {self.n_historic_years} years"
        )

    def fit(self, Q_obs=None, *, sites=None, **kwargs):
        """
        Fit Kirsch generator to preprocessed data.

        Parameters
        ----------
        Q_obs : pd.DataFrame, optional
            If provided, calls preprocessing automatically.
        sites : list, optional
            Sites to use (passed to preprocessing if Q_obs provided).
        **kwargs
            Additional fitting parameters.
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        # Compute monthly statistics using centralized function
        # Need to convert from MultiIndex to DatetimeIndex for compute_monthly_statistics
        temp_df = self.Qm.copy()
        temp_df.index = pd.to_datetime(
            temp_df.index.get_level_values("year").astype(str)
            + "-"
            + temp_df.index.get_level_values("month").astype(str).str.zfill(2)
            + "-01"
        )
        monthly_stats = compute_monthly_statistics(temp_df)
        self.mean_month = monthly_stats["mean"]
        self.std_month = monthly_stats["std"]

        years = self.Qm.index.get_level_values("year").unique()
        Z_h = []
        valid_years = []

        for year in years:
            try:
                year_data = []
                for m in range(1, 13):
                    row = (
                        (self.Qm.loc[(year, m)] - self.mean_month.loc[m])
                        / self.std_month.loc[m]
                    ).values
                    year_data.append(row)
                Z_h.append(year_data)
                valid_years.append(year)
            except KeyError:
                continue

        self.Z_h = np.array(Z_h)  # shape: (n_years, 12, n_sites)
        self.historic_years = np.array(valid_years)
        self.n_historic_years = len(valid_years)

        # Apply normal score transform when using log flow to prevent
        # bias from non-Gaussian standardized residuals interacting with exp()
        if self.generate_using_log_flow:
            self._fit_normal_score_transforms()
            self.Y = self._apply_normal_score_transform(self.Z_h)
        else:
            self.Y = self.Z_h.copy()
        self.Y_prime = np.zeros_like(self.Y[:-1])
        self.Y_prime[:, :6, :] = self.Y[:-1, 6:, :]
        self.Y_prime[:, 6:, :] = self.Y[1:, :6, :]

        for s in range(self.n_sites):
            y_s = self.Y[:, :, s]  # shape: (n_years, 12)
            y_prime_s = self.Y_prime[:, :, s]

            corr_s = np.corrcoef(y_s.T)  # shape: (12, 12)
            corr_prime_s = np.corrcoef(y_prime_s.T)

            self.U_site[s] = self._repair_and_cholesky(corr_s)
            self.U_prime_site[s] = self._repair_and_cholesky(corr_prime_s)

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {self.n_historic_years} years, {self.n_sites} sites"
        )

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters from Kirsch generator.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        # Count parameters:
        # - mean and std for each site for each month: n_sites × 12 × 2
        # - Cholesky matrices for each site: n_sites × 12 × 12 (two matrices)
        n_params = (self.n_sites * 12 * 2) + (self.n_sites * 12 * 12 * 2)

        # Get training period from original data
        training_period = (str(self.Q.index[0].date()), str(self.Q.index[-1].date()))

        # Package correlation matrices
        corr_matrices = {"U_site": self.U_site, "U_prime_site": self.U_prime_site}

        return FittedParams(
            means_=self.mean_month,
            stds_=self.std_month,
            correlations_=corr_matrices,
            distributions_=None,  # Nonparametric, no distributional assumptions
            n_parameters_=n_params,
            sample_size_=self.n_historic_years * 12,  # Total months
            n_sites_=self.n_sites,
            training_period_=training_period,
        )

    def _repair_and_cholesky(self, corr):
        """
        Given a correlation matrix, 'repair' the matrix so it is PSD, and return its Cholesky decomposition.

        Uses the centralized repair_correlation_matrix function from synhydro.core.statistics.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix to be repaired and decomposed.

        Returns
        -------
        np.ndarray
            Cholesky decomposition of the correlation matrix.
        """
        try:
            return np.linalg.cholesky(corr).T
        except np.linalg.LinAlgError:
            self.logger.warning(
                "Matrix not positive definite, repairing... This may cause correlation inflation."
            )
            # Use centralized repair function
            repaired_corr = repair_correlation_matrix(
                corr, method=self.matrix_repair_method
            )
            return np.linalg.cholesky(repaired_corr).T

    def _fit_normal_score_transforms(self):
        """
        Compute empirical CDF mappings for normal score transform per month per site.

        For each (month, site), stores the sorted standardized residuals and their
        corresponding normal scores (via Hazen plotting position). These are used
        during generation to map Cholesky-mixed values back to the correct
        month-specific distribution, preventing bias from the exp() back-transform.
        """
        n_years = self.Z_h.shape[0]
        self._nst_sorted = {}
        self._nst_nscores = {}

        pp = (np.arange(1, n_years + 1) - 0.5) / n_years
        nscores = norm.ppf(pp)

        for m in range(self.n_months):
            for s in range(self.n_sites):
                sorted_vals = np.sort(self.Z_h[:, m, s])
                self._nst_sorted[(m, s)] = sorted_vals
                self._nst_nscores[(m, s)] = nscores

    def _apply_normal_score_transform(self, Z):
        """
        Forward transform: standardized residuals -> normal scores.

        Parameters
        ----------
        Z : np.ndarray
            Standardized residuals with shape (n_years, 12, n_sites).

        Returns
        -------
        np.ndarray
            Normal-scored residuals with same shape.
        """
        Z_norm = np.zeros_like(Z)
        for m in range(self.n_months):
            for s in range(self.n_sites):
                Z_norm[:, m, s] = np.interp(
                    Z[:, m, s], self._nst_sorted[(m, s)], self._nst_nscores[(m, s)]
                )
        return Z_norm

    def _apply_inverse_normal_score_transform(self, ZC):
        """
        Inverse transform: normal scores -> original standardized space per target month.

        Uses linear extrapolation at tails so synthetic values beyond the
        observed range are not clamped.

        Parameters
        ----------
        ZC : np.ndarray
            Normal-scored combined tensor with shape (n_years, 12, n_sites).

        Returns
        -------
        np.ndarray
            Standardized residuals in original space with same shape.
        """
        ZC_orig = np.zeros_like(ZC)
        for m in range(self.n_months):
            for s in range(self.n_sites):
                ns = self._nst_nscores[(m, s)]
                sv = self._nst_sorted[(m, s)]
                # Linear extrapolation at tails
                slope_lo = (sv[1] - sv[0]) / (ns[1] - ns[0]) if ns[1] != ns[0] else 1.0
                slope_hi = (
                    (sv[-1] - sv[-2]) / (ns[-1] - ns[-2]) if ns[-1] != ns[-2] else 1.0
                )
                ns_ext = np.concatenate([[-10.0], ns, [10.0]])
                sv_ext = np.concatenate(
                    [
                        [sv[0] + slope_lo * (-10.0 - ns[0])],
                        sv,
                        [sv[-1] + slope_hi * (10.0 - ns[-1])],
                    ]
                )
                ZC_orig[:, m, s] = np.interp(ZC[:, m, s], ns_ext, sv_ext)
        return ZC_orig

    def _get_bootstrap_indices(self, n_years, max_idx=None, rng=None):
        """
        Return 'M', a matrix of bootstrap indices for the synthetic time series.

        Parameters
        ----------
        n_years : int
            Number of years for which to generate bootstrap indices.
        max_idx : int, optional
            Maximum index for the historic years. If None, uses the number of historic years.
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates a new default generator.

        Returns
        -------
        np.ndarray
            A matrix of shape (n_years, n_months) containing bootstrap indices.
        """
        if rng is None:
            rng = np.random.default_rng()
        max_idx = self.n_historic_years if max_idx is None else max_idx
        return rng.choice(max_idx, size=(n_years, self.n_months), replace=True)

    def _create_bootstrap_tensor(self, M, use_Y_prime=False):
        """
        Create the 'Z' tensor of boostrapped standardized flows.

        Parameters
        ----------
        M : np.ndarray
            Bootstrap indices with shape (n_years, n_months).
        use_Y_prime : bool
            If True, uses Y_prime; otherwise uses Y.

        Returns
        -------
        np.ndarray
            Bootstrap tensor with shape (n_years, n_months, n_sites).
        """
        source = self.Y_prime if use_Y_prime else self.Y
        n_years, n_months = M.shape
        max_idx = source.shape[0]
        output = np.zeros((n_years, n_months, self.n_sites))
        for i in range(n_years):
            for m in range(n_months):
                h_idx = M[i, m]
                if h_idx >= max_idx:
                    h_idx = max_idx - 1
                output[i, m] = source[h_idx, m]
        return output

    def _apply_cholesky_and_combine(self, X, X_prime, n_years):
        """
        Apply Cholesky decomposition and combine Z with Z_prime tensors.

        This is the core pipeline: apply Cholesky to decorrelated residuals,
        then combine across years to preserve Dec-Jan transitions.

        Parameters
        ----------
        X : np.ndarray
            Bootstrap standardized residuals with shape (n_years+1, 12, n_sites).
        X_prime : np.ndarray
            Bootstrap standardized residuals from Y_prime with shape (n_years+1, 12, n_sites).
        n_years : int
            Number of years for the output (before buffering).

        Returns
        -------
        np.ndarray
            Combined and Cholesky-transformed flows with shape (n_years, 12, n_sites).
        """
        Z = np.zeros_like(X)
        Z_prime = np.zeros_like(X_prime)

        for s in range(self.n_sites):
            Z[:, :, s] = X[:, :, s] @ self.U_site[s]
            Z_prime[:, :, s] = X_prime[:, :, s] @ self.U_prime_site[s]

        ZC = self._combine_Z_and_Z_prime(Z, Z_prime)
        return ZC

    def _combine_Z_and_Z_prime(self, Z, Z_prime):
        """
        Combine Z and Z_prime into a single tensor, to preserve intra-year correlations.

        Parameters
        ----------
        Z : np.ndarray
            Standardized flows for the first half of the year with shape (n_years, n_months, n_sites).
        Z_prime : np.ndarray
            Standardized flows for the second half of the year with shape (n_years, n_months, n_sites).

        Returns
        -------
        np.ndarray
            Combined standardized flows with shape (n_years-1, n_months, n_sites).
        """
        # We can combine up to (Z.shape[0] - 1) years since we need year i and year i+1
        n_years = Z.shape[0] - 1
        ZC = np.zeros((n_years, self.n_months, self.n_sites))
        ZC[:, :6, :] = Z_prime[:n_years, 6:, :]
        ZC[:, 6:, :] = Z[1 : n_years + 1, 6:, :]
        return ZC

    def _destandardize_flows(self, Z_combined):
        """
        Reapply mean and standard deviation to standardized flows.

        Parameters
        ----------
        Z_combined : np.ndarray
            Standardized flows with shape (n_years, n_months, n_sites).

        Returns
        -------
        np.ndarray
            Destandardized flows with shape (n_years, n_months, n_sites).
        """
        Q_syn = np.zeros_like(Z_combined)
        for m in range(self.n_months):
            Q_syn[:, m, :] = (
                Z_combined[:, m, :] * self.std_month.iloc[m].values
                + self.mean_month.iloc[m].values
            )
        return Q_syn

    def _reshape_output(self, Q_syn):
        return Q_syn.reshape(-1, self.n_sites)

    def generate_from_indices(self, indices, n_years=None, as_array=True,
                               synthetic_index=None):
        """
        Generate synthetic flows by directly specifying historical year indices.

        This method allows external code (e.g., MOEA-FIND) to inject decision
        variables (year indices) instead of random sampling. Runs the full
        post-bootstrap pipeline: Cholesky, normal-score inversion, re-seasonalization.

        Parameters
        ----------
        indices : np.ndarray
            Array of historical year indices to resample. Shape (n_years+1, 12)
            where each entry is in [0, n_historic_years). The extra year allows
            Dec-Jan cross-year correlation handling. Can be floats (will be cast to int).
        n_years : int, optional
            Number of years for the synthetic output. If None, inferred from
            indices.shape[0] - 1.
        as_array : bool, default=True
            If True, returns numpy array; if False, returns pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom DatetimeIndex for the output. If None, a default index is generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic monthly flows with shape (n_years * 12, n_sites) if as_array=True,
            otherwise a pandas DataFrame.

        Notes
        -----
        This method assumes the generator has been fitted. Indices are treated as
        indices into the historic years array (self.historic_years or [0, 1, ..., n-1]).
        """
        self.validate_fit()

        indices = np.asarray(indices, dtype=int)

        if n_years is None:
            n_years = indices.shape[0] - 1

        if indices.shape != (n_years + 1, self.n_months):
            raise ValueError(
                f"indices must have shape ({n_years + 1}, {self.n_months}), "
                f"got {indices.shape}"
            )

        # Clamp indices to valid range
        indices = np.clip(indices, 0, self.Y.shape[0] - 1)

        # Create bootstrap tensors using provided indices for Y
        X = self._create_bootstrap_tensor(indices, use_Y_prime=False)

        # For Y_prime, we need separate clamping to its valid range
        indices_prime = np.clip(indices, 0, self.Y_prime.shape[0] - 1)
        X_prime = self._create_bootstrap_tensor(indices_prime, use_Y_prime=True)

        # Apply Cholesky and combine
        ZC = self._apply_cholesky_and_combine(X, X_prime, n_years)

        # Inverse normal score transform
        if self.generate_using_log_flow:
            ZC = self._apply_inverse_normal_score_transform(ZC)

        # Destandardize and exponentiate
        Q_syn = self._destandardize_flows(ZC)
        if self.generate_using_log_flow:
            Q_syn = np.exp(Q_syn)

        Q_flat = self._reshape_output(Q_syn)

        if as_array:
            return Q_flat
        else:
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            return pd.DataFrame(Q_flat, columns=self._sites, index=synthetic_index)

    def generate_from_residuals(self, residuals, as_array=True, synthetic_index=None):
        """
        Generate synthetic flows from pre-computed standardized residuals.

        This method allows external code (e.g., MOEA-FIND) to inject decision
        variables (standardized residuals) directly, bypassing the bootstrap
        resampling step. Runs steps 4-8 of the Kirsch pipeline:
        normal-score transform, Cholesky, inverse normal-score, Dec-Jan combination,
        and re-seasonalization.

        Parameters
        ----------
        residuals : np.ndarray
            Array of standardized residuals with shape (n_years, 12, n_sites).
            Each residual should be approximately N(0,1) or representable as such
            within month-specific empirical distributions.
        as_array : bool, default=True
            If True, returns numpy array; if False, returns pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom DatetimeIndex for the output. If None, a default index is generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic monthly flows with shape (n_years * 12, n_sites) if as_array=True,
            otherwise a pandas DataFrame.

        Notes
        -----
        This method assumes the generator has been fitted. Residuals are assumed
        to be standardized residuals; they will be normal-score transformed,
        processed through Cholesky factors, and combined to preserve Dec-Jan correlations.
        """
        self.validate_fit()

        residuals = np.asarray(residuals)

        if residuals.ndim != 3 or residuals.shape[1:] != (self.n_months, self.n_sites):
            raise ValueError(
                f"residuals must have shape (n_years, {self.n_months}, {self.n_sites}), "
                f"got {residuals.shape}"
            )

        n_years = residuals.shape[0]

        # For the pipeline we need a buffered year (n_years+1)
        # Create buffer by repeating the last year
        X = np.zeros((n_years + 1, self.n_months, self.n_sites))
        X[:n_years] = residuals
        X[n_years] = residuals[-1]  # repeat last year for buffer

        # Create X_prime by shifting (Dec-Jan transitions)
        # X_prime mirrors Y_prime structure: has n_years shape
        # Then _apply_cholesky_and_combine will output n_years shape from both
        X_prime_temp = np.zeros((n_years, self.n_months, self.n_sites))
        X_prime_temp[:, :6, :] = X[:-1, 6:, :]
        X_prime_temp[:, 6:, :] = X[1:, :6, :]

        # But _apply_cholesky_and_combine expects both to be (n_years+1, 12, n_sites)
        # So pad X_prime_temp back to n_years+1 by repeating last year
        X_prime = np.zeros((n_years + 1, self.n_months, self.n_sites))
        X_prime[:n_years] = X_prime_temp
        X_prime[n_years] = X_prime_temp[-1]

        # Apply Cholesky and combine
        ZC = self._apply_cholesky_and_combine(X, X_prime, n_years)

        # Inverse normal score transform
        if self.generate_using_log_flow:
            ZC = self._apply_inverse_normal_score_transform(ZC)

        # Destandardize and exponentiate
        Q_syn = self._destandardize_flows(ZC)
        if self.generate_using_log_flow:
            Q_syn = np.exp(Q_syn)

        Q_flat = self._reshape_output(Q_syn)

        if as_array:
            return Q_flat
        else:
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            return pd.DataFrame(Q_flat, columns=self._sites, index=synthetic_index)

    def generate_single_series(
        self, n_years, M=None, as_array=True, synthetic_index=None, rng=None
    ):
        """
        Generate a single synthetic time series.

        Parameters
        ----------
        n_years : int
            Number of years for the synthetic time series.
        M : np.ndarray, optional
            Bootstrap indices for the synthetic time series. If None, random indices will be generated.
        as_array : bool
            If True, returns a numpy array; if False, returns a pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom index for the synthetic time series. If None, a default index will be generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic time series data.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_years_buffered = n_years + 1

        if M is None:
            M = self._get_bootstrap_indices(
                n_years_buffered, max_idx=self.Y.shape[0], rng=rng
            )
        else:
            M = np.asarray(M)
            if M.shape != (n_years_buffered, self.n_months):
                raise ValueError(
                    f"M must have shape ({n_years_buffered}, {self.n_months})"
                )

        # Generate separate bootstrap indices for Y_prime with correct bounds
        # Y_prime has one fewer year than Y due to cross-year correlation structure
        # Using M.copy() would allow indices up to Y.shape[0]-1, but Y_prime only has indices up to Y.shape[0]-2
        M_prime = self._get_bootstrap_indices(
            n_years_buffered, max_idx=self.Y_prime.shape[0], rng=rng
        )

        X = self._create_bootstrap_tensor(M, use_Y_prime=False)
        X_prime = self._create_bootstrap_tensor(M_prime, use_Y_prime=True)

        ZC = self._apply_cholesky_and_combine(X, X_prime, n_years)

        # Inverse normal score transform: map from N(0,1) back to
        # month-specific standardized distributions before destandardization
        if self.generate_using_log_flow:
            ZC = self._apply_inverse_normal_score_transform(ZC)

        Q_syn = self._destandardize_flows(ZC)

        if self.generate_using_log_flow:
            Q_syn = np.exp(Q_syn)

        Q_flat = self._reshape_output(Q_syn)

        if as_array:
            return Q_flat
        else:
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            return pd.DataFrame(Q_flat, columns=self._sites, index=synthetic_index)

    def generate(
        self, n_realizations=1, n_years=None, n_timesteps=None, seed=None, **kwargs
    ):
        """
        Generate an ensemble of synthetic monthly flows.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic time series to generate.
        n_years : int, optional
            Number of years for each synthetic time series. If None, uses the number of historic years.
        n_timesteps : int, optional
            Not used (Kirsch generates by years, not timesteps).
        seed : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional generation parameters.

        Returns
        -------
        Ensemble
            Ensemble object containing all generated realizations.
        """
        # Validate fit
        self.validate_fit()

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Determine number of years
        if n_years is None:
            n_years = self.n_historic_years

        # Generate realizations
        realization_dict = {}
        for i in range(n_realizations):
            df = self.generate_single_series(n_years, as_array=False, rng=rng)
            realization_dict[i] = df

        # Create metadata
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self.n_sites,
            time_resolution=self.output_frequency,
            time_period=(
                str(realization_dict[0].index[0].date()),
                str(realization_dict[0].index[-1].date()),
            ),
        )

        # Create and return Ensemble
        ensemble = Ensemble(realization_dict, metadata=metadata)

        self.logger.info(
            f"Generated {n_realizations} realizations of {n_years} years each"
        )

        return ensemble
