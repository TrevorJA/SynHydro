"""
Valencia-Schaake temporal disaggregation method.

This module implements the foundational Valencia-Schaake (1973) parametric
temporal disaggregation approach for synthetic hydrology. It disaggregates
an aggregate flow volume (e.g., annual total) into sub-period values
(e.g., 12 monthly flows) using a linear regression model that conditions
sub-period flows on the known aggregate and preserves the conditional
mean and covariance structure.

References
----------
Valencia, R.D., and Schaake, J.C. (1973).
Disaggregation processes in stochastic hydrology.
Water Resources Research, 9(3), 580-585.
https://doi.org/10.1029/WR009i003p00580
"""

import logging
from typing import Union, Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from synhydro.core.base import Disaggregator, DisaggregatorParams, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata


logger = logging.getLogger(__name__)


class ValenciaSchaakeDisaggregator(Disaggregator):
    """
    Temporal disaggregation using the Valencia-Schaake method.

    Disaggregates flows from a coarser temporal resolution to a finer
    resolution (e.g., annual to monthly) using a multivariate normal
    distribution conditioned on the known aggregate. Preserves the
    conditional mean and covariance structure of sub-periods given
    the aggregate.

    Parameters
    ----------
    n_subperiods : int, default=12
        Number of sub-periods per aggregate period (e.g., 12 for
        annual-to-monthly, 4 for annual-to-seasonal).
    transform : str, default='log'
        Transformation applied before fitting: 'log', 'boxcox', or 'none'.
    conservation_method : str, default='proportional'
        Method to enforce sum consistency: 'proportional' or 'none'.
    name : str, optional
        Name identifier for this disaggregator instance.
    debug : bool, default=False
        Enable debug logging.

    Attributes
    ----------
    mu_X_ : np.ndarray
        Mean vector of sub-period flows (n_subperiods,).
    S_XX_ : np.ndarray
        Covariance matrix of sub-period flows (n_subperiods, n_subperiods).
    mu_Y_ : float
        Mean of aggregate flows.
    sigma_Y_sq_ : float
        Variance of aggregate flows.
    A_ : np.ndarray
        Regression coefficients (n_subperiods,).
    C_ : np.ndarray
        Cholesky decomposition of conditional covariance
        (n_subperiods, n_subperiods).
    """

    def __init__(
        self,
        *,
        n_subperiods: int = 12,
        transform: str = "log",
        conservation_method: str = "proportional",
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the Valencia-Schaake Disaggregator.

        Parameters
        ----------
        n_subperiods : int, default=12
            Number of sub-periods per aggregate period.
        transform : str, default='log'
            Transformation: 'log', 'boxcox', or 'none'.
        conservation_method : str, default='proportional'
            Conservation method: 'proportional' or 'none'.
        name : str, optional
            Name for this disaggregator instance.
        debug : bool, default=False
            Enable debug logging.
        """
        super().__init__(name=name, debug=debug)

        self.n_subperiods = n_subperiods
        self.transform = transform
        self.conservation_method = conservation_method

        self.init_params.algorithm_params = {
            "method": "Valencia-Schaake Disaggregation",
            "n_subperiods": n_subperiods,
            "transform": transform,
            "conservation_method": conservation_method,
        }

        # Fitted parameters (will be set during fit())
        self.mu_X_ = None
        self.S_XX_ = None
        self.mu_Y_ = None
        self.sigma_Y_sq_ = None
        self.A_ = None
        self.C_ = None
        self.transform_params_ = {}

    @property
    def input_frequency(self) -> str:
        """Valencia-Schaake expects annual input."""
        return "YS"  # Annual Start

    @property
    def output_frequency(self) -> str:
        """Valencia-Schaake produces monthly output."""
        return "MS"  # Month Start

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Preprocess observed flow data.

        Validates input data and aggregates from finer resolution to
        coarser resolution (e.g., monthly to annual).

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed flow data at finer temporal resolution.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs : Any
            Additional preprocessing parameters (currently unused).
        """
        Q_obs_validated = self._store_obs_data(Q_obs, sites)

        self.Q_obs = Q_obs_validated

        # Aggregate to annual resolution
        self.Q_annual = Q_obs_validated.resample("YS").sum()

        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, "
            f"{len(self.Q_obs)} fine-resolution observations, "
            f"{len(self.Q_annual)} annual observations"
        )

        self.update_state(preprocessed=True)

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the Valencia-Schaake disaggregator.

        Computes sub-period statistics, aggregate statistics, regression
        parameters, and conditional covariance. Applies optional
        transformations and Cholesky decomposition.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs : Any
            Additional fitting parameters (currently unused).
        """
        # Auto-call preprocessing if Q_obs is provided
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)

        self.validate_preprocessing()

        # Organize observed data into sub-period matrix
        X = self._organize_subperiods()

        if X.shape[0] == 0:
            raise ValueError(
                "No complete aggregate periods found in data. "
                "Ensure data has at least one full year."
            )

        # Apply transformation if specified
        if self.transform != "none":
            X = self._apply_transform(X)

        # Compute statistics
        self._compute_statistics(X)

        # Compute Cholesky decomposition
        self._compute_cholesky()

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {X.shape[0]} aggregate periods, "
            f"{X.shape[1]} sub-periods, {self.n_sites} sites"
        )

    def _organize_subperiods(self) -> np.ndarray:
        """
        Organize observed data into sub-period matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_years, n_subperiods, n_sites) where each row
            is a complete aggregate period. If multiple sites, returns
            (n_years, n_subperiods, n_sites).
        """
        years = self.Q_obs.index.year.unique()
        complete_years = []
        X_list = []

        for year in years:
            year_data = self.Q_obs[self.Q_obs.index.year == year]

            # Check if we have a complete year (approximately)
            if len(year_data) < self.n_subperiods:
                continue

            # Aggregate to sub-periods using forward resampling
            # For monthly (n_subperiods=12), use 'MS' (month start)
            if self.n_subperiods == 12:
                subperiod_freq = "MS"
            elif self.n_subperiods == 4:
                subperiod_freq = "QS"  # Quarterly start
            else:
                # General case: estimate frequency
                year_start = pd.Timestamp(year=year, month=1, day=1)
                year_end = pd.Timestamp(year=year, month=12, day=31)
                n_days = (year_end - year_start).days
                days_per_subperiod = n_days / self.n_subperiods
                # Use daily resampling as fallback
                subperiod_freq = "D"

            if self.n_subperiods in [12, 4]:
                year_range = pd.date_range(
                    start=f"{year}-01-01", end=f"{year}-12-31", freq=subperiod_freq
                )
                subperiod_data = (
                    self.Q_obs.loc[(self.Q_obs.index.year == year)]
                    .resample(subperiod_freq)
                    .sum()
                )

                if len(subperiod_data) == self.n_subperiods:
                    X_list.append(subperiod_data.values)
                    complete_years.append(year)
            else:
                # Fallback: manually partition year into n_subperiods
                year_start = pd.Timestamp(year=year, month=1, day=1)
                year_end = pd.Timestamp(year=year + 1, month=1, day=1)
                year_range = pd.date_range(
                    start=year_start, end=year_end, periods=self.n_subperiods + 1
                )

                subperiods = []
                for i in range(self.n_subperiods):
                    mask = (self.Q_obs.index >= year_range[i]) & (
                        self.Q_obs.index < year_range[i + 1]
                    )
                    subperiod_sum = self.Q_obs.loc[mask].sum()
                    subperiods.append(subperiod_sum.values)

                if len(subperiods) == self.n_subperiods:
                    X_list.append(np.array(subperiods))
                    complete_years.append(year)

        if len(X_list) == 0:
            raise ValueError("No complete periods found for disaggregation.")

        X = np.array(X_list)

        if X.ndim == 2:
            # Single site: shape (n_years, n_subperiods)
            self.is_multisite = False
        else:
            # Multiple sites: shape (n_years, n_subperiods, n_sites)
            self.is_multisite = True

        self.logger.debug(
            f"Organized {len(complete_years)} complete years into sub-period matrix"
        )

        return X

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation to sub-period data.

        Parameters
        ----------
        X : np.ndarray
            Sub-period data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        if self.transform == "log":
            # Add small constant to avoid log(0)
            epsilon = 1e-6
            X_transformed = np.log(X + epsilon)
            self.transform_params_["type"] = "log"
            self.transform_params_["epsilon"] = epsilon

        elif self.transform == "boxcox":
            # Apply Box-Cox transformation
            # Flatten to 1D, apply transform, reshape back
            X_flat = X.flatten()
            X_flat = np.clip(X_flat, a_min=1e-6, a_max=None)

            X_transformed_flat, lambda_bc = boxcox(X_flat)
            X_transformed = X_transformed_flat.reshape(X.shape)

            self.transform_params_["type"] = "boxcox"
            self.transform_params_["lambda"] = lambda_bc

        else:
            X_transformed = X.copy()
            self.transform_params_["type"] = "none"

        self.logger.debug(f"Applied {self.transform} transformation")

        return X_transformed

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform from transformed to original scale.

        Parameters
        ----------
        X : np.ndarray
            Data in transformed space.

        Returns
        -------
        np.ndarray
            Data in original scale.
        """
        if self.transform_params_.get("type") == "log":
            epsilon = self.transform_params_.get("epsilon", 1e-6)
            X_inv = np.exp(X) - epsilon
            X_inv = np.clip(X_inv, a_min=0, a_max=None)

        elif self.transform_params_.get("type") == "boxcox":
            lambda_bc = self.transform_params_.get("lambda")
            X_inv = inv_boxcox(X, lambda_bc)
            X_inv = np.clip(X_inv, a_min=0, a_max=None)

        else:
            X_inv = X.copy()

        return X_inv

    def _compute_statistics(self, X: np.ndarray) -> None:
        """
        Compute statistics from sub-period data.

        Computes means, covariances, and regression parameters.

        Parameters
        ----------
        X : np.ndarray
            Sub-period data (n_years, n_subperiods) or
            (n_years, n_subperiods, n_sites).
        """
        if self.is_multisite:
            n_years, n_subperiods, n_sites = X.shape
            # Average across sites for univariate statistics
            X_univariate = X.mean(axis=2)
        else:
            n_years, n_subperiods = X.shape
            X_univariate = X

        # Compute sub-period statistics
        self.mu_X_ = X_univariate.mean(axis=0)  # (n_subperiods,)
        self.S_XX_ = np.cov(X_univariate.T)  # (n_subperiods, n_subperiods)

        # Ensure S_XX_ is 2D even for n_subperiods=1
        if self.S_XX_.ndim == 0:
            self.S_XX_ = np.array([[self.S_XX_]])

        # Compute aggregate (annual) statistics
        Y = X_univariate.sum(axis=1)  # Sum across subperiods for each year
        self.mu_Y_ = Y.mean()
        self.sigma_Y_sq_ = Y.var(ddof=1)

        # Compute regression parameters
        # S_XY = Cov(X, Y) = S_XX @ 1_m (covariance between sub-periods and aggregate)
        ones = np.ones(self.n_subperiods)
        S_XY = self.S_XX_ @ ones

        # A = S_XY / sigma_Y^2 (regression coefficients)
        self.A_ = S_XY / (self.sigma_Y_sq_ + 1e-10)

        self.logger.debug(
            f"Statistics computed: mu_Y={self.mu_Y_:.2f}, "
            f"sigma_Y^2={self.sigma_Y_sq_:.2f}"
        )

    def _compute_cholesky(self) -> None:
        """
        Compute Cholesky decomposition of conditional covariance.

        Conditional covariance: S_e = S_XX - A * sigma_Y^2 * A^T

        If S_e is not positive semi-definite, applies spectral repair.
        """
        # Conditional covariance
        S_e = self.S_XX_ - np.outer(self.A_, self.A_) * self.sigma_Y_sq_

        # Ensure positive semi-definiteness
        S_e = self._repair_covariance(S_e)

        # Cholesky decomposition
        try:
            self.C_ = np.linalg.cholesky(S_e)
        except np.linalg.LinAlgError:
            self.logger.warning(
                "Cholesky decomposition failed. "
                "Applying spectral repair and retrying."
            )
            S_e = self._repair_covariance(S_e, aggressive=True)
            self.C_ = np.linalg.cholesky(S_e)

        self.logger.debug(
            f"Cholesky decomposition computed for {self.n_subperiods}x{self.n_subperiods} matrix"
        )

    def _repair_covariance(self, S: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Repair covariance matrix to ensure positive semi-definiteness.

        Uses spectral decomposition to set negative eigenvalues to small
        positive values.

        Parameters
        ----------
        S : np.ndarray
            Covariance matrix to repair.
        aggressive : bool, default=False
            If True, uses larger positive value replacement.

        Returns
        -------
        np.ndarray
            Repaired covariance matrix.
        """
        eigvals, eigvecs = np.linalg.eigh(S)

        # Set negative eigenvalues to small positive value
        min_eigval = 1e-8 if not aggressive else 1e-4
        eigvals_fixed = np.maximum(eigvals, min_eigval)

        # Reconstruct covariance
        S_repaired = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T

        return S_repaired

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        n_params = (
            self.n_subperiods  # mu_X
            + self.n_subperiods * self.n_subperiods  # S_XX
            + 1  # mu_Y
            + 1  # sigma_Y^2
            + self.n_subperiods  # A
            + self.n_subperiods * self.n_subperiods  # C
        )

        training_period = (
            str(self.Q_obs.index[0].date()),
            str(self.Q_obs.index[-1].date()),
        )

        return FittedParams(
            means_=pd.Series(
                self.mu_X_, index=[f"subperiod_{i}" for i in range(self.n_subperiods)]
            ),
            stds_=pd.Series(
                np.sqrt(np.diag(self.S_XX_)),
                index=[f"subperiod_{i}" for i in range(self.n_subperiods)],
            ),
            correlations_=self._get_correlation_matrix(),
            distributions_={
                "type": "multivariate_normal",
                "method": "Valencia-Schaake",
            },
            fitted_models_={
                "n_subperiods": self.n_subperiods,
                "transform": self.transform,
                "conservation_method": self.conservation_method,
            },
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs),
            n_sites_=self.n_sites,
            training_period_=training_period,
        )

    def _get_correlation_matrix(self) -> np.ndarray:
        """
        Compute correlation matrix from covariance.

        Returns
        -------
        np.ndarray
            Correlation matrix.
        """
        stds = np.sqrt(np.diag(self.S_XX_))
        denom = np.outer(stds, stds)
        denom[denom == 0] = 1.0
        corr = self.S_XX_ / denom
        return corr

    def disaggregate(
        self, ensemble: Ensemble, seed: Optional[int] = None, **kwargs: Any
    ) -> Ensemble:
        """
        Disaggregate annual ensemble to monthly.

        Parameters
        ----------
        ensemble : Ensemble
            Input ensemble at annual resolution. Must have frequency 'YS'.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : Any
            Additional disaggregation parameters.

        Returns
        -------
        Ensemble
            Disaggregated ensemble at monthly resolution.

        Raises
        ------
        ValueError
            If ensemble is not at expected input frequency.
        """
        self.validate_fit()
        self.validate_input_ensemble(ensemble)

        rng = np.random.default_rng(seed)

        # Disaggregate each realization
        monthly_realization_dict = {}

        for realization_id, annual_df in ensemble.data_by_realization.items():
            monthly_df = self._disaggregate_single_realization(annual_df, rng=rng)
            monthly_realization_dict[realization_id] = monthly_df

        # Create metadata for monthly ensemble
        metadata = EnsembleMetadata(
            generator_class=ensemble.metadata.generator_class,
            generator_params=ensemble.metadata.generator_params,
            n_realizations=len(monthly_realization_dict),
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(
                str(monthly_realization_dict[0].index[0].date()),
                str(monthly_realization_dict[0].index[-1].date()),
            ),
        )

        monthly_ensemble = Ensemble(monthly_realization_dict, metadata=metadata)

        self.logger.info(
            f"Disaggregated {len(monthly_realization_dict)} realizations "
            f"from annual to monthly"
        )

        return monthly_ensemble

    def _disaggregate_single_realization(
        self, annual_df: pd.DataFrame, *, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Disaggregate a single realization from annual to monthly.

        Parameters
        ----------
        annual_df : pd.DataFrame
            Annual flows for a single realization.
            Shape: (n_years, n_sites).

        Returns
        -------
        pd.DataFrame
            Monthly disaggregated flows.
            Shape: (n_months, n_sites).
        """
        # Extract annual flows
        Y_syn = (
            annual_df.iloc[:, 0].values
            if annual_df.shape[1] == 1
            else annual_df.sum(axis=1).values
        )

        n_years = len(Y_syn)
        monthly_flows = []
        monthly_dates = []

        for i, (year_idx, annual_flow) in enumerate(zip(annual_df.index, Y_syn)):
            # Extract year from timestamp
            year = annual_df.index[i].year

            # Sample monthly flows for this year
            X_month = self._sample_conditional_distribution(annual_flow, rng=rng)

            # Inverse transform
            if self.transform != "none":
                X_month = self._inverse_transform(X_month)

            # Proportional adjustment to enforce sum consistency
            if self.conservation_method == "proportional":
                X_month_sum = X_month.sum()
                if X_month_sum > 0:
                    X_month = X_month * (annual_flow / X_month_sum)

            # Enforce non-negativity
            X_month = np.clip(X_month, 0, None)

            # Store monthly flows
            monthly_flows.append(X_month)

            # Create monthly dates
            for month in range(1, self.n_subperiods + 1):
                monthly_dates.append(pd.Timestamp(year=year, month=month, day=1))

        # Combine into DataFrame
        monthly_flows_array = np.array(monthly_flows).reshape(-1, annual_df.shape[1])

        monthly_df = pd.DataFrame(
            monthly_flows_array,
            index=pd.DatetimeIndex(monthly_dates),
            columns=annual_df.columns,
        )

        return monthly_df

    def _sample_conditional_distribution(
        self, Y_syn: float, *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample from conditional distribution given aggregate.

        Computes conditional mean and samples from conditional normal
        distribution: X_syn ~ N(mu_X|Y, S_e).

        Parameters
        ----------
        Y_syn : float
            Synthetic aggregate (annual) flow value.

        Returns
        -------
        np.ndarray
            Sampled sub-period flows (n_subperiods,).
        """
        # Conditional mean: mu_X|Y = mu_X + A * (Y_syn - mu_Y)
        mu_X_given_Y = self.mu_X_ + self.A_ * (Y_syn - self.mu_Y_)

        # Sample residuals: Z ~ N(0, I_m)
        Z = rng.standard_normal(self.n_subperiods)

        # X_syn = mu_X|Y + C @ Z
        X_syn = mu_X_given_Y + self.C_ @ Z

        return X_syn
