"""
KNN Bootstrap Generator (Lall and Sharma 1996)

Generates synthetic streamflow by conditional resampling using K-Nearest Neighbors.
At each timestep, the current flow determines a neighborhood of K similar historical
states, and the next value is drawn from the successors of those neighbors using
kernel-weighted probabilities. Preserves empirical marginal distributions and nonlinear
dependence structures that parametric models may miss.

For multisite applications, all sites are resampled jointly using the same selected
neighbor index, preserving spatial correlation by construction.

References
----------
Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling hydrologic
time series. Water Resources Research, 32(3), 679-693.
https://doi.org/10.1029/95WR02966

Rajagopalan, B., and Lall, U. (1999). A k-nearest-neighbor simulator for daily
precipitation and other weather variables. Water Resources Research, 35(10), 3089-3101.
https://doi.org/10.1029/1999WR900028
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from sklearn.neighbors import NearestNeighbors

from synhydro.core.base import Generator, GeneratorParams, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata


logger = logging.getLogger(__name__)


class KNNBootstrapGenerator(Generator):
    """
    K-Nearest Neighbor bootstrap generator for synthetic streamflow.

    Conditionally resamples from historical record by finding K nearest neighbors
    to the current state and selecting successor values with Lall-Sharma kernel weights.

    References
    ----------
    Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling
    hydrologic time series. Water Resources Research, 32(3), 679-693.
    """

    def __init__(
        self,
        Q_obs: pd.DataFrame,
        n_neighbors: Optional[int] = None,
        feature_cols: Optional[List[str]] = None,
        index_site: Optional[str] = None,
        block_size: int = 1,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize KNN Bootstrap generator.

        Parameters
        ----------
        Q_obs : pd.DataFrame
            Observed historical flow data with DatetimeIndex.
        n_neighbors : int, optional
            Number of neighbors K. If None, uses ceil(sqrt(n)) where n is the
            number of historical timesteps.
        feature_cols : list, optional
            Column names to use as features for KNN search. If None, uses all columns.
        index_site : str, optional
            Site name to use for distance computation in multisite mode. If None,
            uses multivariate distance across all feature columns.
        block_size : int, default=1
            Number of consecutive timesteps to resample as a block (1 = standard KNN).
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : Any
            Additional parameters (stored but not used).
        """
        super().__init__(Q_obs=Q_obs, name=name, debug=debug)

        # Store algorithm-specific parameters
        self.n_neighbors = n_neighbors
        self.feature_cols = feature_cols
        self.index_site = index_site
        self.block_size = block_size

        # Update init_params
        self.init_params.algorithm_params = {
            'method': 'KNNBootstrap',
            'n_neighbors': n_neighbors,
            'feature_cols': feature_cols,
            'index_site': index_site,
            'block_size': block_size,
        }

        # Will be initialized during preprocessing/fitting
        self._knn_model = None
        self._feature_vectors = None  # Historical feature vectors for KNN
        self._successor_values = None  # Q_{t+1} for each historical t
        self._kernel_weights = None   # Lall-Sharma kernel weights
        self._Q_obs = None
        self._frequency = None

    @property
    def output_frequency(self) -> str:
        """
        Return temporal frequency of generated output.

        Detected from input data frequency (monthly or annual).
        """
        if self._frequency is None:
            raise ValueError("Run preprocessing() first to access output_frequency")
        return self._frequency

    def preprocessing(
        self,
        sites: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Preprocess and validate observed flow data.

        Constructs feature vectors for KNN search and successor pairs. Also detects
        the temporal frequency of the data.

        Parameters
        ----------
        sites : list, optional
            Sites to use. If None, uses all columns.
        **kwargs : Any
            Additional preprocessing parameters.
        """
        # Validate input data
        Q_obs = self.validate_input_data(self._Q_obs_raw)

        # Select sites if specified
        if sites is not None:
            missing = set(sites) - set(Q_obs.columns)
            if missing:
                raise ValueError(f"Sites not found in data: {missing}")
            Q_obs = Q_obs[sites]

        # Store preprocessed data and sites
        self._Q_obs = Q_obs
        self._sites = Q_obs.columns.tolist()

        # Determine temporal frequency
        self._detect_frequency()

        # Determine feature columns for KNN
        if self.feature_cols is None:
            self._feature_cols = self._sites
        else:
            # Validate that specified feature cols exist
            missing = set(self.feature_cols) - set(self._sites)
            if missing:
                raise ValueError(
                    f"Specified feature_cols {missing} not found in data columns {self._sites}"
                )
            self._feature_cols = self.feature_cols

        # Validate index_site if specified
        if self.index_site is not None and self.index_site not in self._sites:
            raise ValueError(
                f"index_site '{self.index_site}' not found in data columns {self._sites}"
            )

        # Determine number of neighbors
        n_historical = len(self._Q_obs)
        if self.n_neighbors is None:
            self._n_neighbors = max(1, int(np.ceil(np.sqrt(n_historical))))
        else:
            self._n_neighbors = self.n_neighbors

        if self._n_neighbors >= n_historical:
            self.logger.warning(
                f"n_neighbors ({self._n_neighbors}) >= number of historical timesteps ({n_historical}). "
                f"Setting n_neighbors to {n_historical - 1}."
            )
            self._n_neighbors = n_historical - 1

        # Build feature vectors and successor pairs
        self._build_feature_successor_pairs()

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, {n_historical} timesteps, "
            f"n_neighbors={self._n_neighbors}, frequency={self._frequency}"
        )

    def _detect_frequency(self) -> None:
        """
        Detect temporal frequency of the data (monthly or annual).

        Sets self._frequency to 'MS' (month start) or 'YS' (annual start).
        """
        if len(self._Q_obs) < 2:
            self._frequency = 'MS'  # Default
            return

        # Get time differences between consecutive timestamps
        time_diffs = self._Q_obs.index[1:] - self._Q_obs.index[:-1]
        median_diff = np.median([td.days for td in time_diffs])

        if median_diff < 10:  # Daily or weekly
            self._frequency = 'D'
        elif median_diff < 200:  # Monthly
            self._frequency = 'MS'
        else:  # Annual
            self._frequency = 'YS'

        self.logger.debug(f"Detected frequency: {self._frequency} (median diff: {median_diff} days)")

    def _build_feature_successor_pairs(self) -> None:
        """
        Build feature vectors and successor pairs for KNN.

        For monthly data, pairs are grouped by calendar month so that the
        neighbor search at generation time is conditioned on the current month
        (Rajagopalan & Lall 1999). For each month m, the feature vector is the
        flow at month m and the successor is the flow at month m+1.

        For annual or daily data, a single global pool is used.
        """
        features_all = self._Q_obs[self._feature_cols].values[:-1]
        successors_all = self._Q_obs.values[1:]
        months_all = self._Q_obs.index[:-1].month  # month of the feature row

        self._is_monthly_conditioned = self._frequency in ('MS', 'M', 'ME')

        if self._is_monthly_conditioned:
            # Build per-month pools
            self._monthly_features = {}
            self._monthly_successors = {}
            for m in range(1, 13):
                mask = months_all == m
                self._monthly_features[m] = features_all[mask]
                self._monthly_successors[m] = successors_all[mask]
                self.logger.debug(
                    "Month %d: %d feature-successor pairs", m, mask.sum()
                )
            # Also keep global pool as fallback
            self._feature_vectors = features_all
            self._successor_values = successors_all
        else:
            self._feature_vectors = features_all
            self._successor_values = successors_all

        self.logger.debug(
            "Built %d feature-successor pairs (features: %d, successors: %d)",
            len(features_all), features_all.shape[1], successors_all.shape[1],
        )

    def _make_kernel_weights(self, k: int) -> np.ndarray:
        """
        Compute Lall-Sharma kernel weights for *k* neighbors.

        Parameters
        ----------
        k : int
            Number of neighbors.

        Returns
        -------
        np.ndarray
            Probability weights summing to 1.
        """
        harmonic_sum = np.sum([1.0 / i for i in range(1, k + 1)])
        return np.array([1.0 / (i + 1) / harmonic_sum for i in range(k)])

    def fit(self, **kwargs: Any) -> None:
        """
        Fit KNN model(s) to preprocessed data.

        For monthly data, fits 12 separate KNN models — one per calendar month —
        so that the neighbor search is conditioned on month (Rajagopalan & Lall
        1999). For annual or daily data, fits a single global model.

        Also computes Lall-Sharma kernel weights for neighbor selection.

        Parameters
        ----------
        **kwargs : Any
            Additional fitting parameters.
        """
        self.validate_preprocessing()

        if self._is_monthly_conditioned:
            # Per-month KNN models
            self._monthly_knn = {}
            self._monthly_weights = {}
            for m in range(1, 13):
                n_m = len(self._monthly_features[m])
                k_m = min(self._n_neighbors, n_m - 1) if n_m > 1 else 1
                knn = NearestNeighbors(
                    n_neighbors=k_m, algorithm='auto', metric='euclidean'
                )
                knn.fit(self._monthly_features[m])
                self._monthly_knn[m] = knn
                self._monthly_weights[m] = self._make_kernel_weights(k_m)
            self.logger.info(
                "Fitting complete: 12 month-conditioned KNN models, K=%d",
                self._n_neighbors,
            )
        else:
            # Global KNN model
            self._knn_model = NearestNeighbors(
                n_neighbors=self._n_neighbors,
                algorithm='auto',
                metric='euclidean',
            )
            self._knn_model.fit(self._feature_vectors)
            self.logger.info(
                "Fitting complete: KNN model trained with %d neighbors",
                self._n_neighbors,
            )

        # Global weights (used by non-monthly path and for metadata)
        self._kernel_weights = self._make_kernel_weights(self._n_neighbors)

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Fitted parameters including kernel weights and sample size.
        """
        # Count fitted parameters: mainly the KNN distances and weights
        n_params = self._n_neighbors

        # Get training period from original data
        training_period = (
            str(self._Q_obs.index[0].date()),
            str(self._Q_obs.index[-1].date())
        )

        return FittedParams(
            means_=None,  # Nonparametric, no distributional assumptions
            stds_=None,
            correlations_=None,
            distributions_=None,
            fitted_models_={'knn_model': self._knn_model},
            n_parameters_=n_params,
            sample_size_=len(self._feature_vectors),
            n_sites_=self.n_sites,
            training_period_=training_period
        )

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> Ensemble:
        """
        Generate synthetic streamflow realizations.

        Uses KNN bootstrap with Lall-Sharma kernel weighting to conditionally
        resample from historical record.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Number of years to generate. If None, uses number of observed years.
        n_timesteps : int, optional
            Number of timesteps to generate explicitly. Overrides n_years if provided.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Ensemble
            Generated synthetic flows with metadata.
        """
        # Validate fit
        self.validate_fit()

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Determine number of timesteps to generate
        if n_timesteps is not None:
            n_generate = n_timesteps
        elif n_years is not None:
            # Estimate timesteps per year based on frequency
            if self._frequency == 'D':
                n_generate = n_years * 365
            elif self._frequency == 'YS':
                n_generate = n_years
            else:  # 'MS' or monthly
                n_generate = n_years * 12
        else:
            # Default: generate same length as observed
            n_generate = len(self._Q_obs)

        # Generate realizations
        realization_dict = {}
        for i in range(n_realizations):
            Q_syn = self._generate_single_realization(n_generate)
            realization_dict[i] = Q_syn

        # Create metadata
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            generator_params=self.get_params(),
            n_realizations=n_realizations,
            n_sites=self.n_sites,
            time_resolution=self._frequency,
            time_period=(
                str(realization_dict[0].index[0].date()),
                str(realization_dict[0].index[-1].date())
            )
        )

        # Create and return Ensemble
        ensemble = Ensemble(realization_dict, metadata=metadata)

        self.logger.info(
            f"Generated {n_realizations} realizations of {n_generate} timesteps each"
        )

        return ensemble

    def _generate_single_realization(self, n_timesteps: int) -> pd.DataFrame:
        """
        Generate a single synthetic realization.

        For monthly data, the neighbor search at each step is conditioned on
        the calendar month of the *current* timestep (Rajagopalan & Lall 1999).
        The successor of the selected neighbor provides the value for the
        *next* timestep, naturally advancing month-to-month.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps to generate.

        Returns
        -------
        pd.DataFrame
            Synthetic flow data with DatetimeIndex and site columns.
        """
        # Build date index first so we know each timestep's month
        if self._frequency == 'D':
            freq = 'D'
        elif self._frequency == 'YS':
            freq = 'YS'
        else:
            freq = 'MS'

        start_date = self._Q_obs.index[-1] + pd.DateOffset(months=1) \
            if freq == 'MS' else self._Q_obs.index[-1] + pd.Timedelta(days=1)
        date_index = pd.date_range(start=start_date, periods=n_timesteps, freq=freq)

        Q_syn = np.zeros((n_timesteps, len(self._sites)))
        n_features = len(self._feature_cols)

        if self._is_monthly_conditioned:
            # Month-conditioned generation
            # Initialize: draw from the first month's pool
            m0 = date_index[0].month
            init_idx = np.random.randint(0, len(self._monthly_successors[m0]))
            Q_syn[0, :] = self._monthly_successors[m0][init_idx, :]

            for t in range(1, n_timesteps):
                # Current month determines which KNN model to query
                m = date_index[t - 1].month
                current_feature = Q_syn[t - 1, :n_features].reshape(1, -1)

                knn = self._monthly_knn[m]
                weights = self._monthly_weights[m]

                _, neighbor_indices = knn.kneighbors(current_feature)
                neighbor_indices = neighbor_indices[0]

                selected_idx = np.random.choice(neighbor_indices, p=weights)
                Q_syn[t, :] = self._monthly_successors[m][selected_idx, :]
        else:
            # Global (non-monthly) generation — original algorithm
            init_idx = np.random.randint(0, len(self._feature_vectors))
            Q_syn[0, :] = self._successor_values[init_idx, :]

            for t in range(1, n_timesteps):
                current_feature = Q_syn[t - 1, :n_features].reshape(1, -1)

                _, neighbor_indices = self._knn_model.kneighbors(
                    current_feature, n_neighbors=self._n_neighbors
                )
                neighbor_indices = neighbor_indices[0]

                selected_idx = np.random.choice(
                    neighbor_indices, p=self._kernel_weights
                )
                Q_syn[t, :] = self._successor_values[selected_idx, :]

        return pd.DataFrame(Q_syn, index=date_index, columns=self._sites)
