"""
Multi-Site Hidden Markov Model Generator (Gold et al. 2024/2025)

Implements a Gaussian Mixture Model HMM for generating synthetic multi-site
streamflow that preserves both temporal dependencies (via hidden states) and
spatial correlations (via multivariate emissions with full covariance matrices).

Based on the methodology from:
Gold, D.F, Reed, P.M. & Gupta, R.S. (In Revision). Exploring the Spatially
Compounding Multi-sectoral Drought Vulnerabilities in Colorado's West Slope
River Basins. Earth's Future

References
----------
Gold, D.F, Reed, P.M. & Gupta, R.S. (In Revision). Exploring the Spatially
Compounding Multi-sectoral Drought Vulnerabilities in Colorado's West Slope
River Basins. Earth's Future
"""

import logging
import warnings
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from hmmlearn import hmm

from sglib.core.base import Generator, FittedParams
from sglib.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


class MultiSiteHMMGenerator(Generator):
    """
    Multi-site Hidden Markov Model generator for synthetic streamflow.

    Generates synthetic streamflow using a Gaussian Mixture Model HMM that
    models temporal dependencies through hidden states and spatial correlations
    through multivariate Gaussian emissions with state-specific covariance matrices.

    The method is particularly suited for capturing drought dynamics across
    multiple sites/basins simultaneously.

    Parameters
    ----------
    Q_obs : pd.Series or pd.DataFrame
        Observed streamflow data with DatetimeIndex. Must be DataFrame for
        multi-site generation (multiple columns = multiple sites).
    n_states : int, default=2
        Number of hidden states. Default is 2 (dry/wet states).
    offset : float, default=1.0
        Small value added before log transformation to handle zeros.
        Recommended: 1.0 for flows in standard units.
    max_iterations : int, default=1000
        Maximum iterations for HMM fitting convergence.
    covariance_type : str, default='full'
        Type of covariance matrix:
        - 'full': Full covariance matrix (captures all correlations)
        - 'diag': Diagonal covariance (independent sites)
        - 'spherical': Single variance for all dimensions
    name : str, optional
        Name identifier for this generator instance.
    debug : bool, default=False
        Enable debug logging.

    Attributes
    ----------
    means_ : np.ndarray
        State means for each site. Shape: (n_states, n_sites).
    covariances_ : np.ndarray
        Covariance matrices for each state. Shape: (n_states, n_sites, n_sites).
    transition_matrix_ : np.ndarray
        State transition probability matrix. Shape: (n_states, n_states).
    stationary_distribution_ : np.ndarray
        Stationary distribution of states. Shape: (n_states,).
    Q_log_ : np.ndarray
        Log-transformed observed flows used for fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from sglib.methods.generation.parametric import MultiSiteHMMGenerator
    >>>
    >>> # Load multi-site annual flows
    >>> Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Initialize generator
    >>> gen = MultiSiteHMMGenerator(Q_annual, n_states=2)
    >>> gen.preprocessing()
    >>> gen.fit()
    >>>
    >>> # Generate 100 realizations of 50 years each
    >>> ensemble = gen.generate(n_realizations=100, n_years=50, seed=42)

    Notes
    -----
    - Designed for annual timestep data (can handle other frequencies)
    - Log transformation ensures positive emissions
    - Full covariance preserves spatial correlations between sites
    - State ordering: states sorted by mean (low mean = dry state)
    """

    def __init__(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        n_states: int = 2,
        offset: float = 1.0,
        max_iterations: int = 1000,
        covariance_type: str = 'full',
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize the MultiSiteHMMGenerator."""
        super().__init__(Q_obs=Q_obs, name=name, debug=debug)

        # Validate parameters
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")

        if offset <= 0:
            raise ValueError(f"offset must be positive, got {offset}")

        if covariance_type not in ('full', 'diag', 'spherical'):
            raise ValueError(
                f"covariance_type must be 'full', 'diag', or 'spherical', "
                f"got '{covariance_type}'"
            )

        self.n_states = n_states
        self.offset = offset
        self.max_iterations = max_iterations
        self.covariance_type = covariance_type

        # Store initialization parameters
        self.init_params.algorithm_params = {
            'method': 'Multi-Site Hidden Markov Model (Gold et al. 2025)',
            'n_states': n_states,
            'offset': offset,
            'max_iterations': max_iterations,
            'covariance_type': covariance_type
        }

        # Initialize fitted parameter storage
        self.means_ = None
        self.covariances_ = None
        self.transition_matrix_ = None
        self.stationary_distribution_ = None
        self.Q_log_ = None
        self._hmm_model = None

    @property
    def output_frequency(self) -> str:
        """
        Output frequency matches input frequency.

        Typically used for annual data ('YS' or 'AS'), but flexible.
        """
        if hasattr(self, '_Q_obs') and self._Q_obs is not None:
            # Infer from preprocessed data
            return pd.infer_freq(self._Q_obs.index) or 'YS'
        return 'YS'  # Default to annual start

    def preprocessing(self, sites: Optional[List[str]] = None, **kwargs) -> None:
        """
        Preprocess observed data for HMM fitting.

        Applies offset and log transformation to handle zeros and ensure
        positive values for fitting.

        Parameters
        ----------
        sites : List[str], optional
            Subset of sites to use. If None, uses all columns.
        **kwargs : dict
            Additional preprocessing parameters (currently unused).

        Raises
        ------
        ValueError
            If data has fewer than 2 sites for multi-site modeling.
        """
        # Validate input data
        Q = self.validate_input_data(self._Q_obs_raw)

        # Handle site selection
        if sites is not None:
            missing_sites = set(sites) - set(Q.columns)
            if missing_sites:
                raise ValueError(f"Sites not found in data: {missing_sites}")
            Q = Q[sites]

        self._sites = Q.columns.tolist()

        # Validate minimum sites for multi-site HMM
        if len(self._sites) < 2:
            self.logger.warning(
                "Multi-site HMM with only 1 site. Consider using univariate HMM."
            )

        # Store original observed data
        self._Q_obs = Q.copy()

        # Apply offset and log transformation
        Q_adj = Q + self.offset
        self.Q_log_ = np.log(Q_adj).values

        # Check for invalid values
        if not np.all(np.isfinite(self.Q_log_)):
            raise ValueError(
                "Log-transformed data contains non-finite values. "
                "Check for negative flows or adjust offset parameter."
            )

        self.logger.info(
            f"Preprocessing complete: {len(Q)} observations, "
            f"{len(self._sites)} sites, offset={self.offset}"
        )

        # Update state
        self.update_state(preprocessed=True)

    def fit(self, random_state: Optional[int] = None, **kwargs) -> None:
        """
        Fit the multi-site HMM to observed data.

        Estimates hidden states, transition probabilities, state-specific means,
        and covariance matrices using the GMMHMM algorithm.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducible fitting. If None, fitting may vary.
        **kwargs : dict
            Additional fitting parameters (currently unused).

        Notes
        -----
        States are automatically ordered by mean (ascending), so state 0
        represents the dry state and higher-numbered states represent
        progressively wetter states.
        """
        self.validate_preprocessing()

        self.logger.debug(
            f"Fitting GMMHMM with {self.n_states} states, "
            f"covariance_type='{self.covariance_type}'"
        )

        # Initialize and fit GMMHMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress hmmlearn convergence warnings

            self._hmm_model = hmm.GMMHMM(
                n_components=self.n_states,
                n_iter=self.max_iterations,
                covariance_type=self.covariance_type,
                random_state=random_state
            )

            self._hmm_model.fit(self.Q_log_)

        # Extract parameters
        means = np.array(self._hmm_model.means_)  # Shape: (n_states, 1, n_sites)
        transition_matrix = np.array(self._hmm_model.transmat_)
        covariances_raw = self._hmm_model.covars_

        # Reshape means (remove middle dimension from GMMHMM output)
        means = means.squeeze(axis=1)  # Shape: (n_states, n_sites)

        # Reshape covariances
        # GMMHMM returns covars with shape (n_states, n_mix, ...) where n_mix=1
        # Need to squeeze the n_mix dimension and handle different covariance types
        n_sites = len(self._sites)

        if self.covariance_type == 'full':
            # Shape: (n_states, 1, n_sites, n_sites) -> (n_states, n_sites, n_sites)
            covariances = covariances_raw.squeeze(axis=1)
        elif self.covariance_type == 'diag':
            # Shape: (n_states, 1, n_sites) -> (n_states, n_sites, n_sites)
            diag_covs = covariances_raw.squeeze(axis=1)  # (n_states, n_sites)
            covariances = np.array([np.diag(cov) for cov in diag_covs])
        else:  # spherical
            # Shape: (n_states, 1) -> (n_states, n_sites, n_sites)
            spherical_covs = covariances_raw.squeeze(axis=1)  # (n_states,)
            covariances = np.array([cov * np.eye(n_sites) for cov in spherical_covs])

        # Order states by mean of first site (dry to wet)
        mean_order = np.argsort(means[:, 0])

        self.means_ = means[mean_order]
        self.covariances_ = covariances[mean_order]
        self.transition_matrix_ = transition_matrix[mean_order, :][:, mean_order]

        # Compute stationary distribution
        self.stationary_distribution_ = self._compute_stationary_distribution()

        self.logger.info(
            f"Fitting complete: {self.n_states} states, "
            f"transition matrix:\n{self.transition_matrix_}"
        )

        # Compute fitted params
        self.fitted_params_ = self._compute_fitted_params()

        # Update state
        self.update_state(fitted=True)

    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution from transition matrix.

        Returns
        -------
        np.ndarray
            Stationary probabilities for each state.
        """
        # Find eigenvector corresponding to eigenvalue = 1
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix_.T)

        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvals - 1.0))

        # Extract and normalize eigenvector
        pi = np.real(eigenvecs[:, idx])
        pi = pi / pi.sum()

        return pi

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Ensemble:
        """
        Generate synthetic streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Number of years to generate. If provided with annual data,
            this equals n_timesteps.
        n_timesteps : int, optional
            Number of timesteps to generate explicitly. Takes precedence
            over n_years if both provided.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional generation parameters (currently unused).

        Returns
        -------
        Ensemble
            Generated synthetic flows as an Ensemble object.

        Raises
        ------
        ValueError
            If neither n_years nor n_timesteps is provided.
        """
        self.validate_fit()

        # Determine number of timesteps
        if n_timesteps is not None:
            n_steps = n_timesteps
        elif n_years is not None:
            n_steps = n_years
        else:
            raise ValueError("Must provide either n_years or n_timesteps")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        self.logger.debug(
            f"Generating {n_realizations} realizations of {n_steps} timesteps"
        )

        realizations = {}

        for r in range(n_realizations):
            # Generate state trajectory
            states = self._generate_state_trajectory(n_steps)

            # Generate emissions for each timestep
            Q_log_syn = np.zeros((n_steps, len(self._sites)))
            for t, state in enumerate(states):
                Q_log_syn[t, :] = np.random.multivariate_normal(
                    self.means_[state],
                    self.covariances_[state]
                )

            # Back-transform from log space
            Q_syn = np.exp(Q_log_syn) - self.offset

            # Ensure non-negative flows
            Q_syn = np.maximum(Q_syn, 0.0)

            # Create DataFrame with appropriate index
            start_date = self._Q_obs.index[0]
            dates = pd.date_range(
                start=start_date,
                periods=n_steps,
                freq=self.output_frequency
            )

            realizations[r] = pd.DataFrame(
                Q_syn,
                index=dates,
                columns=self._sites
            )

        self.logger.info(f"Generated {n_realizations} realizations")

        return Ensemble(realizations)

    def _generate_state_trajectory(self, n_timesteps: int) -> List[int]:
        """
        Generate hidden state trajectory.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps in trajectory.

        Returns
        -------
        List[int]
            Sequence of hidden states.
        """
        # Sample initial state from stationary distribution
        state = np.random.choice(
            self.n_states,
            p=self.stationary_distribution_
        )

        states = [state]

        # Generate remaining states using transition matrix
        for _ in range(1, n_timesteps):
            state = np.random.choice(
                self.n_states,
                p=self.transition_matrix_[state, :]
            )
            states.append(state)

        return states

    def _compute_fitted_params(self) -> FittedParams:
        """Extract and package fitted parameters."""
        # Count parameters
        n_params = (
            self.n_states * len(self._sites) +  # Means
            self.n_states * len(self._sites) * (len(self._sites) + 1) // 2 +  # Covariances (triangular)
            self.n_states * (self.n_states - 1)  # Transition matrix (non-diagonal)
        )

        training_period = (
            str(self._Q_obs.index[0].date()),
            str(self._Q_obs.index[-1].date())
        )

        return FittedParams(
            means_=self.means_,
            correlations_={
                'covariance_matrices': self.covariances_,
                'transition_matrix': self.transition_matrix_,
                'stationary_distribution': self.stationary_distribution_
            },
            distributions_={
                'type': 'Multivariate Gaussian per state',
                'n_states': self.n_states,
                'covariance_type': self.covariance_type
            },
            transformations_={
                'log_transform': True,
                'offset': self.offset
            },
            n_parameters_=n_params,
            sample_size_=len(self._Q_obs),
            n_sites_=len(self._sites),
            training_period_=training_period
        )
