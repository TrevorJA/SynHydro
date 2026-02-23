"""
Wavelet Auto-Regressive Method (WARM) for streamflow generation.

Implements the WARM methodology from Nowak et al. (2011) for generating
synthetic streamflow sequences that preserve non-stationary spectral characteristics.
"""
import logging
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy import signal
from numpy.typing import NDArray

from sglib.core.base import Generator, FittedParams
from sglib.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


class WARMGenerator(Generator):
    """
    Wavelet Auto-Regressive Method (WARM) for non-stationary streamflow generation.

    Implements the 4-step WARM methodology:
    1. Wavelet transform decomposition into periodic components
    2. Scale Averaged Wavelet Power (SAWP) calculation for time-varying normalization
    3. AR model fitting to scaled wavelet coefficients
    4. Stochastic generation with inverse wavelet transform

    The SAWP approach enables preservation of non-stationary spectral characteristics
    and time-varying variability in synthetic sequences.

    Note: WARM is designed for annual streamflow generation (univariate).

    Examples
    --------
    >>> import pandas as pd
    >>> from sglib.methods.generation.parametric.warm import WARMGenerator
    >>> Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
    >>> warm = WARMGenerator(Q_annual.iloc[:, 0], wavelet='morlet', scales=64)
    >>> warm.preprocessing()
    >>> warm.fit()
    >>> ensemble = warm.generate(n_years=100, n_realizations=50, seed=42)

    References
    ----------
    Nowak, K., Rajagopalan, B., & Zagona, E. (2011). A Wavelet Auto-Regressive Method
    (WARM) for multi-site streamflow simulation of data with non-stationary trends.
    Journal of Hydrology, 410(1-2), 1-12.

    Kwon, H.-H., Lall, U., & Khalil, A. F. (2007). Stochastic simulation model for
    nonstationary time series using an autoregressive wavelet decomposition:
    Applications to rainfall and temperature. Water Resources Research, 43(5).
    """

    def __init__(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        wavelet: str = 'morl',
        scales: int = 64,
        ar_order: int = 1,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the WARM Generator.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Annual streamflow data with DatetimeIndex. Must be single site.
            If DataFrame provided, will use first column only.
        wavelet : str, default='morl'
            Wavelet type for continuous wavelet transform.
            Options: 'morl' (Morlet), 'mexh' (Mexican Hat), 'gaus1'-'gaus8'.
            Morlet wavelet recommended for hydrologic applications.
        scales : int, default=64
            Number of scales for wavelet decomposition. Higher values capture
            more frequency components but increase computational cost.
        ar_order : int, default=1
            Order of autoregressive model for each wavelet scale.
            Default AR(1) preserves temporal persistence.
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : dict, optional
            Additional parameters (currently unused).

        Raises
        ------
        ValueError
            If scales < 2 or ar_order < 1.
        """
        # Initialize base class
        super().__init__(Q_obs=Q_obs, name=name, debug=debug)

        # Validate parameters
        if scales < 2:
            raise ValueError(f"scales must be >= 2, got {scales}")
        if ar_order < 1:
            raise ValueError(f"ar_order must be >= 1, got {ar_order}")

        # Store WARM-specific parameters
        self.wavelet = wavelet
        self.scales = scales
        self.ar_order = ar_order

        # Validate wavelet type
        if wavelet not in pywt.wavelist(kind='continuous'):
            raise ValueError(
                f"wavelet '{wavelet}' not recognized. "
                f"Must be one of {pywt.wavelist(kind='continuous')}"
            )

        # Initialize storage for fitted components
        self.wavelet_coeffs_: Optional[NDArray] = None
        self.sawp_: Optional[NDArray] = None
        self.ar_params_: Optional[Dict[int, Dict[str, Any]]] = None
        self.scales_used_: Optional[NDArray] = None

        # Store initialization parameters
        self.init_params.algorithm_params = {
            'method': 'WARM (Wavelet Auto-Regressive Method)',
            'wavelet': wavelet,
            'scales': scales,
            'ar_order': ar_order,
            'reference': 'Nowak et al. (2011)'
        }

    @property
    def output_frequency(self) -> str:
        """WARM generator produces annual output."""
        return 'YS'  # Year Start

    def preprocessing(self, sites: Optional[list] = None, **kwargs) -> None:
        """
        Preprocess observed data for WARM generation.

        Validates input data and ensures annual frequency. WARM is designed
        for annual streamflow generation.

        Parameters
        ----------
        sites : list, optional
            Not used (WARM is univariate).
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        # Validate input data
        Q = self.validate_input_data(self._Q_obs_raw)

        # WARM is univariate - ensure single site
        if Q.shape[1] > 1:
            self.logger.warning("WARM is univariate. Using first column only.")
            Q = Q.iloc[:, 0:1]

        # Store sites
        self._sites = Q.columns.tolist()

        # Convert to Series
        Q_series = Q.iloc[:, 0]

        # Check/resample to annual frequency
        if Q_series.index.freq not in ['YS', 'Y', 'A', 'AS']:
            # If monthly data, resample to annual
            if Q_series.index.freq in ['MS', 'M']:
                self.logger.info("Resampling from monthly to annual (sum)")
                Q_series = Q_series.resample('YS').sum()
            # If daily data, resample to annual
            elif Q_series.index.freq in ['D']:
                self.logger.info("Resampling from daily to annual (sum)")
                Q_series = Q_series.resample('YS').sum()
            else:
                self.logger.warning(
                    f"Unknown frequency {Q_series.index.freq}. "
                    "Assuming annual data."
                )

        # Store annual data
        self.Q_obs_annual = Q_series

        # Validate sufficient data length
        if len(self.Q_obs_annual) < 20:
            self.logger.warning(
                f"Only {len(self.Q_obs_annual)} years of data. "
                "WARM may not perform well with < 20 years."
            )

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {len(self.Q_obs_annual)} annual values"
        )

    def fit(self, **kwargs) -> None:
        """
        Fit WARM model to observed annual flows.

        Implements the 4-step WARM methodology:
        1. Continuous wavelet transform
        2. Scale Averaged Wavelet Power (SAWP) calculation
        3. Normalization by SAWP
        4. AR model fitting to scaled coefficients

        Parameters
        ----------
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        # Validate preprocessing
        self.validate_preprocessing()

        # Extract observed flows as numpy array
        Q = self.Q_obs_annual.values
        n_years = len(Q)

        self.logger.info("Step 1/4: Computing continuous wavelet transform")
        # Step 1: Continuous Wavelet Transform
        scales_array = np.arange(1, self.scales + 1)
        coefficients, frequencies = pywt.cwt(
            Q,
            scales_array,
            self.wavelet,
            sampling_period=1.0
        )

        # Store coefficients and scales
        self.wavelet_coeffs_ = coefficients  # Shape: (n_scales, n_years)
        self.scales_used_ = scales_array

        self.logger.info("Step 2/4: Computing Scale Averaged Wavelet Power (SAWP)")
        # Step 2: Compute Scale Averaged Wavelet Power (SAWP)
        # SAWP captures time-varying power at each time point
        self.sawp_ = self._compute_sawp(coefficients)  # Shape: (n_years,)

        self.logger.info("Step 3/4: Normalizing coefficients by SAWP")
        # Step 3: Normalize coefficients by SAWP
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        sawp_matrix = self.sawp_[np.newaxis, :] + epsilon  # Shape: (1, n_years)
        normalized_coeffs = coefficients / np.sqrt(sawp_matrix)

        self.logger.info("Step 4/4: Fitting AR models to each scale")
        # Step 4: Fit AR(p) model to each scale's normalized coefficients
        self.ar_params_ = {}
        for scale_idx in range(self.scales):
            scale_data = normalized_coeffs[scale_idx, :]
            ar_params = self._fit_ar_model(scale_data, self.ar_order)
            self.ar_params_[scale_idx] = ar_params

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {self.scales} scales, AR({self.ar_order}) models fitted"
        )

    def _compute_sawp(self, coefficients: NDArray) -> NDArray:
        """
        Compute Scale Averaged Wavelet Power (SAWP).

        SAWP is the key innovation of WARM (Nowak et al. 2011) that enables
        capturing time-varying power spectral characteristics. It represents
        the average power across all scales at each time point.

        Parameters
        ----------
        coefficients : NDArray
            Wavelet coefficients with shape (n_scales, n_years).

        Returns
        -------
        NDArray
            SAWP time series with shape (n_years,).
        """
        # Compute power at each scale and time point
        power = np.abs(coefficients) ** 2

        # Average across all scales for each time point
        sawp = np.mean(power, axis=0)

        return sawp

    def _fit_ar_model(
        self,
        data: NDArray,
        order: int
    ) -> Dict[str, Any]:
        """
        Fit autoregressive model to time series data.

        Uses Yule-Walker equations for efficient AR parameter estimation.

        Parameters
        ----------
        data : NDArray
            Time series data to fit AR model.
        order : int
            Order of AR model.

        Returns
        -------
        dict
            Dictionary containing:
            - 'coeffs': AR coefficients (length p)
            - 'sigma': Innovation variance
            - 'mean': Mean of the series
        """
        # Remove mean
        mean = np.mean(data)
        data_centered = data - mean

        # Handle edge cases
        if len(data) <= order:
            self.logger.warning(
                f"Data length ({len(data)}) <= AR order ({order}). "
                "Using simpler model."
            )
            return {
                'coeffs': np.zeros(order),
                'sigma': np.std(data_centered) if len(data) > 1 else 1.0,
                'mean': mean
            }

        # Compute autocorrelation function
        acf = np.correlate(data_centered, data_centered, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]  # Normalize

        # Build Yule-Walker system: R * phi = r
        # where R is Toeplitz matrix of autocorrelations
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = acf[abs(i - j)]

        r = acf[1:order+1]

        # Solve for AR coefficients
        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular matrix in AR fitting. Using ridge regression.")
            # Add small regularization
            phi = np.linalg.solve(R + 1e-6 * np.eye(order), r)

        # Compute innovation variance
        # sigma^2 = gamma_0 * (1 - phi^T * r)
        variance_data = np.var(data_centered)
        innovation_var = variance_data * (1 - np.dot(phi, r))

        # Ensure positive variance
        innovation_var = max(innovation_var, 1e-10)

        return {
            'coeffs': phi,
            'sigma': np.sqrt(innovation_var),
            'mean': mean
        }

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters from WARM model.
        """
        # Count parameters
        # For each scale: p AR coefficients + 1 variance + 1 mean
        params_per_scale = self.ar_order + 2
        n_params = self.scales * params_per_scale + len(self.sawp_)

        # Get training period
        training_period = (
            str(self.Q_obs_annual.index[0].date()),
            str(self.Q_obs_annual.index[-1].date())
        )

        # Extract AR parameters for summary
        ar_coeffs_summary = {}
        ar_sigma_summary = {}
        for scale_idx, params in self.ar_params_.items():
            ar_coeffs_summary[f'scale_{scale_idx}'] = params['coeffs']
            ar_sigma_summary[f'scale_{scale_idx}'] = params['sigma']

        return FittedParams(
            means_=pd.Series({'annual_mean': self.Q_obs_annual.mean()}),
            stds_=pd.Series({'annual_std': self.Q_obs_annual.std()}),
            correlations_=None,  # Not directly applicable for WARM
            distributions_={
                'type': 'Wavelet-based with AR innovations',
                'wavelet': self.wavelet,
                'scales': self.scales,
                'ar_order': self.ar_order
            },
            transformations_={
                'wavelet_transform': {
                    'wavelet': self.wavelet,
                    'n_scales': self.scales,
                    'sawp_shape': self.sawp_.shape
                },
                'ar_coefficients': ar_coeffs_summary,
                'innovation_std': ar_sigma_summary
            },
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs_annual),
            n_sites_=1,  # WARM is univariate
            training_period_=training_period
        )

    def _generate(self, n_years: int, **kwargs) -> pd.DataFrame:
        """
        Generate a single realization of synthetic flows (internal method).

        Implements the WARM generation procedure:
        1. Generate synthetic AR innovations for each scale
        2. Multiply by SAWP (resampled with replacement)
        3. Inverse wavelet transform to reconstruct time series

        Parameters
        ----------
        n_years : int
            Number of years to generate.
        **kwargs : dict, optional
            Additional parameters (currently unused).

        Returns
        -------
        pd.DataFrame
            Single realization of synthetic annual flows.
        """
        # Generate normalized coefficients for each scale using AR models
        synthetic_normalized_coeffs = np.zeros((self.scales, n_years))

        for scale_idx in range(self.scales):
            ar_params = self.ar_params_[scale_idx]

            # Generate AR(p) time series
            coeffs = ar_params['coeffs']
            sigma = ar_params['sigma']
            mean = ar_params['mean']

            # Initialize with zeros (or could use historical values)
            series = np.zeros(n_years)
            innovations = np.random.normal(0, sigma, n_years)

            # Generate AR process
            for t in range(n_years):
                ar_component = 0.0
                for lag in range(1, self.ar_order + 1):
                    if t >= lag:
                        ar_component += coeffs[lag-1] * series[t - lag]

                series[t] = mean + ar_component + innovations[t]

            synthetic_normalized_coeffs[scale_idx, :] = series

        # Resample SAWP with replacement to get time-varying power for synthetic sequence
        # This preserves the range and distribution of power variations
        n_obs = len(self.sawp_)
        resampled_indices = np.random.choice(n_obs, size=n_years, replace=True)
        synthetic_sawp = self.sawp_[resampled_indices]

        # Rescale coefficients by synthetic SAWP
        epsilon = 1e-10
        sawp_matrix = synthetic_sawp[np.newaxis, :] + epsilon
        synthetic_coeffs = synthetic_normalized_coeffs * np.sqrt(sawp_matrix)

        # Inverse continuous wavelet transform
        # Note: pywt doesn't have direct inverse CWT, so we approximate by summing
        # weighted coefficients (this is a standard approach for CWT reconstruction)
        Q_syn = self._inverse_cwt(synthetic_coeffs, self.scales_used_)

        # Ensure non-negative flows
        Q_syn = np.maximum(Q_syn, 0)

        # Create DataFrame with annual dates
        start_year = self.Q_obs_annual.index[0].year
        dates = pd.date_range(start=f'{start_year}-01-01', periods=n_years, freq='YS')

        Q_syn_df = pd.DataFrame(
            Q_syn,
            index=dates,
            columns=[self._sites[0]]
        )

        return Q_syn_df

    def _inverse_cwt(
        self,
        coefficients: NDArray,
        scales: NDArray
    ) -> NDArray:
        """
        Approximate inverse continuous wavelet transform.

        Since pywt does not provide direct inverse CWT, we use the standard
        reconstruction formula: weighted sum of coefficients across scales.

        Parameters
        ----------
        coefficients : NDArray
            Wavelet coefficients with shape (n_scales, n_years).
        scales : NDArray
            Scales array used in CWT.

        Returns
        -------
        NDArray
            Reconstructed time series.
        """
        # Reconstruction formula: sum over scales with scale-dependent weighting
        # For Morlet wavelet, the reconstruction is approximately:
        # x(t) ≈ (dj * dt^0.5 / C_delta * psi_0(0)) * sum_j (W_j(t) / s_j^0.5)
        # where C_delta is wavelet-dependent constant

        # Simplified reconstruction (works well in practice)
        n_scales, n_time = coefficients.shape

        # Weight by inverse square root of scale (standard for CWT reconstruction)
        weights = 1.0 / np.sqrt(scales)
        weights = weights / np.sum(weights)  # Normalize

        # Sum weighted real parts across scales
        reconstructed = np.zeros(n_time)
        for i, weight in enumerate(weights):
            reconstructed += weight * np.real(coefficients[i, :])

        # Scale to match observed flow magnitude
        # (reconstruction scale factor depends on wavelet and scales)
        scale_factor = np.std(self.Q_obs_annual) / (np.std(reconstructed) + 1e-10)
        reconstructed = reconstructed * scale_factor

        # Adjust mean
        mean_obs = np.mean(self.Q_obs_annual)
        mean_syn = np.mean(reconstructed)
        reconstructed = reconstructed + (mean_obs - mean_syn)

        return reconstructed

    def generate(
        self,
        n_years: Optional[int] = None,
        n_realizations: int = 1,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Ensemble:
        """
        Generate synthetic annual streamflows using WARM.

        Parameters
        ----------
        n_years : int, optional
            Number of years to generate per realization.
            If None, uses the length of historic data.
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_timesteps : int, optional
            Number of annual timesteps to generate. If provided, overrides n_years.
            For WARM, n_timesteps = n_years (annual data).
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict, optional
            Additional parameters (currently unused).

        Returns
        -------
        Ensemble
            Ensemble object containing all realizations.

        Raises
        ------
        ValueError
            If neither n_years nor n_timesteps is provided.
        """
        # Validate fit
        self.validate_fit()

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Determine number of years
        if n_timesteps is not None:
            n_years = n_timesteps  # For annual data, timesteps = years
        elif n_years is None:
            n_years = len(self.Q_obs_annual)

        if n_years <= 0:
            raise ValueError(f"n_years must be positive, got {n_years}")

        # Generate realizations
        realizations = {}
        for i in range(n_realizations):
            Q_syn_df = self._generate(n_years)
            realizations[i] = Q_syn_df

        self.logger.info(
            f"Generated {n_realizations} realizations of {n_years} years each"
        )

        return Ensemble(realizations)
