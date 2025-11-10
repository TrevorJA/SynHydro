"""
Kirsch-Nowak generator for daily streamflow generation.

Combines the Kirsch monthly generation method with Nowak multisite disaggregation
to generate daily streamflows while preserving spatial correlations across sites.
"""
import logging
import pandas as pd
import warnings

from sglib.methods.generation.nonparametric.kirsch import KirschGenerator
from sglib.methods.disaggregation.temporal.nowak import NowakDisaggregator

logger = logging.getLogger(__name__)

class KirschNowakGenerator(KirschGenerator):
    """
    Combined Kirsch monthly generation with Nowak multisite disaggregation.

    .. deprecated:: 0.1.0
        `KirschNowakGenerator` is deprecated and will be removed in a future version.
        Use :class:`sglib.pipelines.KirschNowakPipeline` instead, which provides
        the same functionality with a more flexible and maintainable architecture.

        Migration example::

            # Old approach (deprecated)
            from sglib import KirschNowakGenerator
            generator = KirschNowakGenerator(Q_daily)
            generator.preprocessing()
            generator.fit()
            ensemble = generator.generate(n_realizations=10, n_years=50)

            # New approach (recommended)
            from sglib.pipelines import KirschNowakPipeline
            pipeline = KirschNowakPipeline(Q_daily)
            pipeline.preprocessing()
            pipeline.fit()
            ensemble = pipeline.generate(n_realizations=10, n_years=50)

    Generates synthetic daily streamflows by first generating monthly flows using
    the Kirsch method, then disaggregating all sites simultaneously using KNN-based
    Nowak disaggregation while preserving spatial correlations.

    Parameters
    ----------
    Q : pd.DataFrame
        Daily streamflow data with DatetimeIndex and sites as columns.
    generate_using_log_flow : bool, optional
        If True, generate in log-space. Default is True.
    matrix_repair_method : str, optional
        Method for repairing correlation matrices ('spectral'). Default is 'spectral'.
    n_neighbors : int, optional
        Number of nearest neighbors for KNN disaggregation. Default is 5.
    max_month_shift : int, optional
        Maximum days to shift around month center during disaggregation. Default is 7.
    debug : bool, optional
        If True, print debug messages. Default is False.

    Attributes
    ----------
    nowak_disaggregator : NowakDisaggregator
        Multisite disaggregator instance.

    Examples
    --------
    >>> import pandas as pd
    >>> from sglib.methods.generate.nonparametric.kirsch_nowak import KirschNowakGenerator
    >>> Q = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>> generator = KirschNowakGenerator(Q)
    >>> generator.preprocessing()
    >>> generator.fit()
    >>> daily_flows = generator.generate(n_realizations=10, n_years=5)
    """
    def __init__(self, Q: pd.DataFrame, **kwargs):
        """
        Initialize the KirschNowakGenerator.

        .. deprecated:: 0.1.0
            Use :class:`sglib.pipelines.KirschNowakPipeline` instead.

        Parameters
        ----------
        Q : pd.DataFrame
            Daily streamflow data with DatetimeIndex and sites as columns.
        **kwargs : dict, optional
            Additional parameters passed to KirschGenerator and NowakDisaggregator.

            Kirsch parameters:
            - generate_using_log_flow : bool, default=True
            - matrix_repair_method : str, default='spectral'
            - debug : bool, default=False

            Nowak parameters:
            - n_neighbors : int, default=5
            - max_month_shift : int, default=7
        """
        # Issue deprecation warning
        warnings.warn(
            "KirschNowakGenerator is deprecated and will be removed in a future version. "
            "Use sglib.pipelines.KirschNowakPipeline instead for a more flexible and "
            "maintainable approach. See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )

        # Extract nowak-specific parameters before calling parent
        nowak_params = {
            'n_neighbors': kwargs.pop('n_neighbors', 5),
            'max_month_shift': kwargs.pop('max_month_shift', 7)
        }

        # Call the parent KirschGenerator
        super().__init__(Q, **kwargs)

        # Update initialization parameters with Nowak-specific settings
        self.init_params.algorithm_params['nowak_n_neighbors'] = nowak_params['n_neighbors']
        self.init_params.algorithm_params['nowak_max_month_shift'] = nowak_params['max_month_shift']
        self.init_params.algorithm_params['method'] = 'Kirsch-Nowak (monthly gen + daily disaggregation)'

        # Create a single multisite NowakDisaggregator for all sites
        self.nowak_disaggregator = NowakDisaggregator(
            Qh_daily=Q,  # Pass the entire multisite DataFrame
            **nowak_params
        )
        
    def preprocessing(self):
        """
        Preprocess data for both Kirsch generation and Nowak disaggregation.

        Aggregates daily flows to monthly for Kirsch generation and prepares
        the NowakDisaggregator for multisite disaggregation.
        """
        # Preprocess the KirschGenerator
        super().preprocessing()
        
        # Preprocess the multisite NowakDisaggregator
        self.nowak_disaggregator.preprocessing()
    
    def fit(self):
        """
        Fit both Kirsch generator and Nowak disaggregator models.

        Estimates Kirsch parameters (monthly means, stds, correlations) and builds
        the KNN model for Nowak multisite disaggregation.
        """
        # Fit the KirschGenerator
        super().fit()
        
        # Fit the multisite NowakDisaggregator
        if self.params.get('debug', False):
            logger.info("Fitting multisite NowakDisaggregator")

        self.nowak_disaggregator.fit()

        # Override fitted_params_ to include Nowak info
        self.fitted_params_ = self._compute_fitted_params()

    def _compute_fitted_params(self):
        """
        Extract and package fitted parameters for Kirsch-Nowak.

        Extends parent Kirsch parameters with Nowak disaggregator info.

        Returns
        -------
        FittedParams
            Dataclass containing fitted parameters from both components.
        """
        # Get base Kirsch parameters
        fitted_params = super()._compute_fitted_params()

        # Add Nowak-specific fitted info
        if not hasattr(fitted_params, 'fitted_models_') or fitted_params.fitted_models_ is None:
            fitted_params.fitted_models_ = {}

        fitted_params.fitted_models_['nowak_disaggregator'] = {
            'type': 'NowakDisaggregator',
            'n_neighbors': self.init_params.algorithm_params.get('nowak_n_neighbors'),
            'fitted': True
        }

        return fitted_params

    def generate(self,
                 n_realizations: int = 1,
                 n_years: int = 1,
                 as_array: bool = False):
        """
        Generate synthetic daily streamflows using Kirsch-Nowak method.

        Generates monthly flows with Kirsch generator, then disaggregates all sites
        simultaneously using Nowak KNN disaggregation to preserve spatial correlations.

        Parameters
        ----------
        n_realizations : int, optional
            Number of synthetic realizations to generate. Default is 1.
        n_years : int, optional
            Number of years to generate per realization. Default is 1.
        as_array : bool, optional
            If True, return numpy array (not yet implemented). Default is False.

        Returns
        -------
        dict
            Dictionary mapping realization indices to DataFrames of daily flows.
            Each DataFrame has DatetimeIndex and sites as columns.

        Raises
        ------
        ValueError
            If monthly or daily data format is invalid or temporal ranges don't match.
        """
        
        if as_array:
            warnings.warn("as_array=True is not yet implemented. Returning DataFrame instead.")
            as_array = False
        
        ### Generate monthly flows using the KirschGenerator
        # Qse_monthly is dict with {real_id: DataFrame}
        # where each DataFrame has monthly datetime index and columns for each site
        Qse_monthly = super().generate(n_realizations=n_realizations, 
                                      n_years=n_years, 
                                      as_array=False)
        
        # Dict of daily flows, matching Qse_monthly format
        Qse_daily = {}
        
        # For each realization, disaggregate all sites simultaneously
        for real_id in Qse_monthly.keys():
            if self.params.get('debug', False):
                logger.info(f"Disaggregating realization {real_id + 1}/{n_realizations}")
            
            # Get the multisite monthly flows for this realization
            # This should be a pd.DataFrame with sites as columns
            Qs_monthly_multisite = Qse_monthly[real_id]
            
            # Verify the input format
            if not isinstance(Qs_monthly_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for realization {real_id}, got {type(Qs_monthly_multisite)}")
            
            if not all(col in self.site_names for col in Qs_monthly_multisite.columns):
                raise ValueError(f"Monthly data columns {Qs_monthly_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Disaggregate all sites simultaneously using multisite disaggregator
            # Output will be pd.DataFrame of daily flows with sites as columns
            Qs_daily_multisite = self.nowak_disaggregator.disaggregate_monthly_flows(
                Qs_monthly=Qs_monthly_multisite
            )
            
            # Verify the output format
            if not isinstance(Qs_daily_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame output from disaggregator, got {type(Qs_daily_multisite)}")
            
            if not all(col in self.site_names for col in Qs_daily_multisite.columns):
                raise ValueError(f"Daily data columns {Qs_daily_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Verify temporal consistency
            expected_start = Qs_monthly_multisite.index[0]
            expected_end = Qs_monthly_multisite.index[-1] + pd.offsets.MonthEnd(0)
            
            if (Qs_daily_multisite.index[0] != expected_start or 
                Qs_daily_multisite.index[-1] != expected_end):
                raise ValueError(f"Disaggregated daily flows temporal range does not match expected range.\n"
                               f"Expected: {expected_start} to {expected_end}\n"
                               f"Got: {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
            
            # Store in the results dictionary
            Qse_daily[real_id] = Qs_daily_multisite
            
            if self.params.get('debug', False):
                logger.info(f"Successfully disaggregated realization {real_id + 1}: "
                           f"Shape {Qs_daily_multisite.shape}, "
                           f"Date range {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
        
        return Qse_daily