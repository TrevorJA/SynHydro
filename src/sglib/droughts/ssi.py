import logging
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Grouper, Series, Timedelta, Timestamp
import scipy.stats as scs
from scipy.stats import norm
import spei as si
from spei import SI  # Import the original SI class
from spei._typing import ContinuousDist
from spei.utils import get_data_series, group_yearly_df, validate_series

# Import distribution utilities from dedicated module
from sglib.droughts.distributions import get_distribution, validate_distribution


def get_drought_metrics(ssi):
    """
    Extract drought metrics from an SSI time series.

    A drought is defined as any period where SSI drops to -1 or below.
    The drought period extends from when SSI first goes below 0 until
    SSI returns to positive values.

    Parameters
    ----------
    ssi : pd.Series
        Time series of Standardized Streamflow Index values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: start, end, duration, magnitude, severity, max_severity_date
        Returns empty DataFrame if no droughts are detected.
    """
    drought_data = {}
    drought_counter = 0
    in_critical_drought = False
    drought_days = []

    for ind in range(len(ssi)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)

            if ssi.values[ind] <= -1:
                in_critical_drought = True
        else:
            # Record drought info once it ends
            if in_critical_drought:

                # Get date with max severity
                max_severity_date = ssi.index[drought_days][ssi.values[drought_days].argmin()]


                drought_counter += 1
                drought_data[drought_counter] = {
                    'start': pd.Timestamp(ssi.index[drought_days[0]]),
                    'end': pd.Timestamp(ssi.index[drought_days[-1]]),
                    'duration': len(drought_days),
                    'magnitude': sum(ssi.values[drought_days]),
                    'severity': min(ssi.values[drought_days]),
                    'max_severity_date': pd.Timestamp(max_severity_date),
                }

            in_critical_drought = False
            drought_days = []

    # Handle case where series ends in a drought
    if in_critical_drought and len(drought_days) > 0:
        max_severity_date = ssi.index[drought_days][ssi.values[drought_days].argmin()]
        drought_counter += 1
        drought_data[drought_counter] = {
            'start': pd.Timestamp(ssi.index[drought_days[0]]),
            'end': pd.Timestamp(ssi.index[drought_days[-1]]),
            'duration': len(drought_days),
            'magnitude': sum(ssi.values[drought_days]),
            'severity': min(ssi.values[drought_days]),
            'max_severity_date': pd.Timestamp(max_severity_date),
        }

    drought_metrics = pd.DataFrame(drought_data).transpose()

    # Convert columns to proper dtypes
    if len(drought_metrics) > 0:
        # Convert datetime columns
        for col in ['start', 'end', 'max_severity_date']:
            if col in drought_metrics.columns:
                drought_metrics[col] = pd.to_datetime(drought_metrics[col])

        # Convert numeric columns
        for col in ['duration', 'magnitude', 'severity']:
            if col in drought_metrics.columns:
                drought_metrics[col] = pd.to_numeric(drought_metrics[col])

    return drought_metrics

class SSIDroughtMetrics:
    """
    Class to calculate and store drought metrics based on the Standardized Streamflow Index (SSI).

    Attributes:
        timescale (str): Temporal scale ('D' for daily, 'M' for monthly).
        window (int): Rolling window size for aggregation.
        dist (ContinuousDist): Probability distribution for SSI calculation (default: gamma).
        ssi (pd.Series): Series of SSI values.
        drought_metrics (pd.DataFrame): DataFrame containing drought metrics.
    """

    def __init__(self,
                 timescale: str = 'M',
                 window: int = 12,
                 dist: Union[str, ContinuousDist] = 'gamma',
                 data = None,):
        """
        Initialize the SSIDroughtMetrics class.

        Parameters
        ----------
        timescale : str, default 'M'
            Temporal scale: 'D' for daily, 'M' for monthly.
        window : int, default 12
            Rolling window size for aggregation before SSI calculation.
        dist : str or ContinuousDist, default 'gamma'
            Probability distribution to use for SSI calculation.
            Can be string name from registry or scipy distribution object.
            Common string options: 'gamma', 'lognorm', 'pearson3', 'weibull'
            Or pass scipy object directly: scs.gamma, scs.lognorm, etc.
        data : pd.Series or pd.DataFrame, optional
            Initial data to set.

        Examples
        --------
        >>> # Using string name (recommended)
        >>> ssi = SSIDroughtMetrics(dist='gamma')
        >>> # Using scipy object directly
        >>> import scipy.stats as scs
        >>> ssi = SSIDroughtMetrics(dist=scs.lognorm)
        """
        assert isinstance(timescale, str), "Timescale must be a string."
        assert isinstance(window, int), "Window must be an integer."
        assert (timescale in ['D', 'M']), "Timescale must be either 'D' (daily) or 'M' (monthly)."

        self.timescale = timescale
        self.window = window
        self.dist = get_distribution(dist)  # Convert string to distribution object

        if data is not None:
            self._set_data(data)

        
        
    def _set_data(self, data):
        """
        Set the data for the class.

        Parameters:
            data (pd.DataFrame or pd.Series): Data to be set.
        """
        # data can be a array, series or dataframe
        # either way, convert to Series
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
            
            # check if the first column is datetime
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.iloc[:, 0]
            else:
                # set datetime index to the first column of the data
                data.index = pd.date_range(start='1945-01-01', 
                                           periods=data.shape[0], freq='D')
            
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            # set datetime index to the first column of the data
            freq = 'D' if self.timescale == 'D' else 'MS'
            data.index = pd.date_range(start='2000-01-01', 
                                       periods=data.shape[0], freq=freq)
        
        elif not isinstance(data, pd.Series):
            raise TypeError("Data must be a pandas DataFrame, Series, or numpy array.")
        
        self.data = data.copy()

    def calculate_ssi(self, data=None):
        """
        Calculate the Standardized Streamflow Index (SSI).

        Parameters
        ----------
        data : pd.Series or pd.DataFrame, optional
            Data to calculate SSI for. If None, uses previously set data.

        Returns
        -------
        pd.Series
            SSI values.

        Raises
        ------
        ValueError
            If no data is provided and no data was previously set.
        """
        if data is not None:
            self._set_data(data)

        elif not hasattr(self, 'data'):
            raise ValueError("Data not set. Please set data before calculating SSI.")

        # Get rolling sum
        data_rs = self.data.rolling(self.window,
                                    min_periods=self.window).sum().dropna()

        # Calculate the Standardized Streamflow Index (SSI) using configured distribution
        ssi = si.ssfi(series=data_rs,
                     dist=self.dist)

        return ssi
    
    def calculate_drought_metrics(self, ssi=None):
            
            if ssi is None:
                ssi = self.calculate_ssi()
                
            # Get drought metrics
            drought_metrics = get_drought_metrics(ssi)
            
            return drought_metrics
    

@dataclass
class SSI:
    """
    Independent SSI calculator that separates training and scoring phases.

    Uses the original spei.SI class internally for distribution fitting,
    then applies those fitted distributions to new data.

    Parameters
    ----------
    dist : str or ContinuousDist, default 'gamma'
        Probability distribution for SSI calculation.
        Can be string name: 'gamma', 'lognorm', 'pearson3', 'weibull', etc.
        Or scipy distribution object.
    timescale : int, default 12
        Rolling window size for temporal aggregation.
    fit_freq : str, optional
        Frequency for seasonal fitting ('M' for monthly, 'D' for daily).
        If None, fits single distribution to entire dataset.
    fit_window : int, default 0
        Moving window for distribution fitting.
    prob_zero : bool, default False
        Whether to handle zero probability separately.
    normal_scores_transform : bool, default False
        Whether to use normal scores transform instead of parametric fitting.
    agg_func : {'sum', 'mean'}, default 'sum'
        Aggregation function for rolling window.

    Examples
    --------
    >>> # Basic usage with default gamma distribution
    >>> ssi = SSI()
    >>> ssi.fit(training_data)
    >>> ssi_values = ssi.transform(new_data)

    >>> # Using different distribution
    >>> ssi = SSI(dist='lognorm', timescale=6)
    >>> ssi.fit(training_data)
    """

    # Training parameters
    dist: Union[str, ContinuousDist] = 'gamma'
    timescale: int = 12
    fit_freq: str | None = field(default=None)
    fit_window: int = field(default=0)
    prob_zero: bool = field(default=False)
    normal_scores_transform: bool = field(default=False)
    agg_func: Literal["sum", "mean"] = "sum"

    # Internal state
    _fitted_si: SI = field(init=False, repr=False, compare=False)
    _is_fitted: bool = field(default=False, init=False, repr=False, compare=False)
    _training_series: Series = field(init=False, repr=False, compare=False)
    _dist_obj: ContinuousDist = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Convert dist string to distribution object after initialization."""
        self._dist_obj = get_distribution(self.dist)
    
    
    def fit(self, training_series: Series) -> 'SSI':
        """
        Fit distributions using training data.

        Parameters
        ----------
        training_series : Series
            Time series data for fitting distributions

        Returns
        -------
        SSI
            Self for method chaining
        """
        self._training_series = training_series.copy()

        # Apply rolling aggregation if timescale is set
        if self.timescale > 0:
            training_series = (
                training_series.rolling(self.timescale,
                                        min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )


        # Create and fit the original SI object
        self._fitted_si = SI(
            series=training_series,
            dist=self._dist_obj,
            timescale=0,
            fit_freq=self.fit_freq,
            fit_window=self.fit_window,
            prob_zero=self.prob_zero,
            normal_scores_transform=self.normal_scores_transform,
            agg_func=self.agg_func,
        )

        self._fitted_si.fit_distribution()
        self._is_fitted = True

        return self
    
    def transform(self, new_series: Series) -> Series:
        """
        Calculate SSI values for new data using fitted distributions.
        
        Parameters
        ----------
        new_series : Series
            New time series data to transform
            
        Returns
        -------
        Series
            SSI values for the new series
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        # Preprocess new series the same way as training data
        new_series = validate_series(new_series)
        
        if self.timescale > 0:
            new_series = (
                new_series.rolling(self.timescale, 
                                   min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )
        
        # Calculate CDF for new data using fitted distributions
        cdf = self._calculate_cdf_for_new_data(new_series)

        # Clip CDF values to avoid -inf/+inf from norm.ppf
        # Use small epsilon to prevent exactly 0 or 1
        epsilon = 1e-10
        cdf_clipped = np.clip(cdf.values, epsilon, 1 - epsilon)

        # Convert to standard normal (SSI values)
        ssi = Series(
            norm.ppf(cdf_clipped, loc=0, scale=1),
            index=new_series.index,
            dtype=float
        )

        return ssi
    
    def fit_transform(self, 
                      training_series: Series, 
                      new_series: Series = None) -> Series:
        """
        Fit on training data and transform new data in one step.
        
        Parameters
        ----------
        training_series : Series
            Data for fitting distributions
        new_series : Series, optional
            Data to transform. If None, transforms training_series.
            
        Returns
        -------
        Series
            SSI values
        """
        self.fit(training_series)
        
        if new_series is not None:
            return self.transform(new_series)
        else:
            # Transform training data using fitted distributions
            return self._fitted_si.norm_ppf()
    
    def _calculate_cdf_for_new_data(self, new_series: Series) -> Series:
        """
        Calculate CDF for new data using the fitted distributions.
        
        Parameters
        ----------
        new_series : Series
            New time series data
            
        Returns
        -------
        Series
            CDF values for new data
        """
        if self.normal_scores_transform:
            return self._calculate_cdf_nsf(new_series)
        else:
            return self._calculate_cdf_parametric(new_series)
    
    def _calculate_cdf_nsf(self, new_series: Series) -> Series:
        """Calculate CDF using Normal Scores Transform approach."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)
        
        for date, grval in new_grouped.groupby(Grouper(freq=str(self._fitted_si.fit_freq))):
            new_data = get_data_series(grval)
            
            # Find corresponding training data for this period
            if date in self._fitted_si._dist_dict:
                training_data = self._fitted_si._dist_dict[date].data.sort_values()
                n_train = len(training_data)
                
                # For each new data point, find its rank in training data
                for idx, val in new_data.items():
                    # Count how many training values are less than this new value
                    rank = (training_data < val).sum()
                    # Use Weibull plotting position
                    cdf.loc[idx] = (rank + 0.5) / n_train
        
        return cdf
    
    def _calculate_cdf_parametric(self, new_series: Series) -> Series:
        """Calculate CDF using fitted parametric distributions."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)
        
        for date, grval in new_grouped.groupby(Grouper(freq=str(self._fitted_si.fit_freq))):
            new_data = get_data_series(grval)
            
            if date in self._fitted_si._dist_dict:
                fitted_dist = self._fitted_si._dist_dict[date]
                
                # Construct parameters tuple: (shape_params..., loc, scale)
                if fitted_dist.pars is not None:
                    params = (*fitted_dist.pars, fitted_dist.loc, fitted_dist.scale)
                else:
                    params = (fitted_dist.loc, fitted_dist.scale)
                
                # Handle zero probability case
                if self.prob_zero and fitted_dist.p0 > 0:
                    zero_prob = fitted_dist.p0
                    
                    for idx, val in new_data.items():
                        if val == 0:
                            cdf.loc[idx] = zero_prob
                        else:
                            # Get parametric CDF for non-zero values
                            dist_cdf = fitted_dist.dist.cdf(val, *params)
                            cdf.loc[idx] = zero_prob + (1 - zero_prob) * dist_cdf
                else:
                    # No zero handling needed
                    for idx, val in new_data.items():
                        cdf.loc[idx] = fitted_dist.dist.cdf(val, *params)
        
        return cdf
    
    def get_training_ssi(self) -> Series:
        """
        Get SSI values for the training data.
        
        Returns
        -------
        Series
            SSI values for training data
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before get_training_ssi()")
        
        return self._fitted_si.norm_ppf()
    
    @property
    def fitted_distributions(self) -> dict:
        """Access to fitted distributions (read-only)."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before accessing fitted_distributions")
        return self._fitted_si._dist_dict.copy()

