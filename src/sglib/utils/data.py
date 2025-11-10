"""
Example data loading utilities.

Provides functions to load example datasets shipped with SGLib.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from sglib.utils.directories import get_example_data_path, list_example_datasets

logger = logging.getLogger(__name__)


def load_example_data(
    dataset: str = 'usgs_daily_streamflow_cms',
    **kwargs
) -> pd.DataFrame:
    """
    Load an example dataset.

    Parameters
    ----------
    dataset : str, default='usgs_daily_streamflow_cms'
        Name of the dataset to load (without .csv extension).
        Use list_example_datasets() to see available datasets.
    **kwargs
        Additional keyword arguments passed to pd.read_csv().
        Common options: index_col, parse_dates, etc.

    Returns
    -------
    pd.DataFrame
        Example data with datetime index.

    Raises
    ------
    FileNotFoundError
        If the specified dataset doesn't exist.

    Examples
    --------
    Load the default USGS streamflow data:

    >>> from sglib.utils import load_example_data
    >>> Q = load_example_data()
    >>> print(Q.head())

    Load a specific dataset:

    >>> Q = load_example_data('usgs_daily_streamflow_cms')

    See also
    --------
    list_example_datasets : List all available example datasets
    """
    # Add .csv extension if not provided
    if not dataset.endswith('.csv'):
        filename = f'{dataset}.csv'
    else:
        filename = dataset

    # Get full path
    filepath = get_example_data_path(filename)

    logger.info(f"Loading example data from {filepath.name}")

    # Set default kwargs for typical use case
    default_kwargs = {
        'index_col': 0,
        'parse_dates': True
    }
    default_kwargs.update(kwargs)

    # Load data
    try:
        data = pd.read_csv(filepath, **default_kwargs)
        logger.debug(f"Loaded {len(data)} rows, {len(data.columns)} columns")

        # Filter for rows where all columns have non-NaN and positive values
        original_len = len(data)

        # First, check if we have complete data (all columns valid)
        valid_mask = data.notna().all(axis=1) & (data > 0).all(axis=1)

        if valid_mask.sum() > 0:
            # Find the longest continuous sequence of valid data
            valid_groups = (valid_mask != valid_mask.shift()).cumsum()
            valid_groups_filtered = valid_groups[valid_mask]

            # Find the longest continuous group
            group_sizes = valid_groups_filtered.value_counts()
            longest_group_id = group_sizes.idxmax()
            longest_group_mask = valid_groups_filtered == longest_group_id

            # Get the indices for the longest continuous valid period
            valid_indices = longest_group_mask[longest_group_mask].index
            data = data.loc[valid_indices]

            # Trim to complete months
            # Check if we have daily data by inferring frequency
            if len(data) > 1:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq and inferred_freq.startswith('D'):
                    # Daily data - trim to complete months
                    start_date = data.index[0]
                    end_date = data.index[-1]

                    # Move start to beginning of month if not already
                    if start_date.day != 1:
                        start_date = (start_date + pd.offsets.MonthBegin(1)).normalize()

                    # Move end to end of previous complete month
                    # Find the first day of the month containing end_date
                    end_month_start = pd.Timestamp(year=end_date.year, month=end_date.month, day=1)
                    # Get the last day of the previous month
                    end_date = (end_month_start - pd.Timedelta(days=1)).normalize()

                    # Filter to complete months
                    if start_date <= end_date:
                        data = data.loc[start_date:end_date]
                        logger.info(f"Trimmed to complete months: {start_date.date()} to {end_date.date()}")

            # Calculate record length
            removed_rows = original_len - len(data)
            removal_pct = (removed_rows / original_len) * 100

            # Infer frequency and calculate years of data
            if len(data) > 1:
                inferred_freq = pd.infer_freq(data.index)
                if inferred_freq and inferred_freq.startswith('D'):
                    years_of_data = len(data) / 365.25
                elif inferred_freq and inferred_freq in ['MS', 'M']:
                    years_of_data = len(data) / 12
                else:
                    years_of_data = (data.index[-1] - data.index[0]).days / 365.25
            else:
                years_of_data = 0

            # Issue warnings based on hydrologic best practices
            if removal_pct > 10:
                logger.warning(f"DATA MODIFICATION WARNING: Removed {removal_pct:.1f}% of rows ({removed_rows}/{original_len})")
                logger.warning(f"Significant data filtering may affect statistical properties of generated flows")

            if years_of_data < 10:
                logger.warning(f"INSUFFICIENT RECORD LENGTH WARNING: Only {years_of_data:.1f} years of data available")
                logger.warning(f"Hydrologic generators require at least 10 years for robust parameter estimation")
                logger.warning(f"Results may not adequately capture interannual variability and climate patterns")
            elif years_of_data < 20:
                logger.warning(f"LIMITED RECORD LENGTH: {years_of_data:.1f} years of data")
                logger.warning(f"20+ years recommended for capturing climate variability and extreme events")

            logger.info(f"Filtered to {len(data)} rows (removed {removed_rows} rows, {removal_pct:.1f}%)")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]} (~{years_of_data:.1f} years)")
        else:
            # No complete overlap - don't filter, just return as is
            logger.warning(f"DATA QUALITY WARNING: No rows with all columns valid")
            logger.warning(f"Data contains missing values or gaps that may affect generator performance")
            logger.warning(f"Consider regenerating example data or selecting specific columns")

        return data
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        raise


def get_example_data_info(dataset: str = 'usgs_daily_streamflow_cms') -> dict:
    """
    Get information about an example dataset without loading it.

    Parameters
    ----------
    dataset : str, default='usgs_daily_streamflow_cms'
        Name of the dataset.

    Returns
    -------
    dict
        Dictionary containing dataset information:
        - 'name': Dataset name
        - 'path': Full file path
        - 'size_mb': File size in megabytes
        - 'exists': Whether file exists

    Examples
    --------
    >>> from sglib.utils import get_example_data_info
    >>> info = get_example_data_info()
    >>> print(f"Dataset size: {info['size_mb']:.2f} MB")
    """
    if not dataset.endswith('.csv'):
        filename = f'{dataset}.csv'
    else:
        filename = dataset

    try:
        filepath = get_example_data_path(filename)
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        return {
            'name': dataset,
            'path': str(filepath),
            'size_mb': size_mb,
            'exists': True
        }
    except FileNotFoundError:
        return {
            'name': dataset,
            'path': None,
            'size_mb': None,
            'exists': False
        }


__all__ = [
    'load_example_data',
    'get_example_data_info',
]
