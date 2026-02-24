"""
Package directory utilities.

Provides paths to package resources and example data.
"""
from pathlib import Path
from typing import Union


# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.resolve()

# Example data directory located in root /examples/example_data
EXAMPLE_DATA_DIR = PACKAGE_ROOT.parent.parent / "examples" / "example_data"


def get_example_data_path(filename: str) -> Path:
    """
    Get path to an example data file.

    Parameters
    ----------
    filename : str
        Name of example data file (e.g., 'usgs_daily_streamflow_cms.csv').

    Returns
    -------
    Path
        Full path to the example data file.

    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist.

    Examples
    --------
    >>> from synhydro.utils.directories import get_example_data_path
    >>> path = get_example_data_path('usgs_daily_streamflow_cms.csv')
    >>> print(path.exists())
    True
    """
    filepath = EXAMPLE_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Example data file not found: {filename}\n"
            f"Expected location: {filepath}\n"
            f"Available files: {list(EXAMPLE_DATA_DIR.glob('*')) if EXAMPLE_DATA_DIR.exists() else 'directory not found'}"
        )

    return filepath


def list_example_datasets() -> list:
    """
    List all available example datasets.

    Returns
    -------
    list
        List of available example dataset filenames.

    Examples
    --------
    >>> from synhydro.utils.directories import list_example_datasets
    >>> datasets = list_example_datasets()
    >>> print(datasets)
    ['usgs_daily_streamflow_cms.csv', ...]
    """
    if not EXAMPLE_DATA_DIR.exists():
        return []

    return sorted([f.name for f in EXAMPLE_DATA_DIR.glob("*.csv")])
