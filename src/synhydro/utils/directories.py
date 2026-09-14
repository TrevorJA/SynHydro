"""
Package directory utilities.

Provides paths to package resources and to the example datasets shipped in
the ``synhydro.data`` package.
"""

from importlib.resources import files as resource_files
from pathlib import Path


# Installed ``synhydro`` package directory
PACKAGE_ROOT = Path(__file__).parent.parent.resolve()


def get_example_data_dir() -> Path:
    """
    Get the directory holding the example datasets.

    The datasets are package data in ``synhydro.data`` and are located with
    ``importlib.resources`` so the lookup works for editable installs,
    regular ``pip`` installs, and built wheels alike.

    Returns
    -------
    Path
        Directory containing the example CSV files.
    """
    # synhydro.data is a regular package installed as a directory, so the
    # Traversable returned by importlib.resources is a concrete filesystem
    # path.
    return Path(resource_files("synhydro.data"))


# Example data directory. Kept as a module constant because it is exported
# from the synhydro.utils namespace.
EXAMPLE_DATA_DIR = get_example_data_dir()


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

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Example data file not found: {filename}\n"
            f"Expected location: {filepath}\n"
            f"Available files: {list_example_datasets()}"
        )

    return filepath


def list_example_datasets() -> list[str]:
    """
    List all available example datasets.

    Returns
    -------
    list of str
        Sorted list of example dataset filenames.

    Examples
    --------
    >>> from synhydro.utils.directories import list_example_datasets
    >>> list_example_datasets()
    ['usgs_daily_streamflow_cms.csv', 'usgs_monthly_streamflow_cms.csv']
    """
    if not EXAMPLE_DATA_DIR.is_dir():
        return []

    return sorted(f.name for f in EXAMPLE_DATA_DIR.glob("*.csv"))
