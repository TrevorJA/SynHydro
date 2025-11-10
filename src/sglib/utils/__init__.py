"""
Utility functions for SGLib.

This module provides utilities for data loading, validation, and
directory management.
"""

# Data loading
from sglib.utils.data import load_example_data, get_example_data_info

# Directories
from sglib.utils.directories import (
    PACKAGE_ROOT,
    EXAMPLE_DATA_DIR,
    get_example_data_path,
    list_example_datasets
)

# Validation (not exported by default, but available)
# from sglib.utils.validation import (
#     validate_timeseries,
#     validate_frequency,
#     validate_positive,
#     validate_columns
# )

__all__ = [
    # Data loading
    'get_example_data_info',
    'load_example_data',
]
