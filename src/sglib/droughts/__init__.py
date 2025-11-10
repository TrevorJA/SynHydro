"""
Drought analysis tools for SGLib.

This module provides tools for drought identification and analysis based on
the Standardized Streamflow Index (SSI) methodology.
"""

from sglib.droughts.ssi import (
    SSI,
    SSIDroughtMetrics,
    get_drought_metrics,
)

from sglib.droughts.distributions import (
    DISTRIBUTION_REGISTRY,
    get_distribution,
    list_distributions,
    get_distribution_info,
    print_distribution_guide,
    validate_distribution,
)

from sglib.droughts.diagnostics import (
    compare_distributions,
    distribution_summary,
    kolmogorov_smirnov_test,
    anderson_darling_test,
    compute_aic_bic,
)

__all__ = [
    # SSI calculation
    'SSI',
    'SSIDroughtMetrics',
    'get_drought_metrics',
    # Distribution management
    'DISTRIBUTION_REGISTRY',
    'get_distribution',
    'list_distributions',
    'get_distribution_info',
    'print_distribution_guide',
    'validate_distribution',
    # Diagnostics
    'compare_distributions',
    'distribution_summary',
    'kolmogorov_smirnov_test',
    'anderson_darling_test',
    'compute_aic_bic',
]
