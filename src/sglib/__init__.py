"""
SGLib: Synthetic Generation Library

A library for generating synthetic hydrologic timeseries with focus on
statistical preservation and hydrologic applications.
"""

__version__ = "0.0.2"

# Core utilities
from sglib.core import (
    Generator,
    Disaggregator,
    Ensemble,
    EnsembleMetadata,
    GeneratorDisaggregatorPipeline,
)
from sglib.utils import load_example_data

# Drought analysis tools
from sglib.droughts.ssi import SSIDroughtMetrics, SSI, get_drought_metrics
from sglib.droughts.distributions import (
    list_distributions,
    get_distribution,
    DISTRIBUTION_REGISTRY,
)
from sglib.droughts.diagnostics import (
    compare_distributions,
    distribution_summary,
)

# Generators
from sglib.methods.generation.nonparametric.kirsch import KirschGenerator
from sglib.methods.generation.nonparametric.phase_randomization import PhaseRandomizationGenerator
from sglib.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from sglib.methods.generation.parametric.matalas import MATALASGenerator
from sglib.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from sglib.methods.generation.parametric.warm import WARMGenerator

# Disaggregators
from sglib.methods.disaggregation.temporal.nowak import NowakDisaggregator

# Pre-built pipelines (recommended for most users)
from sglib.pipelines import (
    KirschNowakPipeline,
    ThomasFieringNowakPipeline,
)



# Public API
__all__ = [
    # Base classes
    "Generator",
    "Disaggregator",
    # Individual generators
    "KirschGenerator",
    "PhaseRandomizationGenerator",
    "ThomasFieringGenerator",
    "MATALASGenerator",
    "MultiSiteHMMGenerator",
    "WARMGenerator",
    # Individual disaggregators
    "NowakDisaggregator",
    # Pipeline system
    "GeneratorDisaggregatorPipeline",
    "KirschNowakPipeline",
    "ThomasFieringNowakPipeline",
    # Ensemble management
    "Ensemble",
    "EnsembleMetadata",
    # Data utilities
    "load_example_data",
    # Drought analysis - SSI calculation
    "SSIDroughtMetrics",
    "SSI",
    "get_drought_metrics",
    # Drought analysis - distributions
    "list_distributions",
    "get_distribution",
    "DISTRIBUTION_REGISTRY",
    # Drought analysis - diagnostics
    "compare_distributions",
    "distribution_summary",
]
