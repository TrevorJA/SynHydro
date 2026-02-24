"""
SynHydro: Synthetic Generation Library

A library for generating synthetic hydrologic timeseries with focus on
statistical preservation and hydrologic applications.
"""

__version__ = "0.0.2"

# Core utilities
from synhydro.core import (
    Generator,
    Disaggregator,
    Ensemble,
    EnsembleMetadata,
    GeneratorDisaggregatorPipeline,
)
from synhydro.utils import load_example_data

# Drought analysis tools
from synhydro.droughts.ssi import SSIDroughtMetrics, SSI, get_drought_metrics
from synhydro.droughts.distributions import (
    list_distributions,
    get_distribution,
    DISTRIBUTION_REGISTRY,
)
from synhydro.droughts.diagnostics import (
    compare_distributions,
    distribution_summary,
)

# Generators
from synhydro.methods.generation.nonparametric.kirsch import KirschGenerator
from synhydro.methods.generation.nonparametric.phase_randomization import PhaseRandomizationGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.generation.parametric.matalas import MATALASGenerator
from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.methods.generation.parametric.warm import WARMGenerator

# Disaggregators
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator

# Pre-built pipelines (recommended for most users)
from synhydro.pipelines import (
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
