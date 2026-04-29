"""
Generation methods for SynHydro.
"""

from synhydro.methods.generation.nonparametric import (
    KirschGenerator,
    KNNBootstrapGenerator,
    MultisitePhaseRandomizationGenerator,
    PhaseRandomizationGenerator,
)
from synhydro.methods.generation.parametric import (
    ARFIMAGenerator,
    HMMKNNGenerator,
    MatalasGenerator,
    MultiSiteHMMGenerator,
    SMARTAGenerator,
    SPARTAGenerator,
    ThomasFieringGenerator,
    WARMGenerator,
)

__all__ = [
    "ARFIMAGenerator",
    "HMMKNNGenerator",
    "KirschGenerator",
    "KNNBootstrapGenerator",
    "MatalasGenerator",
    "MultisitePhaseRandomizationGenerator",
    "MultiSiteHMMGenerator",
    "PhaseRandomizationGenerator",
    "SMARTAGenerator",
    "SPARTAGenerator",
    "ThomasFieringGenerator",
    "WARMGenerator",
]
