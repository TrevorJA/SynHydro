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
    GaussianCopulaGenerator,
    HMMKNNGenerator,
    MatalasGenerator,
    MultiSiteHMMGenerator,
    ThomasFieringGenerator,
    VineCopulaGenerator,
    WARMGenerator,
)

__all__ = [
    "ARFIMAGenerator",
    "GaussianCopulaGenerator",
    "HMMKNNGenerator",
    "KirschGenerator",
    "KNNBootstrapGenerator",
    "MatalasGenerator",
    "MultisitePhaseRandomizationGenerator",
    "MultiSiteHMMGenerator",
    "PhaseRandomizationGenerator",
    "ThomasFieringGenerator",
    "VineCopulaGenerator",
    "WARMGenerator",
]
