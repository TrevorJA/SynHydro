"""
Generation methods for SynHydro.
"""

from synhydro.methods.generation.nonparametric import (
    KirschGenerator,
    KNNBootstrapGenerator,
    PhaseRandomizationGenerator,
)
from synhydro.methods.generation.parametric import (
    ARFIMAGenerator,
    GaussianCopulaGenerator,
    MATALASGenerator,
    MultiSiteHMMGenerator,
    ThomasFieringGenerator,
    VineCopulaGenerator,
    WARMGenerator,
)

__all__ = [
    "ARFIMAGenerator",
    "GaussianCopulaGenerator",
    "KirschGenerator",
    "KNNBootstrapGenerator",
    "MATALASGenerator",
    "MultiSiteHMMGenerator",
    "PhaseRandomizationGenerator",
    "ThomasFieringGenerator",
    "VineCopulaGenerator",
    "WARMGenerator",
]
