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
    MATALASGenerator,
    MultiSiteHMMGenerator,
    ThomasFieringGenerator,
    WARMGenerator,
)

__all__ = [
    'ARFIMAGenerator',
    'KirschGenerator',
    'KNNBootstrapGenerator',
    'MATALASGenerator',
    'MultiSiteHMMGenerator',
    'PhaseRandomizationGenerator',
    'ThomasFieringGenerator',
    'WARMGenerator',
]
