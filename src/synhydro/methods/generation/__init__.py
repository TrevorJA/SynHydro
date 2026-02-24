"""
Generation methods for SynHydro.
"""
from synhydro.methods.generation.nonparametric import (
    KirschGenerator,
    PhaseRandomizationGenerator,
)
from synhydro.methods.generation.parametric import (
    MATALASGenerator,
    MultiSiteHMMGenerator,
    ThomasFieringGenerator,
    WARMGenerator,
)

__all__ = [
    'KirschGenerator',
    'MATALASGenerator',
    'MultiSiteHMMGenerator',
    'PhaseRandomizationGenerator',
    'ThomasFieringGenerator',
    'WARMGenerator',
]
