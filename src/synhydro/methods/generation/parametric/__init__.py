"""
Parametric generation methods for SynHydro.
"""

from synhydro.methods.generation.parametric.vine_copula import (
    VineCopulaGenerator,
)
from synhydro.methods.generation.parametric.arfima import ARFIMAGenerator
from synhydro.methods.generation.parametric.gaussian_copula import (
    GaussianCopulaGenerator,
)
from synhydro.methods.generation.parametric.matalas import MatalasGenerator
from synhydro.methods.generation.parametric.hmm_knn import HMMKNNGenerator
from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.generation.parametric.warm import WARMGenerator
from synhydro.methods.generation.parametric.smarta import SMARTAGenerator
from synhydro.methods.generation.parametric.sparta import SPARTAGenerator

__all__ = [
    "ARFIMAGenerator",
    "GaussianCopulaGenerator",
    "HMMKNNGenerator",
    "MatalasGenerator",
    "MultiSiteHMMGenerator",
    "SMARTAGenerator",
    "SPARTAGenerator",
    "ThomasFieringGenerator",
    "VineCopulaGenerator",
    "WARMGenerator",
]
