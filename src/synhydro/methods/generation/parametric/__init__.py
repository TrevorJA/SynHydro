"""
Parametric generation methods for SynHydro.
"""
from synhydro.methods.generation.parametric.matalas import MATALASGenerator
from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.methods.generation.parametric.thomas_fiering import ThomasFieringGenerator
from synhydro.methods.generation.parametric.warm import WARMGenerator

__all__ = ['MATALASGenerator', 'MultiSiteHMMGenerator', 'ThomasFieringGenerator', 'WARMGenerator']
