"""
Pre-built pipeline configurations for common generator-disaggregator combinations.

This module provides convenience classes that wrap commonly used combinations
of generators and disaggregators into ready-to-use pipelines.
"""

from sglib.pipelines.prebuilt import (
    KirschNowakPipeline,
    ThomasFieringNowakPipeline,
)

__all__ = [
    'KirschNowakPipeline',
    'ThomasFieringNowakPipeline',
]
