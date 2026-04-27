"""
Tests for synhydro.plotting._utils helpers exposed in the public API.

Covers warn_if_many_realizations, warn_if_few_realizations, and
validate_timestep.
"""

import logging

import pytest

from synhydro.plotting._utils import (
    validate_timestep,
    warn_if_few_realizations,
    warn_if_many_realizations,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# warn_if_many_realizations
# ----------------------------------------------------------------------


def test_warn_if_many_realizations_above_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="synhydro.plotting._utils"):
        warn_if_many_realizations(2000, threshold=1000)
    assert any("2000" in rec.getMessage() for rec in caplog.records)


def test_warn_if_many_realizations_below_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="synhydro.plotting._utils"):
        warn_if_many_realizations(50, threshold=1000)
    assert not any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_warn_if_many_realizations_with_context(caplog):
    with caplog.at_level(logging.WARNING, logger="synhydro.plotting._utils"):
        warn_if_many_realizations(2000, threshold=1000, context="show_members")
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("show_members" in m for m in msgs)


# ----------------------------------------------------------------------
# warn_if_few_realizations
# ----------------------------------------------------------------------


def test_warn_if_few_realizations_below_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="synhydro.plotting._utils"):
        warn_if_few_realizations(5, threshold=30)
    assert any("5" in rec.getMessage() for rec in caplog.records)


def test_warn_if_few_realizations_above_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="synhydro.plotting._utils"):
        warn_if_few_realizations(100, threshold=30)
    assert not any(rec.levelno == logging.WARNING for rec in caplog.records)


# ----------------------------------------------------------------------
# validate_timestep
# ----------------------------------------------------------------------


def test_validate_timestep_coarser_ok(small_ensemble):
    """Daily ensemble can be plotted at monthly: coarser is fine."""
    # Should not raise.
    validate_timestep(small_ensemble, "monthly")


def test_validate_timestep_finer_raises(monthly_ensemble):
    """Monthly ensemble cannot be plotted at daily: finer is rejected."""
    with pytest.raises(ValueError, match="Cannot plot at timestep"):
        validate_timestep(monthly_ensemble, "daily")


def test_validate_timestep_no_frequency_is_noop(small_ensemble):
    """If ensemble.frequency is None, validate_timestep should not raise."""
    original_freq = small_ensemble.metadata.time_resolution
    try:
        small_ensemble.metadata.time_resolution = None
        # Even with a finer requested timestep, no-op when frequency is None.
        validate_timestep(small_ensemble, "daily")
    finally:
        small_ensemble.metadata.time_resolution = original_freq


def test_validate_timestep_same_frequency_ok(small_ensemble):
    """Daily ensemble at daily timestep should not raise."""
    validate_timestep(small_ensemble, "daily")
