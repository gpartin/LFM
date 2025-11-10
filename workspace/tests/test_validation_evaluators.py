# -*- coding: utf-8 -*-
"""
Phase 3 Tests: Validation Evaluator Functions

Tests individual validation evaluators: spherical_symmetry, dispersion,
redshift, time_delay, momentum_drift, etc.
"""
import pytest
import sys
from pathlib import Path

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))

from harness.validation import (
    evaluate_spherical_symmetry,
    evaluate_dispersion,
    evaluate_spacelike_correlation,
    evaluate_redshift,
    evaluate_time_delay,
    evaluate_momentum_drift,
    evaluate_invariant_mass,
    evaluate_isotropy,
    evaluate_directional_isotropy,
    evaluate_phase_delay
)


@pytest.fixture
def spherical_symmetry_meta():
    """Metadata for spherical symmetry test."""
    return {
        "tests": {
            "TEST-01": {
                "name": "Spherical Symmetry Test",
                "validation_criteria": {
                    "primary": {"metric": "spherical_error", "threshold": 0.01},
                    "energy_conservation": {"threshold": 1e-4}
                }
            }
        }
    }

@pytest.fixture
def dispersion_meta():
    """Metadata for dispersion test."""
    return {
        "tests": {
            "TEST-01": {
                "name": "Dispersion Test",
                "validation_criteria": {
                    "primary": {"metric": "rel_err", "threshold": 0.05},
                    "energy_conservation": {"threshold": 1e-4}
                }
            }
        }
    }

@pytest.fixture
def generic_meta():
    """Generic metadata that can work for multiple evaluators."""
    return {
        "tests": {
            "TEST-01": {
                "name": "Generic Test",
                "validation_criteria": {
                    "primary": {"metric": "test_metric", "threshold": 0.01},
                    "energy_conservation": {"threshold": 1e-4}
                }
            }
        }
    }


class TestSphericalSymmetryEvaluator:
    """Test evaluate_spherical_symmetry function."""
    
    def test_pass_within_threshold(self, spherical_symmetry_meta):
        """Spherical error below threshold should pass."""
        passed, key, value, threshold = evaluate_spherical_symmetry(
            spherical_symmetry_meta, "TEST-01", 0.005
        )
        assert passed is True
        assert key == "spherical_error"
        assert threshold == 0.01
    
    def test_fail_exceeds_threshold(self, spherical_symmetry_meta):
        """Spherical error above threshold should fail."""
        passed, key, value, threshold = evaluate_spherical_symmetry(
            spherical_symmetry_meta, "TEST-01", 0.015
        )
        assert passed is False


class TestDispersionEvaluator:
    """Test evaluate_dispersion function."""
    
    def test_pass_within_threshold(self, dispersion_meta):
        """Dispersion error below threshold should pass."""
        passed, key, value, threshold = evaluate_dispersion(
            dispersion_meta, "TEST-01", 0.03
        )
        assert passed is True
        assert key == "rel_err"
        assert threshold == 0.05
    
    def test_fail_exceeds_threshold(self, dispersion_meta):
        """Dispersion error above threshold should fail."""
        passed, key, value, threshold = evaluate_dispersion(
            dispersion_meta, "TEST-01", 0.07
        )
        assert passed is False


class TestEvaluatorFunctionsWorkWithGenericMetadata:
    """Test that all evaluator functions work with proper metadata structure."""
    
    def test_spacelike_correlation_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_spacelike_correlation(
            generic_meta, "NONEXISTENT", 0.0005
        )
        # With missing test, should return False but not crash
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_redshift_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_redshift(
            generic_meta, "NONEXISTENT", 0.01
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_time_delay_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_time_delay(
            generic_meta, "NONEXISTENT", 0.005
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_momentum_drift_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_momentum_drift(
            generic_meta, "NONEXISTENT", 5e-5
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_invariant_mass_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_invariant_mass(
            generic_meta, "NONEXISTENT", 0.005
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_isotropy_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_isotropy(
            generic_meta, "NONEXISTENT", 0.005
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_directional_isotropy_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_directional_isotropy(
            generic_meta, "NONEXISTENT", 0.005
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
    
    def test_phase_delay_returns_result(self, generic_meta):
        """Evaluator should return 4-tuple result."""
        passed, key, value, threshold = evaluate_phase_delay(
            generic_meta, "NONEXISTENT", 0.01
        )
        assert isinstance(passed, bool)
        assert isinstance(key, str)
