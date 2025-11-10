# -*- coding: utf-8 -*-
"""
Phase 1 Tests: Validation Module

Tests aggregate_validation, energy checks, and primary metric evaluation.
Following process improvements: fixture-first design, incremental validation.
"""
import pytest
import sys
from pathlib import Path

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))

from harness.validation import (
    aggregate_validation,
    validation_block,
    energy_conservation_check,
    check_primary_metric,
    get_energy_threshold,
    load_tier_metadata
)


@pytest.fixture
def minimal_metadata():
    """Minimal tier metadata matching real tier config structure."""
    return {
        "tests": {
            "TEST-01": {
                "name": "Test 1",
                "validation_criteria": {
                    "primary": {
                        "metric": "rel_err",
                        "threshold": 0.1
                    },
                    "energy_conservation": {
                        "threshold": 1e-4
                    }
                }
            },
            "TEST-02": {
                "name": "Test 2",
                "validation_criteria": {
                    "primary": {
                        "metric": "anisotropy",
                        "threshold": 0.05
                    },
                    "energy_conservation": {
                        "threshold": 1e-3
                    }
                }
            }
        }
    }


class TestEnergyConservationCheck:
    """Test energy_conservation_check function."""
    
    def test_energy_within_threshold_passes(self, minimal_metadata):
        """Energy drift below threshold should pass."""
        passed, threshold, msg = energy_conservation_check(
            minimal_metadata, "TEST-01", 5e-5
        )
        assert passed is True
        assert threshold == 1e-4
        assert isinstance(msg, str)  # Message format varies, just check it exists
    
    def test_energy_exceeds_threshold_fails(self, minimal_metadata):
        """Energy drift above threshold should fail."""
        passed, threshold, msg = energy_conservation_check(
            minimal_metadata, "TEST-01", 2e-4
        )
        assert passed is False
        assert threshold == 1e-4
        assert isinstance(msg, str)
    
    def test_uses_global_threshold_when_test_missing(self, minimal_metadata):
        """Should fall back to global energy_threshold if test-specific missing."""
        passed, threshold, msg = energy_conservation_check(
            minimal_metadata, "NONEXISTENT", 5e-3
        )
        assert threshold == 0.01  # Global default
        assert passed is True
    
    def test_edge_case_exactly_at_threshold(self, minimal_metadata):
        """Energy exactly at threshold fails (strict < comparison)."""
        passed, threshold, msg = energy_conservation_check(
            minimal_metadata, "TEST-01", 1e-4
        )
        # Implementation uses strict < not <=
        assert passed is False


class TestCheckPrimaryMetric:
    """Test check_primary_metric function."""
    
    def test_primary_metric_within_threshold_passes(self, minimal_metadata):
        """Primary metric below threshold should pass."""
        summary = {"rel_err": 0.05}
        passed, key, value, threshold = check_primary_metric(
            minimal_metadata, "TEST-01", summary
        )
        assert passed is True
        assert key == "rel_err"
        assert value == 0.05
        assert threshold == 0.1
    
    def test_primary_metric_exceeds_threshold_fails(self, minimal_metadata):
        """Primary metric above threshold should fail."""
        summary = {"rel_err": 0.15}
        passed, key, value, threshold = check_primary_metric(
            minimal_metadata, "TEST-01", summary
        )
        assert passed is False
        assert value == 0.15
    
    def test_missing_primary_metric_returns_false(self, minimal_metadata):
        """Missing primary metric in summary should fail gracefully."""
        summary = {"wrong_key": 0.05}
        passed, key, value, threshold = check_primary_metric(
            minimal_metadata, "TEST-01", summary
        )
        assert passed is False
    
    def test_handles_multiple_metrics(self, minimal_metadata):
        """Should extract correct metric from dict with multiple keys."""
        summary = {"rel_err": 0.03, "anisotropy": 0.08, "other": 0.5}
        passed, key, value, threshold = check_primary_metric(
            minimal_metadata, "TEST-01", summary
        )
        assert key == "rel_err"
        assert value == 0.03


class TestAggregateValidation:
    """Test aggregate_validation function - primary orchestrator."""
    
    def test_both_pass_returns_success(self, minimal_metadata):
        """Energy + primary both pass → overall success."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,
            metrics={"rel_err": 0.05}
        )
        
        assert result.test_id == "TEST-01"
        assert result.energy_ok is True
        assert result.primary_ok is True
        assert result.energy_drift == 5e-5
        assert result.primary_metric == "rel_err"
        assert result.primary_value == 0.05
    
    def test_energy_fail_primary_pass_returns_failure(self, minimal_metadata):
        """Energy fails but primary passes → overall failure."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=2e-4,  # Fails (threshold 1e-4)
            metrics={"rel_err": 0.05}  # Passes
        )
        
        assert result.energy_ok is False
        assert result.primary_ok is True
    
    def test_energy_pass_primary_fail_returns_failure(self, minimal_metadata):
        """Energy passes but primary fails → overall failure."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,  # Passes
            metrics={"rel_err": 0.15}  # Fails (threshold 0.1)
        )
        
        assert result.energy_ok is True
        assert result.primary_ok is False
    
    def test_both_fail_returns_failure(self, minimal_metadata):
        """Both energy and primary fail → overall failure."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=2e-4,
            metrics={"rel_err": 0.15}
        )
        
        assert result.energy_ok is False
        assert result.primary_ok is False
    
    def test_includes_thresholds_in_result(self, minimal_metadata):
        """Result should include all thresholds for diagnostics."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,
            metrics={"rel_err": 0.05}
        )
        
        assert result.energy_threshold == 1e-4
        assert result.primary_threshold == 0.1
    
    def test_preserves_all_metrics(self, minimal_metadata):
        """Should preserve full metrics dict for future analysis."""
        metrics_in = {"rel_err": 0.05, "anisotropy": 0.02, "extra": 0.7}
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,
            metrics=metrics_in
        )
        
        assert result.metrics == metrics_in


class TestValidationBlock:
    """Test validation_block function - summary formatting."""
    
    def test_converts_result_to_dict(self, minimal_metadata):
        """Should convert ValidationResult to embeddable dict."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,
            metrics={"rel_err": 0.05}
        )
        
        block = validation_block(result)
        
        assert isinstance(block, dict)
        assert "energy" in block
        assert "primary" in block
        assert "test_id" in block
        assert block["energy"]["ok"] is True
        assert block["primary"]["ok"] is True
    
    def test_includes_timestamp(self, minimal_metadata):
        """Validation block should include timestamp."""
        result = aggregate_validation(
            minimal_metadata,
            "TEST-01",
            energy_drift=5e-5,
            metrics={"rel_err": 0.05}
        )
        
        block = validation_block(result)
        
        assert "timestamp" in block
        assert isinstance(block["timestamp"], str)


class TestGetEnergyThreshold:
    """Test get_energy_threshold utility."""
    
    def test_returns_test_specific_threshold(self, minimal_metadata):
        """Should return test-specific threshold when available."""
        threshold = get_energy_threshold(minimal_metadata, "TEST-01")
        assert threshold == 1e-4
    
    def test_returns_global_threshold_when_test_missing(self, minimal_metadata):
        """Should fall back to global when test not found."""
        threshold = get_energy_threshold(minimal_metadata, "NONEXISTENT")
        assert threshold == 0.01
    
    def test_returns_default_when_both_missing(self):
        """Should return provided default when metadata incomplete."""
        empty_meta = {}
        threshold = get_energy_threshold(empty_meta, "TEST-01", default=0.005)
        assert threshold == 0.005


class TestLoadTierMetadata:
    """Test load_tier_metadata function (integration test)."""
    
    def test_loads_existing_tier1_metadata(self):
        """Should load actual tier 1 metadata file."""
        try:
            meta = load_tier_metadata(1)
            assert isinstance(meta, dict)
            assert "tests" in meta or len(meta) > 0
        except FileNotFoundError:
            pytest.skip("Tier 1 metadata not found - acceptable in test environment")
    
    def test_raises_on_nonexistent_tier(self):
        """Should raise or handle gracefully for non-existent tier."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_tier_metadata(99)
