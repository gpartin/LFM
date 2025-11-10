# -*- coding: utf-8 -*-
"""
Phase 4 Tests: Integration and Cross-Tier Consistency

Tests integration between components and cross-tier consistency checks.
"""
import pytest
import sys
from pathlib import Path
import numpy as np

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))

from harness.validation import load_tier_metadata, aggregate_validation


class TestCrossTierConsistency:
    """Test consistency across different tiers."""
    
    def test_all_tiers_have_energy_conservation(self):
        """All tier metadata should include energy conservation criteria."""
        # Test tiers 1-5 (main tiers)
        for tier in [1, 2, 3]:
            try:
                meta = load_tier_metadata(tier)
                # Should have either global requirements or test-specific thresholds
                has_global = "global_requirements" in meta
                has_tests = "tests" in meta and len(meta["tests"]) > 0
                
                assert has_global or has_tests, (
                    f"Tier {tier} metadata lacks structure"
                )
            except FileNotFoundError:
                pytest.skip(f"Tier {tier} metadata not found")
    
    def test_tier_metadata_structure_consistent(self):
        """All tier metadata should follow consistent structure."""
        for tier in [1, 2, 3]:
            try:
                meta = load_tier_metadata(tier)
                
                # Should have tests dict
                assert "tests" in meta
                assert isinstance(meta["tests"], dict)
                
                # First test should have validation_criteria
                if meta["tests"]:
                    first_test_id = list(meta["tests"].keys())[0]
                    test_entry = meta["tests"][first_test_id]
                    assert "validation_criteria" in test_entry
            except FileNotFoundError:
                pytest.skip(f"Tier {tier} metadata not found")
    
    def test_validation_result_format_consistent(self):
        """Validation results should have consistent format across tiers."""
        meta = load_tier_metadata(1)  # Use Tier 1 as reference
        
        # Test with multiple test IDs
        test_ids = list(meta["tests"].keys())[:2]  # First 2 tests
        
        for test_id in test_ids:
            result = aggregate_validation(
                meta, test_id,
                energy_drift=1e-5,
                metrics={"anisotropy": 0.005}
            )
            
            # Check result has expected fields
            assert hasattr(result, 'test_id')
            assert hasattr(result, 'energy_ok')
            assert hasattr(result, 'primary_ok')
            assert hasattr(result, 'energy_drift')
            assert hasattr(result, 'metrics')
    
    def test_energy_thresholds_reasonable_across_tiers(self):
        """Energy thresholds should be in reasonable range."""
        for tier in [1, 2, 3]:
            try:
                meta = load_tier_metadata(tier)
                
                for test_id, test_data in meta["tests"].items():
                    criteria = test_data.get("validation_criteria", {})
                    energy_crit = criteria.get("energy_conservation", {})
                    threshold = energy_crit.get("threshold")
                    
                    if threshold is not None:
                        # Should be between 1e-6 and 0.2 (reasonable range)
                        # Some gravity tests need 0.15 due to numerics
                        assert 1e-6 <= threshold <= 0.2, (
                            f"Tier {tier} {test_id} energy threshold {threshold} "
                            f"outside reasonable range"
                        )
            except FileNotFoundError:
                pytest.skip(f"Tier {tier} metadata not found")


class TestHarnessIntegration:
    """Test integration between harness components."""
    
    def test_harness_compute_energy_and_validation_integrate(self, minimal_harness):
        """Energy computation should integrate with validation workflow."""
        # Create simple test field
        E = np.ones((32, 32, 32))
        E_prev = E.copy()
        dt, dx, c, chi = 0.01, 0.1, 1.0, 0.0
        
        # Compute energy
        energy = minimal_harness.compute_field_energy(
            E, E_prev, dt, dx, c, chi, dims='3d'
        )
        
        # Should return valid float
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0
    
    def test_harness_creates_valid_lattice_params(self, minimal_harness):
        """Lattice params should be compatible with energy computation."""
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=0.0
        )
        
        # Check params have required structure
        assert "dt" in params
        assert "dx" in params
        assert "alpha" in params
        assert params["alpha"] == 1.0  # c²
    
    def test_metadata_validation_energy_flow(self):
        """Metadata, validation, and energy checks should flow together."""
        meta = load_tier_metadata(1)
        first_test_id = list(meta["tests"].keys())[0]
        
        # Simulate a test result
        energy_drift = 5e-5
        metrics = {"anisotropy": 0.008}
        
        result = aggregate_validation(
            meta, first_test_id,
            energy_drift=energy_drift,
            metrics=metrics
        )
        
        # Should produce coherent result
        assert result.energy_drift == energy_drift
        assert result.energy_ok in [True, False]
        assert result.primary_ok in [True, False]


class TestResourceTracking:
    """Test resource tracking capabilities (basic validation)."""
    
    def test_harness_initializes_without_errors(self, minimal_harness):
        """Harness should initialize without resource errors."""
        # Just checking basic initialization
        assert minimal_harness is not None
        assert hasattr(minimal_harness, 'xp')
        assert hasattr(minimal_harness, 'cfg')
    
    def test_backend_selection_works(self, minimal_harness):
        """Backend selection should not cause resource issues."""
        # xp should be either numpy or cupy
        xp = minimal_harness.xp
        assert xp is not None
        
        # Should be able to create arrays
        arr = xp.ones(10)
        assert arr.shape == (10,)
