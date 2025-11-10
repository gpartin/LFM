# -*- coding: utf-8 -*-
"""
Phase 1 Tests: Metadata Loading and Structure

Tests load_tier_metadata, metadata structure validation, and test lookup.
"""
import pytest
import sys
from pathlib import Path

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))

from harness.validation import load_tier_metadata


class TestLoadTierMetadata:
    """Test load_tier_metadata function for all tiers."""
    
    def test_loads_tier1_metadata(self):
        """Should load Tier 1 relativistic metadata."""
        meta = load_tier_metadata(1)
        assert isinstance(meta, dict)
        assert "tests" in meta
        assert isinstance(meta["tests"], dict)
        # Tier 1 has REL-01 through REL-17
        assert len(meta["tests"]) > 0
    
    def test_loads_tier2_metadata(self):
        """Should load Tier 2 gravity metadata."""
        try:
            meta = load_tier_metadata(2)
            assert isinstance(meta, dict)
            assert "tests" in meta
        except FileNotFoundError:
            pytest.skip("Tier 2 metadata not yet created")
    
    def test_loads_tier3_metadata(self):
        """Should load Tier 3 energy metadata."""
        try:
            meta = load_tier_metadata(3)
            assert isinstance(meta, dict)
            assert "tests" in meta
        except FileNotFoundError:
            pytest.skip("Tier 3 metadata not yet created")
    
    def test_raises_on_invalid_tier(self):
        """Should raise error for non-existent tier."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_tier_metadata(99)


class TestMetadataStructure:
    """Test metadata structure conforms to expected schema."""
    
    def test_has_tests_dict(self):
        """Metadata should have 'tests' key with dict value."""
        meta = load_tier_metadata(1)
        assert "tests" in meta
        assert isinstance(meta["tests"], dict)
    
    def test_test_entries_have_required_fields(self):
        """Each test entry should have name and validation_criteria."""
        meta = load_tier_metadata(1)
        tests = meta["tests"]
        
        # Check first test (REL-01)
        first_test_id = list(tests.keys())[0]
        test_entry = tests[first_test_id]
        
        assert "name" in test_entry
        assert "validation_criteria" in test_entry
        assert isinstance(test_entry["validation_criteria"], dict)
    
    def test_validation_criteria_structure(self):
        """Validation criteria should have primary and energy_conservation."""
        meta = load_tier_metadata(1)
        tests = meta["tests"]
        first_test_id = list(tests.keys())[0]
        criteria = tests[first_test_id]["validation_criteria"]
        
        assert "primary" in criteria
        assert "energy_conservation" in criteria
        assert "metric" in criteria["primary"]
        assert "threshold" in criteria["primary"]
        assert "threshold" in criteria["energy_conservation"]


class TestTestLookup:
    """Test looking up specific tests in metadata."""
    
    def test_can_lookup_test_by_id(self):
        """Should be able to access test by ID."""
        meta = load_tier_metadata(1)
        tests = meta["tests"]
        
        # Tier 1 should have REL-01
        assert "REL-01" in tests
        test = tests["REL-01"]
        assert test["name"] == "Isotropy — Coarse Grid"
    
    def test_missing_test_returns_none(self):
        """Looking up non-existent test should return None gracefully."""
        meta = load_tier_metadata(1)
        tests = meta["tests"]
        
        # Non-existent test
        assert tests.get("NONEXISTENT") is None
    
    def test_all_test_ids_follow_convention(self):
        """All test IDs should follow TIER-NN convention."""
        meta = load_tier_metadata(1)
        tests = meta["tests"]
        
        for test_id in tests.keys():
            # Should match pattern like REL-01, GRAV-12, etc.
            assert "-" in test_id
            parts = test_id.split("-")
            assert len(parts) == 2
            assert parts[1].isdigit()
