# -*- coding: utf-8 -*-
"""
Phase 1 Tests: Lattice Parameters Construction

Tests make_lattice_params helper function that creates standardized
parameter dicts for lattice_step() calls.
"""
import pytest
import sys
from pathlib import Path
import numpy as np

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))


class TestMakeLatticeParams:
    """Test make_lattice_params helper function."""
    
    def test_creates_dict_with_required_keys(self, minimal_harness):
        """Should create dict with dt, dx, alpha, beta, chi."""
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=0.0
        )
        
        assert isinstance(params, dict)
        assert "dt" in params
        assert "dx" in params
        assert "alpha" in params
        assert "beta" in params
        assert "chi" in params
    
    def test_alpha_equals_c_squared(self, minimal_harness):
        """Alpha should be c² (correct Klein-Gordon form)."""
        c = 2.0
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=c, chi=0.0
        )
        
        assert params["alpha"] == c * c
        assert params["alpha"] == 4.0
    
    def test_beta_defaults_to_one(self, minimal_harness):
        """Beta should default to 1.0."""
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=0.0
        )
        
        assert params["beta"] == 1.0
    
    def test_preserves_chi_parameter(self, minimal_harness):
        """Should preserve chi parameter (scalar or array)."""
        chi_scalar = 0.5
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=chi_scalar
        )
        assert params["chi"] == chi_scalar
        
        # Test with array
        chi_array = np.array([0.1, 0.2, 0.3])
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=chi_array
        )
        assert np.array_equal(params["chi"], chi_array)
    
    def test_includes_backend_parameter(self, minimal_harness):
        """Should include backend parameter (default 'baseline')."""
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=0.0
        )
        
        # Backend should be present (either 'baseline' or explicitly set)
        assert "backend" in params
    
    def test_kwargs_passthrough_for_tier_specific(self, minimal_harness):
        """Should pass through additional kwargs for tier-specific parameters."""
        params = minimal_harness.make_lattice_params(
            dt=0.01, dx=0.1, c=1.0, chi=0.0,
            B_field=np.array([0, 0, 1]),  # EM tier parameter
            source=np.zeros((64, 64)),    # Source term
            custom_param=42
        )
        
        assert "B_field" in params
        assert "source" in params
        assert "custom_param" in params
        assert params["custom_param"] == 42
