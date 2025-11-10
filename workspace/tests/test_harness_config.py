# -*- coding: utf-8 -*-
"""
Phase 2 Tests: Config Loading and Search Paths

Tests config file loading, search path resolution, and tier parameter extraction.
"""
import pytest
import sys
from pathlib import Path
import json

# Add workspace/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'workspace' / 'src'))


class TestConfigLoading:
    """Test config file loading from standard paths."""
    
    def test_tier1_config_exists_and_loads(self):
        """Should find and load Tier 1 config."""
        # From tests/ directory, config is at ../config/
        config_path = Path(__file__).parent.parent / 'config' / 'config_tier1_relativistic.json'
        assert config_path.exists(), f"Tier 1 config not found at {config_path}"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        assert isinstance(config, dict)
        assert "tests" in config or "variants" in config
    
    def test_tier3_config_exists_and_loads(self):
        """Should find and load Tier 3 config."""
        config_path = Path(__file__).parent.parent / 'config' / 'config_tier3_energy.json'
        assert config_path.exists(), f"Tier 3 config not found at {config_path}"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        assert isinstance(config, dict)
    
    def test_config_has_runtime_settings(self):
        """Config should have runtime settings."""
        config_path = Path(__file__).parent.parent / 'config' / 'config_tier1_relativistic.json'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Should have run_settings or parameters
        assert "run_settings" in config or "parameters" in config
    
    def test_config_has_simulation_params(self):
        """Config should have simulation parameters."""
        config_path = Path(__file__).parent.parent / 'config' / 'config_tier1_relativistic.json'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Should have tests or variants
        assert "tests" in config or "variants" in config


class TestConfigSearchPaths:
    """Test config file search path resolution."""
    
    def test_config_dir_relative_to_workspace(self):
        """Config directory should be at workspace/config/."""
        # From tests/, go up to workspace/ root, then to config/
        config_dir = Path(__file__).parent.parent / 'config'
        
        assert config_dir.exists(), f"Config directory not found at {config_dir}"
        assert config_dir.is_dir()
    
    def test_can_find_all_tier_configs(self):
        """Should be able to enumerate all tier config files."""
        config_dir = Path(__file__).parent.parent / 'config'
        tier_configs = list(config_dir.glob('config_tier*.json'))
        
        # Should have at least tier 1-5 configs
        assert len(tier_configs) >= 5, f"Expected at least 5 tier configs, found {len(tier_configs)}"
        
        # Check we have tier 1
        tier1_found = any('tier1' in cfg.name for cfg in tier_configs)
        assert tier1_found, "Tier 1 config not found"
