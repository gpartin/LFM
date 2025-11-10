# -*- coding: utf-8 -*-
"""
Mock configuration and metadata fixtures for testing.
"""
import json
from pathlib import Path
from typing import Dict, Any


def create_minimal_tier_config() -> Dict[str, Any]:
    """
    Create minimal valid tier configuration.
    
    Returns:
        Config dict suitable for testing
    """
    return {
        "lattice": {
            "N": 64,
            "dx": 0.1,
            "dt": 0.01,
            "T": 1.0
        },
        "physics": {
            "c": 1.0,
            "chi": 0.0
        },
        "hardware": {
            "gpu_enabled": False,
            "backend": "baseline"
        }
    }


def create_minimal_tier_metadata(tier_number: int) -> Dict[str, Any]:
    """
    Create minimal valid tier metadata.
    
    Args:
        tier_number: Tier number (1-7)
        
    Returns:
        Metadata dict suitable for testing
    """
    return {
        "tier": tier_number,
        "tier_name": f"Tier {tier_number}",
        "description": f"Test metadata for tier {tier_number}",
        "energy_threshold": 1e-4,
        "tests": [
            {
                "id": f"TEST-{tier_number:02d}",
                "name": f"Mock test {tier_number}",
                "primary_metric": "test_metric",
                "threshold": 0.1,
                "rationale": "Mock test for validation"
            }
        ]
    }


def create_test_config_file(tmp_path: Path, config: Dict[str, Any], 
                            filename: str = "test_config.json") -> Path:
    """
    Create temporary config file for testing.
    
    Args:
        tmp_path: Pytest tmp_path fixture
        config: Config dictionary
        filename: Config filename
        
    Returns:
        Path to created config file
    """
    config_path = tmp_path / filename
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    return config_path


def create_test_metadata_file(tmp_path: Path, metadata: Dict[str, Any],
                              tier_number: int) -> Path:
    """
    Create temporary metadata file for testing.
    
    Args:
        tmp_path: Pytest tmp_path fixture
        metadata: Metadata dictionary
        tier_number: Tier number (1-7)
        
    Returns:
        Path to created metadata file
    """
    metadata_path = tmp_path / f"tier{tier_number}_validation_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def create_malformed_json_file(tmp_path: Path, 
                               filename: str = "malformed.json") -> Path:
    """
    Create malformed JSON file for error testing.
    
    Args:
        tmp_path: Pytest tmp_path fixture
        filename: Filename
        
    Returns:
        Path to malformed file
    """
    bad_path = tmp_path / filename
    with open(bad_path, 'w', encoding='utf-8') as f:
        f.write('{"incomplete": "json"')  # Missing closing brace
    return bad_path


def create_non_utf8_file(tmp_path: Path,
                        filename: str = "bad_encoding.json") -> Path:
    """
    Create file with non-UTF-8 encoding for error testing.
    
    Args:
        tmp_path: Pytest tmp_path fixture
        filename: Filename
        
    Returns:
        Path to bad encoding file
    """
    bad_path = tmp_path / filename
    with open(bad_path, 'wb') as f:
        # Write CP1252 encoded text with special characters
        f.write(b'{"name": "caf\xe9"}')  # CP1252 é character
    return bad_path
