#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.

"""
Website Sync Validation
=======================
Validates that website experiment metadata matches test harness configs.

Called by:
  - pre_commit_validation.py (before git commit)
  - CI/CD pipeline (before deploy)

Purpose:
  Ensure website always reflects actual validated test parameters.
  Prevent drift between test harness (source of truth) and website.

Exit codes:
  0 - Website in sync with test harness
  1 - Website out of sync (mismatch detected)
  2 - Error running validation

Usage:
  python workspace/tools/validate_website_sync.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Paths relative to workspace root
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = WORKSPACE_ROOT / "config"
WEBSITE_GENERATED_FILE = WORKSPACE_ROOT / "website" / "src" / "data" / "research-experiments-generated.ts"

# Tier definitions (must match generate_website_experiments.js)
TIERS = [
    {"tier": 1, "name": "Relativistic", "config": "config_tier1_relativistic.json", "expected": 16},
    {"tier": 2, "name": "Gravity", "config": "config_tier2_gravityanalogue.json", "expected": 26},
    {"tier": 3, "name": "Energy", "config": "config_tier3_energy.json", "expected": 11},
    {"tier": 4, "name": "Quantization", "config": "config_tier4_quantization.json", "expected": 9},
    {"tier": 5, "name": "Electromagnetic", "config": "config_tier5_electromagnetic.json", "expected": 21},
    {"tier": 6, "name": "Coupling", "config": "config_tier6_coupling.json", "expected": 12},
    {"tier": 7, "name": "Thermodynamics", "config": "config_tier7_thermodynamics.json", "expected": 5},
]


def load_test_configs() -> Dict[str, Dict]:
    """
    Load all test harness configs.
    
    Returns:
        Dict[test_id] = {
            'tier': int,
            'tierName': str,
            'config_file': str,
            'latticeSize': int,
            'dt': float,
            'dx': float,
            'steps': int,
            'chi': float,
        }
    """
    tests = {}
    
    for tier_info in TIERS:
        config_path = CONFIG_DIR / tier_info["config"]
        
        if not config_path.exists():
            print(f"Warning: Config not found: {config_path}")
            continue
        
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = json.load(f)
        
        base_params = config.get("parameters", {})
        
        # Handle both "variants" and "tests" schema
        test_list = config.get("variants") or config.get("tests", [])
        
        for test in test_list:
            test_id = test.get("test_id") or test.get("id")
            if not test_id:
                continue
            
            # Extract chi value - can be scalar or gradient
            # Match the logic in experiment_sync_api.js extractChi()
            chi_value = None
            if "chi_const" in test:
                chi_value = test["chi_const"]
            elif "chi_gradient" in test:
                chi_value = test["chi_gradient"]  # list [min, max]
            else:
                # Check all common chi field names
                chi_keys = ['chi', 'chi_base', 'chi_bg', 'chi_left', 'chi_right', 
                           'chi_inside', 'chi_outside', 'chi_barrier']
                for key in chi_keys:
                    if key in test:
                        chi_value = test[key]
                        break
                
                # Fallback to base params
                if chi_value is None:
                    if "chi" in base_params:
                        chi_value = base_params["chi"]
                    else:
                        chi_value = 0.0
            
            # Extract latticeSize with same priority as generate_website_experiments.js:
            # test.grid_points -> test.grid_size -> base_params.grid_points -> base_params.N -> 256
            lattice_size = test.get("grid_points")
            if lattice_size is None:
                lattice_size = test.get("grid_size")
            if lattice_size is None:
                lattice_size = base_params.get("grid_points")
            if lattice_size is None:
                lattice_size = base_params.get("N", 256)
            
            # Extract dt with test-specific override priority
            dt_value = test.get("dt")
            if dt_value is None:
                dt_value = base_params.get("dt") or base_params.get("time_step", 0.001)
            
            # Extract dx with test-specific override priority
            dx_value = test.get("dx")
            if dx_value is None:
                dx_value = base_params.get("dx") or base_params.get("space_step", 0.01)
            
            tests[test_id] = {
                "tier": tier_info["tier"],
                "tierName": tier_info["name"],
                "config_file": tier_info["config"],
                "latticeSize": lattice_size,
                "dt": dt_value,
                "dx": dx_value,
                "steps": test.get("steps", base_params.get("steps", 5000)),
                "chi": chi_value,
            }
    
    return tests


def parse_website_experiments() -> Dict[str, Dict]:
    """
    Parse website generated TypeScript file.
    
    Returns:
        Dict[test_id] = {
            'tier': int,
            'latticeSize': int,
            'dt': float,
            'dx': float,
            'steps': int,
            'chi': float,
        }
    """
    if not WEBSITE_GENERATED_FILE.exists():
        print(f"❌ Website generated file not found: {WEBSITE_GENERATED_FILE}")
        print("   Run: node workspace/tools/generate_website_experiments.js")
        return {}
    
    with open(WEBSITE_GENERATED_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract JSON from TypeScript export
    # Find the array between 'export const RESEARCH_EXPERIMENTS = [' and '];'
    start_marker = "export const RESEARCH_EXPERIMENTS: ExperimentDefinition[] = ["
    end_marker = "\n];"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"❌ Could not parse website file: start marker not found")
        return {}
    
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print(f"❌ Could not parse website file: end marker not found")
        return {}
    
    json_array_str = "[" + content[start_idx:end_idx].strip() + "]"
    
    try:
        experiments = json.loads(json_array_str)
    except json.JSONDecodeError as e:
        print(f"❌ Could not parse website JSON: {e}")
        return {}
    
    # Convert to dict keyed by testId
    result = {}
    for exp in experiments:
        test_id = exp.get("testId")
        if not test_id:
            continue
        
        initial = exp.get("initialConditions", {})
        result[test_id] = {
            "tier": exp.get("tier"),
            "latticeSize": initial.get("latticeSize"),
            "dt": initial.get("dt"),
            "dx": initial.get("dx"),
            "steps": initial.get("steps"),
            "chi": initial.get("chi"),
        }
    
    return result


def validate_sync() -> Tuple[bool, List[str]]:
    """
    Compare test configs vs website data.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    print("Loading test harness configs...")
    config_tests = load_test_configs()
    print(f"  Found {len(config_tests)} tests in configs\n")
    
    print("Parsing website generated file...")
    website_tests = parse_website_experiments()
    print(f"  Found {len(website_tests)} tests in website\n")
    
    if not website_tests:
        errors.append("Website generated file is empty or invalid")
        return False, errors
    
    config_ids = set(config_tests.keys())
    website_ids = set(website_tests.keys())
    
    # Check for missing tests
    missing_from_website = config_ids - website_ids
    if missing_from_website:
        errors.append(f"Tests missing from website: {sorted(missing_from_website)}")
    
    extra_in_website = website_ids - config_ids
    if extra_in_website:
        errors.append(f"Extra tests in website (not in configs): {sorted(extra_in_website)}")
    
    # Check parameter matches for common tests
    common_ids = config_ids & website_ids
    for test_id in sorted(common_ids):
        config = config_tests[test_id]
        website = website_tests[test_id]
        
        mismatches = []
        
        # Compare tier
        if config["tier"] != website["tier"]:
            mismatches.append(f"tier: config={config['tier']}, website={website['tier']}")
        
        # Compare latticeSize (website uses max dimension if config has array)
        config_size = config["latticeSize"]
        website_size = website["latticeSize"]
        # Normalize config_size: if array, take max dimension
        if isinstance(config_size, list):
            config_size_normalized = max(config_size)
        else:
            config_size_normalized = config_size
        
        if config_size_normalized != website_size:
            mismatches.append(f"latticeSize: config={config['latticeSize']}, website={website['latticeSize']}")
        
        # Compare dt (allow small float differences)
        if abs(config["dt"] - website["dt"]) > 1e-10:
            mismatches.append(f"dt: config={config['dt']}, website={website['dt']}")
        
        # Compare dx (allow small float differences)
        if abs(config["dx"] - website["dx"]) > 1e-10:
            mismatches.append(f"dx: config={config['dx']}, website={website['dx']}")
        
        # Compare steps
        if config["steps"] != website["steps"]:
            mismatches.append(f"steps: config={config['steps']}, website={website['steps']}")
        
        # Compare chi (allow small float differences or list equality)
        config_chi = config["chi"]
        website_chi = website["chi"]
        chi_mismatch = False
        if isinstance(config_chi, list) and isinstance(website_chi, list):
            # Both are lists - compare element-wise
            if len(config_chi) != len(website_chi):
                chi_mismatch = True
            else:
                for c, w in zip(config_chi, website_chi):
                    if abs(c - w) > 1e-10:
                        chi_mismatch = True
                        break
        elif isinstance(config_chi, (int, float)) and isinstance(website_chi, (int, float)):
            # Both are numbers - compare with tolerance
            if abs(config_chi - website_chi) > 1e-10:
                chi_mismatch = True
        else:
            # Type mismatch
            chi_mismatch = True
        
        if chi_mismatch:
            mismatches.append(f"chi: config={config['chi']}, website={website['chi']}")
        
        if mismatches:
            errors.append(f"{test_id}: {', '.join(mismatches)}")
    
    # Check tier counts
    print("Validating tier counts...")
    for tier_info in TIERS:
        tier_num = tier_info["tier"]
        expected = tier_info["expected"]
        
        config_count = sum(1 for t in config_tests.values() if t["tier"] == tier_num)
        website_count = sum(1 for t in website_tests.values() if t["tier"] == tier_num)
        
        print(f"  Tier {tier_num} ({tier_info['name']}): config={config_count}, website={website_count}, expected={expected}")
        
        if config_count != website_count:
            errors.append(f"Tier {tier_num} count mismatch: config={config_count}, website={website_count}")
    
    print()
    
    return len(errors) == 0, errors


def main():
    """Run validation."""
    print("=" * 70)
    print("Website Sync Validation")
    print("=" * 70)
    print()
    
    is_valid, errors = validate_sync()
    
    if is_valid:
        print("[PASS] Website in sync with test harness")
        print("   All test IDs present, parameters match")
        return 0
    else:
        print("[FAIL] Website out of sync with test harness!")
        print()
        print("Errors found:")
        for err in errors:
            print(f"  • {err}")
        print()
        print("To fix:")
        print("  1. Run: node workspace/tools/generate_website_experiments.js")
        print("  2. Verify website data updated")
        print("  3. Retry pre-commit validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
