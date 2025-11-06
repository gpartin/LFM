#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
test_output_requirements.py — Tier Test Output Requirements & Validation
------------------------------------------------------------------------
Purpose:
    Define and validate required outputs for ALL tier test runs.
    Ensures uniformity across tiers and consistency for any new tests.

Requirements Framework:
    - Core requirements (ALL tests must have)
    - Per-tier requirements (tier-specific outputs)
    - Per-test requirements (special outputs for individual tests)

Usage as pytest:
    pytest test_output_requirements.py -v
    pytest test_output_requirements.py::test_tier1_outputs -v
    pytest test_output_requirements.py -k "double_slit" -v

Usage as validation script:
    python test_output_requirements.py --tier 1
    python test_output_requirements.py --test GRAV-16
    python test_output_requirements.py --check-all
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


# ============================================================================
# CORE REQUIREMENTS - ALL TESTS MUST HAVE THESE
# ============================================================================

CORE_REQUIREMENTS = {
    "summary.json": {
        "description": "Test summary with metadata and results",
        "required_keys": [
            "tier",
            "category", 
            "test_id",
            "description",
            "timestamp",
            "status",
            "metrics"
        ],
        "metrics_keys": [
            "peak_cpu_percent",
            "peak_memory_mb",
            "peak_gpu_memory_mb"
        ]
    },
    "readme.txt": {
        "description": "Human-readable summary of this result folder",
        "type": "file"
    },
    "test_metrics_history.json": {
        "description": "Global test metrics database entry",
        "location": "results/test_metrics_history.json",
        "required_keys_per_run": [
            "exit_code",
            "runtime_sec",
            "peak_cpu_percent",
            "peak_memory_mb",
            "peak_gpu_memory_mb",
            "timestamp"
        ]
    },
    "diagnostics/": {
        "description": "Diagnostics directory (may be empty for passing tests)",
        "type": "directory"
    },
    "plots/": {
        "description": "Plots directory for visualizations",
        "type": "directory"
    }
}


# ============================================================================
# TIER-SPECIFIC REQUIREMENTS
# ============================================================================

TIER1_REQUIREMENTS = {
    "category": "Relativistic",
    "required_files": [
        "summary.json",
        "diagnostics/",
        "plots/"
    ],
    "plots_required": [
        # Most tests generate dispersion plots
        "dispersion_*.png"  # Pattern match
    ]
}

TIER2_REQUIREMENTS = {
    "category": "Gravity",
    "required_files": [
        "summary.json",
        "diagnostics/",
        "plots/"
    ],
    "csv_data": [
        # Time-dilation and time-delay tests produce CSV data
        "probe_*.csv",  # Pattern for probe measurements
        "packet_tracking_*.csv"  # Pattern for packet tracking
    ]
}

TIER3_REQUIREMENTS = {
    "category": "Energy",
    "required_files": [
        "summary.json",
        "diagnostics/",
        "plots/"
    ],
    "energy_tracking": [
        "energy_drift_log.csv",  # Energy conservation tracking
        "diagnostics/energy_*.csv"
    ]
}

TIER4_REQUIREMENTS = {
    "category": "Quantization",
    "required_files": [
        "summary.json",
        "diagnostics/",
        "plots/"
    ],
    "spectra_data": [
        "spectrum_*.csv",  # Spectral analysis data
        "eigenstate_*.csv"  # Bound state data
    ]
}


# ============================================================================
# PER-TEST SPECIAL REQUIREMENTS
# ============================================================================

SPECIAL_TEST_REQUIREMENTS = {
    # Double-slit interference tests
    "GRAV-16": {
        "description": "Double-slit wave interference (camera view)",
        "additional_outputs": [
            "plots/interference_pattern.png",
            "plots/double_slit_interference_t*.png",
            "intensity_profile.csv"
        ],
        "validation": {
            "interference_pattern.png": {
                "type": "image",
                "min_size_kb": 10,
                "expected_dimensions": (256, 256),  # Minimum
                "description": "Wave interference pattern showing fringes"
            }
        }
    },
    
    "GRAV-15": {
        "description": "3D wave propagation visualization",
        "additional_outputs": [
            "plots/wave_3d_*.png",
            "plots/wave_evolution.png"
        ]
    },
    
    # Time-dilation tests (require FFT analysis)
    "GRAV-07": {
        "description": "Time dilation - deep potential",
        "additional_outputs": [
            "probe_serial.csv",
            "probe_parallel.csv",
            "plots/time_dilation_comparison.png"
        ],
        "validation": {
            "probe_serial.csv": {
                "type": "csv",
                "required_columns": ["step", "E_real", "E_imag", "time"],
                "min_rows": 100
            }
        }
    },
    
    "GRAV-08": {
        "description": "Time dilation - moderate potential",
        "additional_outputs": [
            "probe_serial.csv",
            "probe_parallel.csv"
        ]
    },
    
    "GRAV-09": {
        "description": "Time dilation - weak potential",
        "additional_outputs": [
            "probe_serial.csv",
            "probe_parallel.csv"
        ]
    },
    
    "GRAV-10": {
        "description": "Time dilation - variable chi profile",
        "additional_outputs": [
            "probe_serial.csv",
            "probe_parallel.csv"
        ]
    },
    
    # Time-delay tests (require packet tracking)
    "GRAV-11": {
        "description": "Shapiro delay - wave packet",
        "additional_outputs": [
            "packet_tracking_serial.csv",
            "packet_tracking_parallel.csv",
            "plots/packet_trajectory.png"
        ],
        "validation": {
            "packet_tracking_serial.csv": {
                "type": "csv",
                "required_columns": ["step", "x_peak", "delay_measured"],
                "min_rows": 10
            }
        }
    },
    
    "GRAV-12": {
        "description": "Shapiro delay - group vs phase velocity",
        "additional_outputs": [
            "packet_tracking_serial.csv",
            "packet_tracking_parallel.csv"
        ]
    },
    
    # Bound state tests
    "QUAN-09": {
        "description": "Uncertainty principle",
        "additional_outputs": [
            "uncertainty_measurements.csv",
            "plots/uncertainty_relation.png"
        ]
    },
    
    "QUAN-10": {
        "description": "Bound state quantization",
        "additional_outputs": [
            "eigenstate_energies.csv",
            "plots/energy_levels.png",
            "plots/wavefunction_n*.png"
        ],
        "validation": {
            "eigenstate_energies.csv": {
                "type": "csv",
                "required_columns": ["n", "E_numeric", "E_analytic", "error"],
                "min_rows": 3
            }
        }
    },
    
    # Cavity spectroscopy
    "QUAN-07": {
        "description": "Cavity mode spectroscopy",
        "additional_outputs": [
            "cavity_spectrum.csv",
            "plots/cavity_modes.png"
        ]
    },
    
    # Energy conservation long-runs
    "ENER-02": {
        "description": "Long-run energy conservation",
        "additional_outputs": [
            "energy_drift_log.csv",
            "plots/energy_drift.png"
        ],
        "validation": {
            "energy_drift_log.csv": {
                "type": "csv",
                "required_columns": ["step", "energy_total", "drift_fraction"],
                "min_rows": 100
            }
        }
    }
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def get_test_output_dir(test_id: str) -> Optional[Path]:
    """Get the output directory for a test."""
    project_root = Path(__file__).parent
    results_dir = project_root / "results"
    
    # Determine tier category from test ID prefix
    category_map = {
        "REL": "Relativistic",
        "GRAV": "Gravity",
        "ENER": "Energy",
        "QUAN": "Quantization"
    }
    
    prefix = test_id.split("-")[0]
    category = category_map.get(prefix)
    if not category:
        return None
    
    test_dir = results_dir / category / test_id
    return test_dir if test_dir.exists() else None


def check_core_requirements(test_dir: Path, test_id: str) -> Tuple[List[str], List[str]]:
    """
    Check core requirements for a test.
    
    Returns:
        (passes, failures) - lists of requirement descriptions
    """
    passes = []
    failures = []
    
    # Check summary.json
    summary_file = test_dir / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Check required keys
            missing_keys = []
            for key in CORE_REQUIREMENTS["summary.json"]["required_keys"]:
                if key not in summary:
                    missing_keys.append(key)
            
            # Check metrics
            if "metrics" in summary and isinstance(summary["metrics"], dict):
                for key in CORE_REQUIREMENTS["summary.json"]["metrics_keys"]:
                    if key not in summary["metrics"]:
                        missing_keys.append(f"metrics.{key}")
            else:
                missing_keys.append("metrics (dict)")
            
            if missing_keys:
                failures.append(f"summary.json missing keys: {', '.join(missing_keys)}")
            else:
                passes.append("summary.json structure valid")
        except Exception as e:
            failures.append(f"summary.json parse error: {e}")
    else:
        failures.append("summary.json not found")
    
    # Check readme.txt presence
    readme_file = test_dir / "readme.txt"
    if readme_file.exists():
        passes.append("readme.txt exists")
    else:
        failures.append("readme.txt not found")

    # Check directories
    for dir_name in ["diagnostics", "plots"]:
        dir_path = test_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            passes.append(f"{dir_name}/ exists")
        else:
            failures.append(f"{dir_name}/ not found")
    
    # Check test_metrics_history.json
    project_root = Path(__file__).parent
    metrics_file = project_root / "results" / "test_metrics_history.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if test_id in history:
                runs = history[test_id].get("runs", [])
                if runs:
                    last_run = runs[-1]
                    missing_keys = []
                    for key in CORE_REQUIREMENTS["test_metrics_history.json"]["required_keys_per_run"]:
                        if key not in last_run:
                            missing_keys.append(key)
                    
                    if missing_keys:
                        failures.append(f"test_metrics_history.json missing: {', '.join(missing_keys)}")
                    else:
                        passes.append("test_metrics_history.json entry valid")
                else:
                    failures.append("test_metrics_history.json has no runs")
            else:
                failures.append(f"test_metrics_history.json missing entry for {test_id}")
        except Exception as e:
            failures.append(f"test_metrics_history.json error: {e}")
    else:
        failures.append("test_metrics_history.json not found")
    
    return passes, failures


def check_special_requirements(test_dir: Path, test_id: str) -> Tuple[List[str], List[str]]:
    """
    Check special per-test requirements.
    
    Returns:
        (passes, failures) - lists of requirement descriptions
    """
    if test_id not in SPECIAL_TEST_REQUIREMENTS:
        return [], []  # No special requirements
    
    passes = []
    failures = []
    special = SPECIAL_TEST_REQUIREMENTS[test_id]
    
    # Check additional output files
    for output_pattern in special.get("additional_outputs", []):
        # Handle glob patterns
        if "*" in output_pattern:
            matches = list(test_dir.glob(output_pattern))
            if matches:
                passes.append(f"Found {len(matches)} file(s) matching {output_pattern}")
            else:
                failures.append(f"No files matching pattern: {output_pattern}")
        else:
            output_file = test_dir / output_pattern
            if output_file.exists():
                passes.append(f"{output_pattern} exists")
            else:
                failures.append(f"{output_pattern} not found")
    
    # Check validation rules
    for filename, rules in special.get("validation", {}).items():
        file_path = test_dir / filename
        if not file_path.exists():
            failures.append(f"Validation target {filename} not found")
            continue
        
        file_type = rules.get("type")
        
        if file_type == "image":
            # Check image file size
            size_kb = file_path.stat().st_size / 1024
            min_size = rules.get("min_size_kb", 0)
            if size_kb >= min_size:
                passes.append(f"{filename} size OK ({size_kb:.1f} KB)")
            else:
                failures.append(f"{filename} too small: {size_kb:.1f} KB < {min_size} KB")
        
        elif file_type == "csv":
            # Check CSV structure
            try:
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    # Check columns
                    required_cols = rules.get("required_columns", [])
                    if reader.fieldnames:
                        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
                        if missing_cols:
                            failures.append(f"{filename} missing columns: {', '.join(missing_cols)}")
                        else:
                            passes.append(f"{filename} columns valid")
                    
                    # Check row count
                    min_rows = rules.get("min_rows", 0)
                    if len(rows) >= min_rows:
                        passes.append(f"{filename} has {len(rows)} rows (>= {min_rows})")
                    else:
                        failures.append(f"{filename} only {len(rows)} rows (need >= {min_rows})")
            except Exception as e:
                failures.append(f"{filename} CSV validation error: {e}")
    
    return passes, failures


def validate_test_outputs(test_id: str, verbose: bool = True) -> bool:
    """
    Validate all requirements for a test.
    
    Returns:
        True if all requirements pass, False otherwise
    """
    test_dir = get_test_output_dir(test_id)
    if not test_dir:
        if verbose:
            print(f"❌ {test_id}: Output directory not found")
        return False
    
    # Core requirements
    core_passes, core_failures = check_core_requirements(test_dir, test_id)
    
    # Special requirements
    special_passes, special_failures = check_special_requirements(test_dir, test_id)
    
    all_passes = core_passes + special_passes
    all_failures = core_failures + special_failures
    
    if verbose:
        if all_failures:
            print(f"❌ {test_id}: {len(all_failures)} requirement(s) failed")
            for failure in all_failures:
                print(f"   ✗ {failure}")
        else:
            print(f"✅ {test_id}: All {len(all_passes)} requirements passed")
    
    return len(all_failures) == 0


# ============================================================================
# PYTEST INTEGRATION
# ============================================================================

def get_all_test_ids() -> List[str]:
    """Get all test IDs from results directory."""
    project_root = Path(__file__).parent
    results_dir = project_root / "results"
    
    test_ids = []
    for category_dir in results_dir.iterdir():
        if category_dir.is_dir() and category_dir.name in ["Relativistic", "Gravity", "Energy", "Quantization"]:
            for test_dir in category_dir.iterdir():
                if test_dir.is_dir():
                    test_ids.append(test_dir.name)
    
    return sorted(test_ids)


@pytest.mark.parametrize("test_id", get_all_test_ids())
def test_output_requirements(test_id):
    """Pytest: Validate output requirements for each test."""
    assert validate_test_outputs(test_id, verbose=False), f"{test_id} failed output requirements"


def test_tier1_outputs():
    """Pytest: Validate all Tier 1 (Relativistic) test outputs."""
    test_ids = [tid for tid in get_all_test_ids() if tid.startswith("REL-")]
    failures = []
    for test_id in test_ids:
        if not validate_test_outputs(test_id, verbose=False):
            failures.append(test_id)
    
    assert not failures, f"Tier 1 tests failed: {', '.join(failures)}"


def test_tier2_outputs():
    """Pytest: Validate all Tier 2 (Gravity) test outputs."""
    test_ids = [tid for tid in get_all_test_ids() if tid.startswith("GRAV-")]
    failures = []
    for test_id in test_ids:
        if not validate_test_outputs(test_id, verbose=False):
            failures.append(test_id)
    
    assert not failures, f"Tier 2 tests failed: {', '.join(failures)}"


def test_tier3_outputs():
    """Pytest: Validate all Tier 3 (Energy) test outputs."""
    test_ids = [tid for tid in get_all_test_ids() if tid.startswith("ENER-")]
    failures = []
    for test_id in test_ids:
        if not validate_test_outputs(test_id, verbose=False):
            failures.append(test_id)
    
    assert not failures, f"Tier 3 tests failed: {', '.join(failures)}"


def test_tier4_outputs():
    """Pytest: Validate all Tier 4 (Quantization) test outputs."""
    test_ids = [tid for tid in get_all_test_ids() if tid.startswith("QUAN-")]
    failures = []
    for test_id in test_ids:
        if not validate_test_outputs(test_id, verbose=False):
            failures.append(test_id)
    
    assert not failures, f"Tier 4 tests failed: {', '.join(failures)}"


def test_double_slit_interference_pattern():
    """Pytest: Validate double-slit test generates interference pattern image."""
    test_id = "GRAV-16"
    test_dir = get_test_output_dir(test_id)
    if not test_dir:
        pytest.skip(f"{test_id} not run yet")
    
    pattern_file = test_dir / "plots" / "interference_pattern.png"
    assert pattern_file.exists(), "Double-slit interference pattern image not found"
    
    # Check file size (should be non-trivial)
    size_kb = pattern_file.stat().st_size / 1024
    assert size_kb >= 10, f"Interference pattern too small: {size_kb:.1f} KB"


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate test output requirements across all tiers"
    )
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                       help="Check specific tier only")
    parser.add_argument("--test", type=str,
                       help="Check specific test ID only")
    parser.add_argument("--check-all", action="store_true",
                       help="Check all tests")
    args = parser.parse_args()
    
    test_ids = []
    
    if args.test:
        test_ids = [args.test]
    elif args.tier:
        prefix_map = {1: "REL", 2: "GRAV", 3: "ENER", 4: "QUAN"}
        prefix = prefix_map[args.tier]
        test_ids = [tid for tid in get_all_test_ids() if tid.startswith(prefix)]
    else:
        test_ids = get_all_test_ids()
    
    print(f"\n{'='*70}")
    print(f"TEST OUTPUT REQUIREMENTS VALIDATION")
    print(f"{'='*70}\n")
    
    total = len(test_ids)
    passed = 0
    failed = 0
    
    for test_id in test_ids:
        if validate_test_outputs(test_id, verbose=True):
            passed += 1
        else:
            failed += 1
        print()
    
    print(f"{'='*70}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed output requirements")
        exit(1)
    else:
        print(f"✅ All tests meet output requirements!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
