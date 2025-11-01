#!/usr/bin/env python3
"""
validate_resource_tracking.py — Multi-Point Inspection for Resource Metrics
--------------------------------------------------------------------------
Purpose:
    Validate that all 4 tier runners correctly populate resource metrics
    (CPU, RAM, GPU, runtime) after test execution.

Inspection Points:
    1. Test metrics history JSON (test_metrics_history.json)
    2. Individual test summary.json files
    3. Master test status CSV (MASTER_TEST_STATUS.csv)
    
Expected Behavior:
    - peak_cpu_percent > 0 (unless psutil unavailable)
    - peak_memory_mb > 0 (unless psutil unavailable)
    - peak_gpu_memory_mb >= 0 (0 if no GPU or nvidia-smi unavailable)
    - runtime_sec > 0
    - All metrics present (no missing keys)

Usage:
    python validate_resource_tracking.py [--tier TIER] [--test TEST_ID]
    
    Examples:
        python validate_resource_tracking.py --tier 1
        python validate_resource_tracking.py --test REL-01
        python validate_resource_tracking.py  # Check all
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_metrics_history(project_root: Path) -> Dict:
    """Load test metrics history JSON."""
    metrics_file = project_root / "results" / "test_metrics_history.json"
    if not metrics_file.exists():
        print(f"{Colors.WARNING}⚠ Metrics history not found: {metrics_file}{Colors.ENDC}")
        return {}
    
    with open(metrics_file, 'r') as f:
        return json.load(f)


def load_test_summary(test_dir: Path) -> Optional[Dict]:
    """Load individual test summary.json."""
    summary_file = test_dir / "summary.json"
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def validate_metrics(metrics: Dict, test_id: str, source: str) -> List[str]:
    """
    Validate resource metrics for a single test.
    
    Returns:
        List of validation errors (empty if all checks pass)
    """
    errors = []
    
    # Check required keys
    required_keys = ["peak_cpu_percent", "peak_memory_mb", "peak_gpu_memory_mb", "runtime_sec"]
    for key in required_keys:
        if key not in metrics:
            errors.append(f"Missing key: {key}")
    
    if errors:
        return errors  # Can't proceed without keys
    
    # Extract values
    cpu = metrics["peak_cpu_percent"]
    mem = metrics["peak_memory_mb"]
    gpu = metrics["peak_gpu_memory_mb"]
    runtime = metrics["runtime_sec"]
    
    # Validate runtime (must be > 0)
    if runtime <= 0:
        errors.append(f"runtime_sec is {runtime} (should be > 0)")
    
    # Validate CPU (should be > 0 unless psutil unavailable)
    if cpu == 0.0:
        errors.append(f"peak_cpu_percent is 0.0 (possible psutil unavailable or test too fast)")
    elif cpu < 0:
        errors.append(f"peak_cpu_percent is negative: {cpu}")
    
    # Validate memory (should be > 0 unless psutil unavailable)
    if mem == 0.0:
        errors.append(f"peak_memory_mb is 0.0 (possible psutil unavailable or test too fast)")
    elif mem < 0:
        errors.append(f"peak_memory_mb is negative: {mem}")
    
    # Validate GPU (>= 0, can be 0 if no GPU)
    if gpu < 0:
        errors.append(f"peak_gpu_memory_mb is negative: {gpu}")
    
    # Check for placeholder values (exactly 0.0 for all)
    if cpu == 0.0 and mem == 0.0 and gpu == 0.0 and runtime > 0:
        errors.append("PLACEHOLDER VALUES DETECTED (all zeros except runtime)")
    
    return errors


def get_tier_prefix(tier: int) -> str:
    """Get test ID prefix for tier."""
    prefixes = {1: "REL", 2: "GRAV", 3: "ENER", 4: "QUAN"}
    return prefixes.get(tier, "")


def find_tier_results_dirs(project_root: Path, tier: Optional[int] = None) -> List[Path]:
    """Find all tier result directories."""
    results_dir = project_root / "results"
    if not results_dir.exists():
        return []
    
    tier_dirs = []
    tier_names = {
        1: "Relativistic",
        2: "Gravity",
        3: "Energy",
        4: "Quantization"
    }
    
    if tier:
        # Single tier
        tier_name = tier_names.get(tier)
        if tier_name:
            tier_dir = results_dir / tier_name
            if tier_dir.exists():
                tier_dirs.append(tier_dir)
    else:
        # All tiers
        for tier_name in tier_names.values():
            tier_dir = results_dir / tier_name
            if tier_dir.exists():
                tier_dirs.append(tier_dir)
    
    return tier_dirs


def inspect_test_from_history(metrics_history: Dict, test_id: str) -> Dict:
    """Inspect test from metrics history."""
    if test_id not in metrics_history:
        return {
            "test_id": test_id,
            "found": False,
            "source": "metrics_history",
            "error": "Test not found in metrics history"
        }
    
    # Get the most recent run from the runs array
    runs = metrics_history[test_id].get("runs", [])
    if not runs:
        return {
            "test_id": test_id,
            "found": False,
            "source": "metrics_history",
            "error": "No runs found for test"
        }
    
    last_run = runs[-1]
    errors = validate_metrics(last_run, test_id, "metrics_history")
    
    return {
        "test_id": test_id,
        "found": True,
        "source": "metrics_history",
        "metrics": last_run,
        "errors": errors,
        "passed": len(errors) == 0
    }


def inspect_test_from_summary(test_dir: Path, test_id: str) -> Dict:
    """Inspect test from summary.json."""
    summary = load_test_summary(test_dir)
    if not summary:
        return {
            "test_id": test_id,
            "found": False,
            "source": "summary.json",
            "error": "summary.json not found"
        }
    
    # Summary might have metrics in different structure
    metrics = {}
    if "metrics" in summary and isinstance(summary["metrics"], dict):
        metrics = {
            "peak_cpu_percent": summary["metrics"].get("peak_cpu_percent", 0.0),
            "peak_memory_mb": summary["metrics"].get("peak_memory_mb", 0.0),
            "peak_gpu_memory_mb": summary["metrics"].get("peak_gpu_memory_mb", 0.0),
            "runtime_sec": summary.get("runtime_sec", 0.0)
        }
    else:
        # Some summaries may not have resources yet
        return {
            "test_id": test_id,
            "found": True,
            "source": "summary.json",
            "error": "No resource metrics in summary (may need update)",
            "passed": False
        }
    
    errors = validate_metrics(metrics, test_id, "summary.json")
    
    return {
        "test_id": test_id,
        "found": True,
        "source": "summary.json",
        "metrics": metrics,
        "errors": errors,
        "passed": len(errors) == 0
    }


def print_test_inspection(inspection: Dict):
    """Print inspection results for a single test."""
    test_id = inspection["test_id"]
    
    if not inspection["found"]:
        print(f"  {Colors.WARNING}⚠ {test_id}: {inspection.get('error', 'Not found')}{Colors.ENDC}")
        return
    
    if "error" in inspection and "passed" in inspection and not inspection["passed"]:
        print(f"  {Colors.FAIL}✗ {test_id}: {inspection['error']}{Colors.ENDC}")
        return
    
    if inspection["passed"]:
        metrics = inspection["metrics"]
        print(f"  {Colors.OKGREEN}✓ {test_id}{Colors.ENDC}: "
              f"CPU={metrics['peak_cpu_percent']:.1f}%, "
              f"RAM={metrics['peak_memory_mb']:.1f}MB, "
              f"GPU={metrics['peak_gpu_memory_mb']:.1f}MB, "
              f"Runtime={metrics['runtime_sec']:.2f}s")
    else:
        print(f"  {Colors.FAIL}✗ {test_id}{Colors.ENDC}:")
        for error in inspection["errors"]:
            print(f"    - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate resource tracking metrics across all tier runners"
    )
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4],
                       help="Check specific tier only")
    parser.add_argument("--test", type=str,
                       help="Check specific test ID only (e.g., REL-01)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== Resource Tracking Validation ==={Colors.ENDC}\n")
    
    # Load metrics history
    metrics_history = load_metrics_history(project_root)
    print(f"Loaded metrics history: {len(metrics_history)} tests\n")
    
    # Single test mode
    if args.test:
        print(f"{Colors.OKBLUE}Inspecting test: {args.test}{Colors.ENDC}\n")
        
        # Check metrics history
        print(f"1. Metrics History (test_metrics_history.json):")
        history_inspection = inspect_test_from_history(metrics_history, args.test)
        print_test_inspection(history_inspection)
        
        # Find test directory
        tier_dirs = find_tier_results_dirs(project_root, args.tier)
        found_summary = False
        for tier_dir in tier_dirs:
            for test_dir in tier_dir.iterdir():
                if test_dir.is_dir() and test_dir.name == args.test:
                    print(f"\n2. Test Summary ({test_dir.name}/summary.json):")
                    summary_inspection = inspect_test_from_summary(test_dir, args.test)
                    print_test_inspection(summary_inspection)
                    found_summary = True
                    break
        
        if not found_summary:
            print(f"\n2. Test Summary: {Colors.WARNING}Not found{Colors.ENDC}")
        
        return
    
    # Multi-tier mode
    tier_dirs = find_tier_results_dirs(project_root, args.tier)
    
    if not tier_dirs:
        print(f"{Colors.FAIL}No tier directories found{Colors.ENDC}")
        return
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_missing = 0
    
    for tier_dir in tier_dirs:
        tier_name = tier_dir.name
        print(f"{Colors.OKBLUE}{Colors.BOLD}Tier: {tier_name}{Colors.ENDC}")
        
        test_dirs = [d for d in tier_dir.iterdir() if d.is_dir()]
        
        if not test_dirs:
            print(f"  {Colors.WARNING}No test directories found{Colors.ENDC}\n")
            continue
        
        for test_dir in sorted(test_dirs):
            test_id = test_dir.name
            total_tests += 1
            
            # Inspect from metrics history
            inspection = inspect_test_from_history(metrics_history, test_id)
            
            if not inspection["found"]:
                total_missing += 1
            elif inspection["passed"]:
                total_passed += 1
            else:
                total_failed += 1
            
            print_test_inspection(inspection)
        
        print()  # Blank line between tiers
    
    # Summary
    print(f"{Colors.HEADER}{Colors.BOLD}=== Summary ==={Colors.ENDC}")
    print(f"Total tests inspected: {total_tests}")
    print(f"{Colors.OKGREEN}✓ Passed: {total_passed}{Colors.ENDC}")
    print(f"{Colors.FAIL}✗ Failed: {total_failed}{Colors.ENDC}")
    print(f"{Colors.WARNING}⚠ Missing: {total_missing}{Colors.ENDC}")
    
    if total_failed > 0 or total_missing > 0:
        print(f"\n{Colors.FAIL}Validation FAILED - some tests have invalid or missing metrics{Colors.ENDC}")
        exit(1)
    else:
        print(f"\n{Colors.OKGREEN}✓ All tests have valid resource metrics!{Colors.ENDC}")


if __name__ == "__main__":
    main()
