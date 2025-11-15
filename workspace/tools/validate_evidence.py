#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Validation Tool

Checks that test outputs contain all required world-class artifacts.
Run after test suite to ensure deterministic, complete evidence generation.

Usage:
    python tools/validate_evidence.py --test REL-01
    python tools/validate_evidence.py --tier 1
    python tools/validate_evidence.py --all
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.evidence_schema import get_schema_for_test, validate_test_evidence


def validate_test_directory(test_dir: Path, test_id: str) -> Dict:
    """Validate a single test's evidence."""
    return validate_test_evidence(test_dir, test_id)


def find_test_directories(results_root: Path, tier: int = None) -> List[Path]:
    """Find all test result directories."""
    test_dirs = []
    
    if tier is not None:
        tier_dirs = {
            1: results_root / "Relativistic",
            2: results_root / "GravityAnalogue",
            3: results_root / "Energy",
            4: results_root / "Quantization",
            5: results_root / "Electromagnetic",
            6: results_root / "Coupling",
            7: results_root / "Thermodynamics",
        }
        tier_dir = tier_dirs.get(tier)
        if tier_dir and tier_dir.exists():
            for test_dir in tier_dir.iterdir():
                if test_dir.is_dir() and (test_dir / "summary.json").exists():
                    test_dirs.append(test_dir)
    else:
        # All tiers
        for category_dir in results_root.iterdir():
            if category_dir.is_dir():
                for test_dir in category_dir.iterdir():
                    if test_dir.is_dir() and (test_dir / "summary.json").exists():
                        test_dirs.append(test_dir)
    
    return test_dirs


def print_validation_report(results: List[Dict]) -> None:
    """Print formatted validation report."""
    print("\n" + "="*80)
    print("EVIDENCE VALIDATION REPORT".center(80))
    print("="*80 + "\n")
    
    total = len(results)
    validated = sum(1 for r in results if r.get("schema_status") == "validated")
    minimal = sum(1 for r in results if r.get("schema_status") == "minimal")
    complete = sum(1 for r in results if r.get("complete", False))
    incomplete = total - complete
    
    print(f"Total Tests Validated: {total}")
    print(f"  Schema-validated: {validated}")
    print(f"  Minimal-validated: {minimal}")
    print(f"Complete Evidence: {complete} ({complete/total*100:.1f}%)")
    print(f"Incomplete Evidence: {incomplete} ({incomplete/total*100:.1f}%)")
    print()
    
    if incomplete > 0:
        print("\nINCOMPLETE TESTS:")
        print("-" * 80)
        for r in results:
            if not r.get("complete", False):
                test_id = r.get("test_id", "UNKNOWN")
                domain = r.get("domain", "unknown")
                status = r.get("schema_status", "unknown")
                missing = r.get("missing", [])
                
                print(f"\n{test_id} ({domain}, {status})")
                print(f"  Missing artifacts ({len(missing)}):")
                for artifact in missing:
                    print(f"    - {artifact}")
    
    print("\n" + "="*80)
    
    if incomplete > 0:
        print(f"\n❌ VALIDATION FAILED: {incomplete} tests missing required evidence")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS HAVE COMPLETE EVIDENCE")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Validate LFM test evidence artifacts")
    parser.add_argument("--test", type=str, help="Validate specific test ID (e.g., REL-01)")
    parser.add_argument("--tier", type=int, help="Validate entire tier (1-7)")
    parser.add_argument("--all", action="store_true", help="Validate all tests")
    parser.add_argument("--results-dir", type=str, 
                       default="c:/LFM/workspace/results",
                       help="Path to results directory")
    
    args = parser.parse_args()
    
    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        sys.exit(1)
    
    # Determine which tests to validate
    test_dirs = []
    
    if args.test:
        # Find specific test
        test_id = args.test.upper()
        for category_dir in results_root.iterdir():
            if category_dir.is_dir():
                test_dir = category_dir / test_id
                if test_dir.exists() and (test_dir / "summary.json").exists():
                    test_dirs.append(test_dir)
                    break
        if not test_dirs:
            print(f"❌ Test not found: {test_id}")
            sys.exit(1)
    
    elif args.tier:
        test_dirs = find_test_directories(results_root, tier=args.tier)
        if not test_dirs:
            print(f"❌ No tests found for Tier {args.tier}")
            sys.exit(1)
    
    elif args.all:
        test_dirs = find_test_directories(results_root)
        if not test_dirs:
            print(f"❌ No tests found in {results_root}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Validate each test
    validation_results = []
    for test_dir in test_dirs:
        test_id = test_dir.name
        result = validate_test_directory(test_dir, test_id)
        validation_results.append(result)
    
    # Print report
    print_validation_report(validation_results)


if __name__ == "__main__":
    main()
