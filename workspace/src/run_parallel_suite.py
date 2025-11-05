#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Run Parallel Test Suite - Main CLI for adaptive parallel test execution
======================================================================
Intelligently schedules and runs LFM test suites in parallel based on
resource availability and historical metrics.

Usage:
    python run_parallel_suite.py --tiers 1,2         # Run Tier 1 and 2
    python run_parallel_suite.py --fast              # Run fast test subset
    python run_parallel_suite.py --tests REL-01,REL-02,GRAV-01  # Specific tests
    python run_parallel_suite.py --max-concurrent 4  # Control parallelism
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from harness.lfm_test_metrics import TestMetrics, load_test_configs
from adaptive_scheduler import AdaptiveScheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive parallel test suite runner for LFM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help="Comma-separated tier numbers (e.g., '1,2,3'). If omitted, uses --fast or --tests."
    )
    
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated test IDs (e.g., 'REL-01,GRAV-12')"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast test subset for validation (REL-01, REL-02, GRAV-12, GRAV-23)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of tests to run concurrently (default: 4)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress messages (default: True)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    return parser.parse_args()


def build_test_list_from_tiers(tiers: List[int]) -> List[Tuple[str, int, Dict]]:
    """
    Build test list from tier numbers.
    
    Args:
        tiers: List of tier numbers (1-4)
    
    Returns:
        List of (test_id, tier, config) tuples
    """
    test_list = []
    
    for tier in tiers:
        configs = load_test_configs(tier)
        for test_id, config in configs:
            test_list.append((test_id, tier, config))
    
    return test_list


def build_test_list_from_ids(test_ids: List[str]) -> List[Tuple[str, int, Dict]]:
    """
    Build test list from specific test IDs.
    
    Args:
        test_ids: List of test IDs (e.g., ["REL-01", "GRAV-12"])
    
    Returns:
        List of (test_id, tier, config) tuples
    """
    # Map test prefixes to tier numbers
    prefix_to_tier = {
        "REL": 1,
        "GRAV": 2,
        "ENER": 3,
        "QUAN": 4,
        "UNIF": 0
    }
    
    test_list = []
    
    for test_id in test_ids:
        # Determine tier from prefix
        prefix = test_id.split("-")[0]
        tier = prefix_to_tier.get(prefix)
        
        if tier is None:
            print(f"Warning: Unknown test ID format: {test_id}")
            continue
        
        # Load config for this tier
        configs = load_test_configs(tier)
        config_dict = dict(configs)
        
        if test_id in config_dict:
            test_list.append((test_id, tier, config_dict[test_id]))
        else:
            # Create minimal config
            print(f"Warning: No config found for {test_id}, using defaults")
            test_list.append((test_id, tier, {
                "grid_points": 512,
                "steps": 6000,
                "dimensions": 1,
                "use_gpu": True
            }))
    
    return test_list


def get_fast_test_list() -> List[Tuple[str, int, Dict]]:
    """Get fast test subset for quick validation."""
    # These tests are known to be fast (2-5 seconds each)
    fast_tests = [
        ("REL-01", 1, {"dimensions": 1, "grid_points": 512, "steps": 6000, "use_gpu": True}),
        ("REL-02", 1, {"dimensions": 1, "grid_points": 512, "steps": 6000, "use_gpu": True}),
        ("GRAV-12", 2, {"dimensions": 1, "grid_points": 64, "steps": 4800, "use_gpu": True}),
        ("GRAV-23", 2, {"dimensions": 1, "grid_points": 64, "steps": 600, "use_gpu": True}),
    ]
    return fast_tests


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine test list
    test_list = []
    
    if args.fast:
        print("=== FAST TEST MODE ===")
        print("Running 4 fast tests for validation\n")
        test_list = get_fast_test_list()
    
    elif args.tests:
        test_ids = [t.strip() for t in args.tests.split(",")]
        print(f"=== SPECIFIC TESTS MODE ===")
        print(f"Running {len(test_ids)} specified tests\n")
        test_list = build_test_list_from_ids(test_ids)
    
    elif args.tiers:
        tiers = [int(t.strip()) for t in args.tiers.split(",")]
        print(f"=== TIER MODE ===")
        print(f"Running tests from tier(s): {tiers}\n")
        test_list = build_test_list_from_tiers(tiers)
    
    else:
        print("Error: Must specify --tiers, --tests, or --fast")
        sys.exit(1)
    
    if not test_list:
        print("Error: No tests to run")
        sys.exit(1)
    
    print(f"Total tests queued: {len(test_list)}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Verbose: {not args.quiet}\n")
    
    # Create scheduler
    verbose = not args.quiet if args.quiet else args.verbose
    scheduler = AdaptiveScheduler(verbose=verbose)
    
    # Run tests
    results = scheduler.schedule_tests(test_list, max_concurrent=args.max_concurrent)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total runtime: {results['total_runtime_sec']:.1f}s ({results['total_runtime_sec']/60:.1f} min)")
    print(f"Tests passed: {results['completed']}")
    print(f"Tests failed: {results['failed']}")
    
    if results['failed'] > 0:
        print("\nFailed tests:")
        for ft in results['failed_tests']:
            print(f"  ✗ {ft['test_id']}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
