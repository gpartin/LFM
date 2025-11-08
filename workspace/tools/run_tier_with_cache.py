#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Cached Tier Test Runner

Wraps tier test harnesses with caching to skip tests when dependencies haven't changed.

USAGE:
    python tools/run_tier_with_cache.py --tier 5               # Run all EM tests with caching
    python tools/run_tier_with_cache.py --tier 5 --test EM-01  # Run single test with caching
    python tools/run_tier_with_cache.py --tier 5 --no-cache    # Force re-run all tests
    python tools/run_tier_with_cache.py --clear-cache          # Clear entire cache
    python tools/run_tier_with_cache.py --cache-stats          # Show cache statistics
"""

import argparse
import shutil
import sys
from pathlib import Path

# Bootstrap path
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))

from utils.path_utils import get_workspace_dir, add_workspace_to_sys_path

WS = get_workspace_dir(__file__)
# Ensure workspace and src are on sys.path
add_workspace_to_sys_path(__file__)
SRC_DIR = WS / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
BUILD_DIR = WS.parent / 'build'
CACHE_ROOT = BUILD_DIR / 'cache' / 'test_results'

# Import cache manager from src/utils
from utils.cache_manager_runtime import TestCacheManager

# Import tier harnesses
from run_tier5_electromagnetic import Tier5ElectromagneticHarness


TIER_MAP = {
    5: {
        'config': 'config_tier5_electromagnetic.json',
        'harness_class': Tier5ElectromagneticHarness,
        'results_subdir': 'Electromagnetic'
    }
    # Add other tiers as needed:
    # 1: {'config': 'config_tier1_relativistic.json', 'harness_class': Tier1Harness, ...},
    # 2: {'config': 'config_tier2_gravityanalogue.json', 'harness_class': Tier2Harness, ...},
}


def run_test_with_cache(
    test_config: dict,
    harness,
    cache_manager: TestCacheManager,
    config_file: Path,
    results_dir: Path,
    use_cache: bool = True
):
    """
    Run a single test with caching support.
    
    Args:
        test_config: Test configuration dict from config JSON
        harness: Tier harness instance
        cache_manager: Cache manager instance
        config_file: Path to tier config file
        results_dir: Results directory for this tier
        use_cache: If False, always re-run test
    
    Returns:
        Test result dict
    """
    test_id = test_config['id']
    
    # Check cache validity
    if use_cache and cache_manager.is_cache_valid(test_id, config_file):
        print(f"  ✓ {test_id}: Using cached results (dependencies unchanged)")
        
        # Copy cached results to output directory
        cached_results = cache_manager.get_cached_results(test_id)
        output_test_dir = results_dir / test_id
        
        if output_test_dir.exists():
            shutil.rmtree(output_test_dir)
        
        shutil.copytree(cached_results, output_test_dir)
        
        # Read cached summary
        summary_file = output_test_dir / 'summary.json'
        if summary_file.exists():
            import json
            summary = json.loads(summary_file.read_text(encoding='utf-8'))
            return {
                'test_id': test_id,
                'passed': summary.get('passed', False),
                'cached': True
            }
        
        return {'test_id': test_id, 'passed': True, 'cached': True}
    
    # Cache miss - run the test
    cache_status = "cache miss" if use_cache else "caching disabled"
    print(f"  ⚙ {test_id}: Running test ({cache_status})")
    
    # Run test through harness
    result = harness.run_test(test_config)
    
    # Store results in cache
    if use_cache:
        output_test_dir = results_dir / test_id
        if output_test_dir.exists():
            cache_manager.store_test_results(
                test_id=test_id,
                results_dir=output_test_dir,
                config_file=config_file,
                metadata={
                    'passed': result.passed,
                    'description': test_config.get('description', '')
                }
            )
    
    return {
        'test_id': test_id,
        'passed': result.passed,
        'cached': False
    }


def main():
    parser = argparse.ArgumentParser(description='Run tier tests with caching')
    parser.add_argument('--tier', type=int, help='Tier number (e.g., 5 for EM tests)')
    parser.add_argument('--test', help='Run specific test (e.g., EM-01)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache (force re-run)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear entire cache')
    parser.add_argument('--cache-stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    args = parser.parse_args()
    
    # Initialize cache manager
    cache_manager = TestCacheManager(CACHE_ROOT, WS)
    
    # Handle utility commands
    if args.clear_cache:
        cache_manager.invalidate_all()
        print("✓ Cache cleared")
        return
    
    if args.cache_stats:
        stats = cache_manager.get_cache_stats()
        print("Test Cache Statistics:")
        print(f"  Cached tests: {stats['total_cached_tests']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
        if stats['cached_tests']:
            print(f"  Tests: {', '.join(stats['cached_tests'])}")
        return
    
    # Tier is required for actual test execution
    if not args.tier:
        parser.error("--tier is required (or use --cache-stats, --clear-cache)")
    
    if args.tier not in TIER_MAP:
        print(f"Error: Tier {args.tier} not yet supported in cached runner")
        print(f"Supported tiers: {list(TIER_MAP.keys())}")
        return 1
    
    tier_info = TIER_MAP[args.tier]
    
    # Load config
    config_path = WS / 'config' / tier_info['config']
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1
    
    # Initialize harness
    harness_class = tier_info['harness_class']
    harness = harness_class(str(config_path))
    
    # Override quick/gpu settings if specified
    if args.quick:
        harness.quick = True
    if args.gpu:
        from core.lfm_backend import pick_backend
        harness.xp, harness.use_gpu = pick_backend(True)
    
    results_dir = WS / 'results' / tier_info['results_subdir']
    use_cache = not args.no_cache
    
    # Get tests to run
    tests_to_run = []
    for test_config in harness.config.get('tests', []):
        if not test_config.get('enabled', True):
            continue
        
        if args.test:
            if test_config['id'] == args.test:
                tests_to_run.append(test_config)
                break
        else:
            tests_to_run.append(test_config)
    
    if not tests_to_run:
        if args.test:
            print(f"Error: Test {args.test} not found or not enabled")
        else:
            print("No enabled tests found in config")
        return 1
    
    # Run tests
    print(f"\n{'='*70}")
    print(f"TIER {args.tier} TEST RUNNER (with caching)")
    print(f"{'='*70}")
    print(f"Tests to run: {len(tests_to_run)}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print()
    
    results = []
    for test_config in tests_to_run:
        result = run_test_with_cache(
            test_config=test_config,
            harness=harness,
            cache_manager=cache_manager,
            config_file=config_path,
            results_dir=results_dir,
            use_cache=use_cache
        )
        results.append(result)
    
    # Summary
    print()
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in results if r['passed'])
    cached = sum(1 for r in results if r.get('cached', False))
    
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"From cache: {cached}")
    print(f"Executed: {len(results) - cached}")
    
    # Show cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\nCache: {stats['total_cached_tests']} tests, {stats['cache_size_mb']:.1f} MB")
    
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
