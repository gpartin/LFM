#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
StandardTierTemplate.py — Base template for creating new LFM test tiers

This template provides the standardized patterns used across all LFM tiers:
- Consistent argument parsing with post-validation hooks
- Standardized console output formatting
- Common configuration loading patterns
- Master test status integration
- Metrics tracking integration

Usage:
1. Copy this template to run_tierN_description.py
2. Replace TierNHarness class with tier-specific implementation
3. Implement tier-specific test functions
4. Update configuration patterns
5. Done!

This should reduce new tier creation time from 8+ hours to ~2 hours.
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# Standard LFM imports - available across all tiers
from ui.lfm_console import log
from utils.lfm_config import LFMConfig


@dataclass 
class StandardTestResult:
    """Standardized test result format across all tiers"""
    test_id: str
    description: str
    passed: bool
    metrics: Dict[str, Any]
    runtime_sec: float


def create_standard_tier_parser(description: str) -> argparse.ArgumentParser:
    """Create standardized argument parser used across all tiers"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--test", type=str, default=None,
                       help="Run single test by ID (e.g., TEST-01). If omitted, runs all tests.")
    parser.add_argument("--config", type=str, 
                       help="Path to config file")
    # Post-run validation hooks
    parser.add_argument('--post-validate', choices=['tier', 'all'], default=None,
                        help='Run validator after the suite: "tier" validates this tier + master status; "all" runs end-to-end')
    parser.add_argument('--strict-validate', action='store_true',
                        help='In strict mode, warnings cause validation to fail')
    parser.add_argument('--quiet-validate', action='store_true',
                        help='Reduce validator verbosity') 
    parser.add_argument('--update-upload', action='store_true',
                        help='Rebuild docs/upload package (refresh status, stage docs, comprehensive PDF, manifest)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode for upload build (fixed timestamps, reproducible zip)')
    return parser


class StandardTierHarness:
    """Base harness providing common tier functionality"""
    
    def __init__(self, config_path: str, tier_name: str, tier_number: int):
        self.config_path = Path(config_path)
        self.tier_name = tier_name
        self.tier_number = tier_number
        self.config = self.load_config()
        self.output_dir = Path(f"results/{tier_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self) -> Dict:
        """Load tier configuration with error handling"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            log(f"[ERROR] Config file not found: {self.config_path}", "FAIL")
            exit(1)
        except json.JSONDecodeError as e:
            log(f"[ERROR] Invalid JSON in config: {e}", "FAIL")
            exit(1)
    
    def get_enabled_tests(self) -> List[Dict]:
        """Get list of enabled tests from configuration"""
        return [test for test in self.config.get("tests", []) 
                if test.get("enabled", True)]
    
    def run_single_test(self, test_config: Dict) -> StandardTestResult:
        """Override this method in tier-specific implementation"""
        raise NotImplementedError("Implement run_single_test in tier-specific harness")
    
    def run_all_tests(self) -> List[StandardTestResult]:
        """Run all enabled tests using standard pattern"""
        enabled_tests = self.get_enabled_tests()
        log(f"Running {len(enabled_tests)} enabled tests...", "INFO")
        
        results = []
        for test_config in enabled_tests:
            test_id = test_config.get("id", "UNKNOWN")
            test_name = test_config.get("name", "Unknown Test")
            
            log(f"Running {test_id}: {test_name}", "INFO")
            
            try:
                result = self.run_single_test(test_config)
                results.append(result)
                
                # Standard result logging
                status = "PASSED" if result.passed else "FAILED"
                log_level = "INFO" if result.passed else "FAIL"
                log(f"  {result.test_id}: {status} ({result.runtime_sec:.2f}s)", log_level)
                
            except Exception as e:
                log(f"  {test_id}: ERROR - {str(e)}", "FAIL")
                # Create error result
                error_result = StandardTestResult(
                    test_id=test_id,
                    description=test_name,
                    passed=False,
                    metrics={"error": str(e)},
                    runtime_sec=0.0
                )
                results.append(error_result)
        
        return results
    
    def print_summary(self, results: List[StandardTestResult]):
        """Print standardized tier summary"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        
        log("="*60, "INFO")
        log(f"TIER {self.tier_number} SUMMARY", "INFO")
        log("="*60, "INFO")
        log(f"Total tests: {total_tests}", "INFO")
        log(f"Passed: {passed_tests}", "INFO")
        log(f"Failed: {total_tests - passed_tests}", "INFO")
        log(f"Success rate: {passed_tests/total_tests*100:.1f}%", "INFO")
    
    def update_master_status(self):
        """Update master test status with error handling"""
        try:
            from harness.lfm_test_harness import update_master_test_status
            update_master_test_status(Path("results"))
            log("Updated master test status", "INFO")
        except Exception as e:
            log(f"Warning: Could not update master status: {e}", "WARN")


class TierNHarness(StandardTierHarness):
    """
    TEMPLATE: Replace this with your tier-specific harness
    Example: TierQuantumHarness, TierGravityHarness, etc.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path, "TierN", N)  # Replace N with tier number
        # Add tier-specific initialization here
    
    def run_single_test(self, test_config: Dict) -> StandardTestResult:
        """
        TEMPLATE: Implement tier-specific test execution
        
        This method should:
        1. Extract test parameters from test_config
        2. Execute the tier-specific test logic
        3. Collect metrics and determine pass/fail
        4. Return StandardTestResult
        """
        start_time = time.time()
        test_id = test_config.get("id", "UNKNOWN")
        
        # EXAMPLE implementation - replace with actual test logic:
        try:
            # Your tier-specific test implementation goes here
            # Examples:
            # - Quantum coherence tests
            # - Gravitational wave tests  
            # - Energy conservation tests
            # - etc.
            
            # Placeholder logic:
            test_parameter = test_config.get("test_parameter", 1.0)
            # ... actual test implementation ...
            
            # Determine pass/fail based on your criteria
            passed = True  # Replace with actual logic
            
            # Collect tier-specific metrics
            metrics = {
                "example_metric": 0.001,
                "test_parameter": test_parameter,
                # Add your tier-specific metrics
            }
            
            runtime = time.time() - start_time
            
            return StandardTestResult(
                test_id=test_id,
                description=test_config.get("name", "Test"),
                passed=passed,
                metrics=metrics,
                runtime_sec=runtime
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return StandardTestResult(
                test_id=test_id,
                description=test_config.get("name", "Test"),
                passed=False,
                metrics={"error": str(e)},
                runtime_sec=runtime
            )


def main():
    """Standardized main function - copy this pattern exactly"""
    # Standard argument parsing
    parser = create_standard_tier_parser("Tier N Test Suite")  # Update description
    # Update the default config path in the standard parser instead of adding duplicate
    for action in parser._actions:
        if action.dest == 'config':
            action.default = "config/config_tierN.json"  # Update this path
    args = parser.parse_args()
    
    # Create tier harness
    harness = TierNHarness(args.config)  # Replace with your harness class
    
    # Standard tier startup message
    log("=== LFM TIER N: DESCRIPTION ===", "INFO")  # Update tier name and description
    
    # Handle single test vs full suite
    if args.test:
        # Single test execution
        enabled_tests = harness.get_enabled_tests()
        test_config = None
        for test in enabled_tests:
            if test.get("id") == args.test:
                test_config = test
                break
        
        if test_config:
            log(f"=== Running Single Test: {args.test} ===", "INFO")
            result = harness.run_single_test(test_config)
            
            status = "PASSED" if result.passed else "FAILED"
            log_level = "INFO" if result.passed else "FAIL"
            log(f"  {result.test_id}: {status} ({result.runtime_sec:.2f}s)", log_level)
            
            exit_code = 0 if result.passed else 1
            exit(exit_code)
        else:
            log(f"[ERROR] Test '{args.test}' not found in config", "FAIL")
            exit(1)
    else:
        # Full suite execution
        enabled_tests = harness.get_enabled_tests()
        log(f"=== Tier-N Test Suite Start (running {len(enabled_tests)} tests) ===", "INFO")  # Update tier number
        
        results = harness.run_all_tests()
        harness.print_summary(results)
        harness.update_master_status()
        
        log("Tier N tests completed!", "INFO")  # Update tier number


if __name__ == "__main__":
    main()


# ============================================================================
# TIER CREATION CHECKLIST:
# ============================================================================
# 
# To create a new tier using this template:
# 
# 1. COPY & RENAME:
#    - Copy this file to run_tierN_description.py
#    - Replace all "TierN" with your tier name (e.g., "Tier6Quantum")
#    - Replace all "N" with your tier number
# 
# 2. IMPLEMENT HARNESS:
#    - Rename TierNHarness to your tier name
#    - Implement run_single_test() with your test logic
#    - Add tier-specific initialization if needed
# 
# 3. CREATE CONFIG:
#    - Create config/config_tierN_description.json
#    - Define your test cases with standard format:
#      {
#        "tests": [
#          {
#            "id": "TEST-01",
#            "name": "Test Description",
#            "enabled": true,
#            "config": {
#              "test_parameter": 1.0
#            }
#          }
#        ]
#      }
# 
# 4. UPDATE MAIN:
#    - Update tier number and description in log messages
#    - Update default config path
# 
# 5. TEST:
#    - Run single test: python run_tierN.py --test TEST-01
#    - Run full suite: python run_tierN.py
# 
# 6. INTEGRATION:
#    - Add to main test runner
#    - Update documentation
#    - Add to CI/CD pipeline
# 
# ESTIMATED TIME: 2 hours for basic tier (vs 8+ hours without template)
#
# ============================================================================