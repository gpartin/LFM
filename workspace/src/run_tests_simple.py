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
Simple sequential test runner with visible output - for debugging.
Works perfectly but runs sequentially. Use this until we fix parallel threading issues.
"""

import subprocess
import sys
import time
from pathlib import Path
from test_metrics import TestMetrics

def run_test(test_id, tier, test_metrics):
    """Run a single test with visible output and metric recording."""
    
    # Map tier to runner script
    runners = {
        1: "run_tier1_relativistic.py",
        2: "run_tier2_gravityanalogue.py",
        3: "run_tier3_energy.py",
        4: "run_tier4_quantization.py"
    }
    
    runner = runners.get(tier)
    if not runner:
        print(f"ERROR: Unknown tier {tier}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Running {test_id} (Tier {tier})")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    
    # Run with -u for unbuffered output
    cmd = ["python", "-u", runner, "--test", test_id]
    
    start_time = time.time()
    
    # Run with inherited stdout/stderr - should show output immediately
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        text=True
    )
    
    runtime = time.time() - start_time
    
    # Record metrics
    metrics = {
        "exit_code": result.returncode,
        "runtime_sec": runtime,
        "peak_cpu_percent": 0.0,  # Not monitoring for now
        "peak_memory_mb": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    test_metrics.record_run(test_id, metrics)
    
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n{test_id}: {status} ({runtime:.1f}s)\n")
    sys.stdout.flush()
    
    return result.returncode == 0


def main():
    """Run fast tests."""
    
    tests = [
        ("REL-01", 1),
        ("REL-02", 1),
        ("GRAV-12", 2),
        ("GRAV-23", 2),
    ]
    
    print("="*70)
    print("SIMPLE TEST RUNNER - SEQUENTIAL MODE")
    print("="*70)
    print(f"Running {len(tests)} tests...")
    
    # Load test metrics
    test_metrics = TestMetrics()
    
    start_time = time.time()
    passed = 0
    failed = 0
    
    for test_id, tier in tests:
        if run_test(test_id, tier, test_metrics):
            passed += 1
        else:
            failed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_time:.1f}s")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
