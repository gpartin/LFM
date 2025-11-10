#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Pre-Commit Validation Script
============================
Runs one fast test from each tier to validate that all test infrastructure
is working before committing changes.

Usage:
    cd c:\\LFM\\workspace\\src
    python ..\\tools\\pre_commit_validation.py
    
Exit codes:
    0 - All tests passed
    1 - One or more tests failed
    2 - Error running tests
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Default fast tests fallback (used if telemetry unavailable)
DEFAULT_FAST_TESTS = [
    ("REL-01", 1, 30),    # Tier 1: Relativistic - Isotropy coarse
    ("GRAV-12", 2, 30),   # Tier 2: Gravity - Frequency ratio test
    ("ENER-01", 3, 30),   # Tier 3: Energy - Conservation flat space
    ("QUAN-01", 4, 45),   # Tier 4: Quantization - Bound state detection
    ("EM-01", 5, 30),     # Tier 5: Electromagnetic - Gauss's Law
    ("COUP-02", 6, 60),   # Tier 6: Coupling - L2 convergence validation
]

# Tests that require optional modules or are unsuitable for pre-commit
EXCLUDED_TESTS = {
    # Tier 2 dynamic chi/gravity wave/self-consistency require chi_field_equation
    "GRAV-20", "GRAV-23", "GRAV-24",
}

# Map test prefix to tier
PREFIX_TO_TIER = {"REL": 1, "GRAV": 2, "ENER": 3, "QUAN": 4, "EM": 5, "COUP": 6}

# How many fastest tests per tier to include when telemetry is available
TOP_K_PER_TIER = 3


def _load_metrics_history() -> dict:
    """Load telemetry metrics from harness/results/test_metrics_history.json if present."""
    metrics_path = Path(__file__).parent.parent / "src" / "harness" / "results" / "test_metrics_history.json"
    if not metrics_path.exists():
        return {}
    try:
        import json
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _pick_fastest_per_tier(metrics: dict) -> List[Tuple[str, int, int]]:
    """Select the fastest passing test per tier from telemetry.

    Returns list of (test_id, tier, timeout_sec)
    """
    from math import ceil

    per_tier = {}

    for test_id, info in metrics.items():
        # Skip excluded or malformed entries
        if test_id in EXCLUDED_TESTS:
            continue
        prefix = test_id.split("-")[0]
        tier = PREFIX_TO_TIER.get(prefix)
        if not tier:
            continue

        runs = info.get("runs", [])
        if not runs:
            continue

        # Use last successful run only
        last_successful = None
        for r in reversed(runs):
            if r.get("exit_code", 1) == 0:
                last_successful = r
                break
        if not last_successful:
            continue

        est = info.get("estimated_resources", {})
        runtime = est.get("runtime_sec", last_successful.get("runtime_sec", 30))
        timeout_mult = est.get("timeout_multiplier", 3.0)
        # Compute timeout with headroom
        timeout_sec = int(max(30, min(180, ceil(runtime * timeout_mult + 10))))

        # Keep the currently fastest per tier
        existing = per_tier.get(tier)
        if not existing or timeout_sec < existing[2]:
            per_tier[tier] = (test_id, tier, timeout_sec)

    # Ensure we have all tiers 1..6; if not, fill from defaults
    selected: List[Tuple[str, int, int]] = []
    defaults_by_tier = {tr: (tid, tr, to) for (tid, tr, to) in DEFAULT_FAST_TESTS}
    for tier in [1, 2, 3, 4, 5, 6]:
        if tier in per_tier:
            selected.append(per_tier[tier])
        else:
            selected.append(defaults_by_tier[tier])

    return selected


def _pick_top_k_per_tier(metrics: dict, k: int = TOP_K_PER_TIER) -> List[Tuple[str, int, int]]:
    """Select up to k fastest passing tests per tier from telemetry.

    Returns list of (test_id, tier, timeout_sec) across tiers 1..6.
    """
    from math import ceil

    per_tier_lists: Dict[int, List[Tuple[str, int, int]]] = {t: [] for t in range(1, 7)}

    for test_id, info in metrics.items():
        if test_id in EXCLUDED_TESTS:
            continue
        prefix = test_id.split("-")[0]
        tier = PREFIX_TO_TIER.get(prefix)
        if not tier:
            continue
        runs = info.get("runs", [])
        if not runs:
            continue
        # Only consider tests with a successful run on record
        last_successful = None
        for r in reversed(runs):
            if r.get("exit_code", 1) == 0:
                last_successful = r
                break
        if not last_successful:
            continue
        est = info.get("estimated_resources", {})
        runtime = est.get("runtime_sec", last_successful.get("runtime_sec", 30))
        timeout_mult = est.get("timeout_multiplier", 3.0)
        timeout_sec = int(max(30, min(300, ceil(runtime * timeout_mult + 10))))
        per_tier_lists[tier].append((test_id, tier, timeout_sec))

    # Sort and trim lists
    for t in per_tier_lists:
        per_tier_lists[t].sort(key=lambda x: x[2])  # shorter timeout first
        if len(per_tier_lists[t]) > k:
            per_tier_lists[t] = per_tier_lists[t][:k]

    # Fallback to defaults (at most one per tier) if empty
    defaults_by_tier = {tr: (tid, tr, to) for (tid, tr, to) in DEFAULT_FAST_TESTS}
    for t in range(1, 7):
        if not per_tier_lists[t] and t in defaults_by_tier:
            per_tier_lists[t] = [defaults_by_tier[t]]

    # Flatten in tier order
    selected: List[Tuple[str, int, int]] = []
    for t in range(1, 7):
        selected.extend(per_tier_lists[t])
    return selected


def run_parallel(tests: List[Tuple[str, int, int]]) -> Tuple[bool, str, str]:
    """Run selected tests in parallel via the suite orchestrator.

    Args:
        tests: List of (test_id, tier, timeout_sec)

    Returns:
        (success, stdout, stderr)
    """
    workspace_src = Path(__file__).parent.parent / "src"
    venv_python = Path(__file__).parent.parent.parent / ".venv" / "Scripts" / "python.exe"

    # CSV of test IDs
    test_ids = ",".join([tid for (tid, _tier, _to) in tests])
    # Parallel timeout: use max individual timeout + buffer; cap grows with test count
    max_to = max((to for (_tid, _tier, to) in tests), default=60)
    cap = 600 if len(tests) > 8 else 360
    overall_timeout = max(90, min(cap, max_to + 60))

    cmd = [
        str(venv_python),
        str(workspace_src / "run_parallel_suite.py"),
        "--tests", test_ids,
    "--max-concurrent", "6"  # modest parallelism for laptop GPU
    ]

    print(f"Invoking parallel suite for: {test_ids}")

    try:
        import os
        env = os.environ.copy()
        # Force UTF-8 stdout/stderr in child Python to avoid Windows cp1252 issues with Unicode
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        env.setdefault('PYTHONUTF8', '1')
        result = subprocess.run(
            cmd,
            cwd=str(workspace_src),
            capture_output=True,
            text=True,
            timeout=overall_timeout,
            encoding='utf-8',
            env=env
        )
        return (result.returncode == 0), result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return False, "", f"Parallel suite timeout after {overall_timeout}s"
    except Exception as e:
        return False, "", f"Parallel suite error: {e}"


def main():
    """Run pre-commit validation tests."""
    print("=" * 70)
    print("LFM Pre-Commit Validation")

    # STEP 1: Validate website sync with test harness
    print("\n" + "=" * 70)
    print("STEP 1: Website Sync Validation")
    print("=" * 70)
    
    # venv is at repository root (c:\LFM\.venv), not workspace
    venv_python = Path(__file__).parent.parent.parent / ".venv" / "Scripts" / "python.exe"
    validate_script = Path(__file__).parent / "validate_website_sync.py"
    
    if not venv_python.exists():
        print(f"❌ Python executable not found: {venv_python}")
        print(f"   Expected venv at: {venv_python.parent.parent}")
        return 1
    
    if not validate_script.exists():
        print(f"❌ Validation script not found: {validate_script}")
        return 1
    
    try:
        import os
        env = os.environ.copy()
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        env.setdefault('PYTHONUTF8', '1')
        result = subprocess.run(
            [str(venv_python), str(validate_script)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30,
            env=env
        )
        
        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)
        
        if result.returncode != 0:
            print("\n❌ WEBSITE OUT OF SYNC WITH TEST HARNESS")
            print("   Fix: node workspace/tools/generate_website_experiments.js")
            print("   Then retry commit")
            return 1
        
        print("✅ Website sync validated")
        
    except subprocess.TimeoutExpired:
        print("❌ Website validation timeout")
        return 1
    except Exception as e:
        print(f"❌ Website validation error: {e}")
        return 1
    # STEP 2: Physics Test Validation
    print("\n" + "=" * 70)
    print("STEP 2: Physics Test Validation")
    print("=" * 70)
    print("=" * 70)

    # Determine tests: prefer telemetry-driven selection
    metrics = _load_metrics_history()
    if metrics:
        fast_tests = _pick_top_k_per_tier(metrics, k=TOP_K_PER_TIER)
    else:
        fast_tests = DEFAULT_FAST_TESTS

    # Display selection summary by tier
    by_tier: Dict[int, int] = {}
    for _tid, tr, _to in fast_tests:
        by_tier[tr] = by_tier.get(tr, 0) + 1
    pretty = ", ".join([f"T{tr}:{cnt}" for tr, cnt in sorted(by_tier.items())])
    print(f"Selected {len(fast_tests)} tests (per-tier counts -> {pretty})\n")

    success, out, err = run_parallel(fast_tests)

    # STEP 3: Test harness unit tests
    print("\n" + "=" * 70)
    print("STEP 3: Test Harness Unit Tests")
    print("=" * 70)
    
    tests_dir = Path(__file__).parent.parent / "tests"
    harness_tests_ok = False
    
    try:
        harness_test_files = [
            "test_harness_energy.py",
            "test_validation.py",
            "test_harness_metadata.py",
            "test_harness_lattice_params.py",
            "test_harness_config.py",
            "test_harness_frequency.py",
            "test_validation_evaluators.py",
            "test_harness_integration.py"
        ]
        
        print(f"Running {len(harness_test_files)} harness test suites...")
        
        import os
        env = os.environ.copy()
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        env.setdefault('PYTHONUTF8', '1')
        
        result_harness = subprocess.run(
            [str(venv_python), "-m", "pytest"] + harness_test_files + ["-v", "--tb=short", "-q"],
            cwd=str(tests_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=120,
            env=env
        )
        
        # Extract summary line (last line with "passed")
        lines = result_harness.stdout.strip().split('\n')
        summary = [l for l in lines if 'passed' in l.lower() or 'failed' in l.lower()]
        if summary:
            print(summary[-1])
        
        harness_tests_ok = (result_harness.returncode == 0)
        
        if harness_tests_ok:
            print("✅ All harness tests passed")
        else:
            print("❌ Harness tests FAILED")
            print(result_harness.stdout[-1000:] if len(result_harness.stdout) > 1000 else result_harness.stdout)
            
    except subprocess.TimeoutExpired:
        print("❌ Harness tests timeout")
        harness_tests_ok = False
    except Exception as e:
        print(f"❌ Harness tests error: {e}")
        harness_tests_ok = False

    # STEP 4: Metadata validation of executed tests
    print("\n" + "=" * 70)
    print("STEP 4: Metadata Result Conformance")
    print("=" * 70)
    try:
        validator_script = Path(__file__).parent / "validate_metadata_results.py"
        if not validator_script.exists():
            print("[WARN] Metadata validator script missing; skipping.")
            metadata_ok = True
        else:
            test_ids_csv = ",".join([tid for (tid, _tier, _to) in fast_tests])
            venv_python = Path(__file__).parent.parent.parent / ".venv" / "Scripts" / "python.exe"
            result_meta = subprocess.run(
                [str(venv_python), str(validator_script), "--tests", test_ids_csv],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120
            )
            print(result_meta.stdout)
            if result_meta.stderr:
                print('[stderr]', result_meta.stderr)
            metadata_ok = (result_meta.returncode == 0)
            if not metadata_ok:
                print("❌ Metadata validation FAILED")
            else:
                print("✅ Metadata validation passed")
    except subprocess.TimeoutExpired:
        print("❌ Metadata validation timeout")
        metadata_ok = False
    except Exception as e:
        print(f"❌ Metadata validation error: {e}")
        metadata_ok = False

    # Echo orchestrator summary (tail if long)
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✅ Website sync: PASSED")
    print(f"{'✅' if success else '❌'} Physics tests: {'PASSED' if success else 'FAILED'}")
    print(f"{'✅' if harness_tests_ok else '❌'} Harness tests: {'PASSED' if harness_tests_ok else 'FAILED'}")
    print(f"{'✅' if metadata_ok else '❌'} Metadata conformance: {'PASSED' if metadata_ok else 'FAILED'}")
    print("=" * 70)
    if out:
        tail = out if len(out) < 4000 else out[-4000:]
        # Sanitize non-ASCII to avoid Windows cp1252 console encode errors
        safe_tail = ''.join(ch if ord(ch) < 128 else '?' for ch in tail)
        print(safe_tail)
    if err:
        print("\n[stderr]")
        print(err)

    if success and harness_tests_ok and metadata_ok:
        print("\n✅ ALL VALIDATION TESTS PASSED")
        print("Safe to commit changes")
        return 0
    else:
        if not success:
            print("\n❌ One or more physics tests failed (see summary above)")
        if not harness_tests_ok:
            print("\n❌ One or more harness unit tests failed")
        if not metadata_ok:
            print("\n❌ One or more tests violated tier metadata criteria")
        print("\nFIX ISSUES BEFORE COMMITTING")
        return 1


if __name__ == "__main__":
    sys.exit(main())
