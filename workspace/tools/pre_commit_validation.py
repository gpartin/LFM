#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    print(f"Selected {len(fast_tests)} tests (per-tier counts → {pretty})\n")

    success, out, err = run_parallel(fast_tests)

    # Echo orchestrator summary (tail if long)
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (Parallel)")
    print("=" * 70)
    if out:
        tail = out if len(out) < 4000 else out[-4000:]
        print(tail)
    if err:
        print("\n[stderr]")
        print(err)

    if success:
        print("\n✅ ALL VALIDATION TESTS PASSED")
        print("✅ Safe to commit changes")
        return 0
    else:
        print("\n❌ One or more validation tests failed (see summary above)")
        print("\n⚠️  FIX FAILING TESTS BEFORE COMMITTING ⚠️")
        return 1


if __name__ == "__main__":
    sys.exit(main())
