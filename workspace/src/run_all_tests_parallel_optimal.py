#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Optimal Parallel Test Execution (Alias)

This is a thin wrapper that delegates to the unified parallel runner
`run_all_tiers_parallel.py` in per-test mode. It preserves the familiar
entry point while using the richer features of the unified runner:
- Longest-first scheduling based on prior runtimes/heuristics
- Live heartbeats (done/active/queued) and per-test START lines
- Per-test log files (UTF-8 safe), final banner, and JSON summary

Usage examples (PowerShell):
  python src\run_all_tests_parallel_optimal.py
  python src\run_all_tests_parallel_optimal.py --max-workers 8 --test-timeout 1200
  python src\run_all_tests_parallel_optimal.py --limit 6  # smoke run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimal per-test parallel runner (alias)")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum parallel workers")
    parser.add_argument("--test-timeout", type=int, default=1200, help="Per-test timeout (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N tests (smoke)")
    parser.add_argument("--only-missing", action="store_true", help="Run only tests that have not produced summary.json yet")
    parser.add_argument("--only-failed", action="store_true", help="Run only tests that previously failed (status FAIL)")
    args = parser.parse_args()

    workspace = Path(__file__).parent.parent
    delegate = workspace / "src" / "run_all_tiers_parallel.py"

    cmd = [
        sys.executable,
        str(delegate),
        "--mode", "tests",
        "--max-workers", str(args.max_workers),
        "--test-timeout", str(args.test_timeout),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.only_missing:
        cmd += ["--only-missing"]
    if args.only_failed:
        cmd += ["--only-failed"]

    print("Delegating to unified runner:")
    print("  ", " ".join(cmd))

    # Run with workspace root as CWD so configs resolve correctly
    result = subprocess.run(cmd, cwd=str(workspace))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
