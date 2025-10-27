#!/usr/bin/env python3
"""
lfm_console.py — Unified console output control for all tiers.
Handles runtime verbosity, per-test status lines, and end summaries.
"""

import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------
# Core print helpers
# ---------------------------------------------------------------------
def log(msg, level="INFO", end="\n", flush=True):
    """Timestamped console output with color by level."""
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    color = {
        "INFO": "\033[97m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "PASS": "\033[92m",
        "FAIL": "\033[91m",
        "RESET": "\033[0m"
    }
    prefix = f"[{timestamp}] [{level}] "
    sys.stdout.write(f"{color.get(level, '')}{prefix}{msg}{color['RESET']}{end}")
    if flush:
        sys.stdout.flush()

# ---------------------------------------------------------------------
# Progress / status utilities
# ---------------------------------------------------------------------
def test_start(test_id, desc, steps=None):
    log(f"→ Starting {test_id}: {desc} ({steps or '?'} steps)", "INFO")

def test_pass(test_id, metric=None):
    msg = f"{test_id} PASS ✅"
    if metric:
        msg += f" ({metric})"
    log(msg, "PASS")

def test_fail(test_id, reason=None):
    msg = f"{test_id} FAIL ❌"
    if reason:
        msg += f" ({reason})"
    log(msg, "FAIL")

def suite_summary(results):
    """Pretty-print a formatted PASS/FAIL table and overall result."""
    log("=== SUITE SUMMARY ===", "INFO")
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    for r in results:
        tid = r.get("test_id", r.get("id", "?"))
        desc = r.get("description", r.get("desc", "?"))
        status = "PASS ✅" if r.get("passed") else "FAIL ❌"
        log(f"{tid:>7} | {status:7} | {desc}", "INFO")

    log(f"\nTOTAL: {passed}/{total} passed", "INFO")
    overall = "PASS ✅" if passed == total else "FAIL ❌"
    log(f"OVERALL RESULT: {overall}", "INFO")

# ---------------------------------------------------------------------
# Simple runtime timer
# ---------------------------------------------------------------------
class Timer:
    def __init__(self, label=""):
        self.label = label
        self.start_time = None
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.start_time
        log(f"{self.label} completed in {dt:.2f}s", "INFO")
