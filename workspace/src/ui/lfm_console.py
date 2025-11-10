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
lfm_console.py — Unified console output control for all tiers.
Handles runtime verbosity, per-test status lines, and end summaries.
"""

import sys
import io
# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import datetime

# ---------------------------------------------------------------------
# Core print helpers
# ---------------------------------------------------------------------
# Global toggle: when False, diagnostic-level messages (from monitors
# and integrity checks) should be suppressed. Set by harness at start.
DIAGNOSTICS_ENABLED = False
# Optional global logger (LFMLogger) to emit structured JSON events alongside
# console messages. Set by harness with `set_logger()` so console helpers can
# write lightweight JSONL progress/events into the same results tree.
_GLOBAL_LOGGER = None

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


def set_diagnostics_enabled(enabled: bool):
    """Toggle whether diagnostic messages (non-critical warnings) are printed.

    This is intended to be set once at program startup by the harness based on
    the run config (run_settings.debug.enable_diagnostics)."""
    global DIAGNOSTICS_ENABLED
    DIAGNOSTICS_ENABLED = bool(enabled)

def set_logger(logger):
    """Bind an `LFMLogger` instance so console helpers can emit JSON events.

    The harness should call this after creating its `LFMLogger` so subsequent
    `test_start`, `report_progress`, and `suite_summary` can produce JSON
    events in the session log.
    """
    global _GLOBAL_LOGGER
    _GLOBAL_LOGGER = logger

def log_run_config(cfg: dict, out_dir=None):
    """Pretty-print the top-level run config to console and write a
    machine-readable copy to `out_dir/run_config.json` (if out_dir provided).
    """
    # Print a compact summary to console
    rs = cfg.get("run_settings", {}) if isinstance(cfg, dict) else {}
    msg = f"Run settings: quick_mode={rs.get('quick_mode', False)}; use_gpu={rs.get('use_gpu', True)}; verbose={rs.get('verbose', False)}"
    log(msg, "INFO")
    # Also write structured copy to the results folder if available
    if out_dir is not None:
        try:
            from utils.lfm_results import write_json
            from pathlib import Path
            write_json(Path(out_dir) / "run_config.json", cfg)
        except Exception:
            # best-effort only; don't raise on failure to write config
            pass

# ---------------------------------------------------------------------
# Progress / status utilities
# ---------------------------------------------------------------------
def test_start(test_id, desc, steps=None):
    log(f"-> Starting {test_id}: {desc} ({steps or '?'} steps)", "INFO")
    if _GLOBAL_LOGGER is not None:
        try:
            _GLOBAL_LOGGER.log_json({"event": "test_start", "test_id": test_id, "description": desc, "steps": steps})
        except Exception:
            pass

def test_pass(test_id, metric=None):
    msg = f"{test_id} PASS [OK]"
    if metric:
        msg += f" ({metric})"
    log(msg, "PASS")

def test_fail(test_id, reason=None):
    msg = f"{test_id} FAIL [X]"
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
    if _GLOBAL_LOGGER is not None:
        try:
            _GLOBAL_LOGGER.log_json({"event": "suite_summary", "total": total, "passed": passed, "overall": overall})
        except Exception:
            pass

def report_progress(test_id: str, percent: int, phase: str = None):
    """Emit a progress update to console and to the bound JSON logger.

    This is intended to be called occasionally (e.g. on percent boundaries)
    to avoid overly spammy logs. The harness should throttle reports as
    appropriate.
    """
    msg = f"[{test_id}] {percent}% complete"
    if phase:
        msg += f" ({phase})"
    log(msg, "INFO")
    if _GLOBAL_LOGGER is not None:
        try:
            _GLOBAL_LOGGER.log_json({"event": "progress", "test_id": test_id, "percent": int(percent), "phase": phase})
        except Exception:
            pass

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
