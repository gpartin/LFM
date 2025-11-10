#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate test results against tier validation metadata files.

Usage:
    # Validate specific tests by ID (auto-detect tier by prefix)
    python tools/validate_metadata_results.py --tests "REL-01,GRAV-12,ENER-01"

    # Validate all tests for a tier by scanning results (Tier 1 shown)
    python tools/validate_metadata_results.py --tier 1

Exit codes:
    0 - All requested tests conform to metadata
    1 - One or more tests violate metadata
    2 - Error (bad input, missing files)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure workspace/src is on sys.path so 'harness' package resolves when run from tools/
try:
    _ROOT = Path(__file__).resolve().parents[1]
    _SRC = _ROOT / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
except Exception:
    pass

from harness.validation import load_tier_metadata, check_primary_metric, get_energy_threshold

PREFIX_TO_TIER = {
    "REL": 1, "GRAV": 2, "ENER": 3, "QUAN": 4, "EM": 5, "COUP": 6, "THERM": 7
}


def _workspace_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()


def _tier_dir_name(tier: int) -> str:
    return {
        1: "Relativistic",
        2: "Gravity",
        3: "Energy",
        4: "Quantization",
        5: "Electromagnetic",
        6: "Coupling",
        7: "Thermodynamics",
    }.get(int(tier), f"Tier{tier}")


def _find_summary(test_id: str) -> Path:
    prefix = test_id.split("-")[0].upper()
    tier = PREFIX_TO_TIER.get(prefix)
    if not tier:
        raise FileNotFoundError(f"Unknown test prefix for {test_id}")
    tier_dir = _tier_dir_name(tier)
    # Corrected path: results live under workspace/results not repo_root/results
    root = _workspace_root() / "results" / tier_dir / test_id
    return root / "summary.json"


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_tests(test_ids: List[str]) -> Tuple[bool, List[str]]:
    """Validate a set of tests by IDs.

    Returns: (all_ok, messages)
    """
    messages: List[str] = []
    any_fail = False

    # Group by tier for metadata loading
    by_tier: Dict[int, List[str]] = {}
    for tid in test_ids:
        p = tid.split("-")[0].upper()
        tr = PREFIX_TO_TIER.get(p)
        if not tr:
            messages.append(f"[ERROR] Unknown test ID {tid}")
            any_fail = True
            continue
        by_tier.setdefault(tr, []).append(tid)

    for tier, tids in sorted(by_tier.items()):
        try:
            meta = load_tier_metadata(tier)
        except Exception as e:
            messages.append(f"[WARN] Tier {tier} metadata missing: {e} (skipping)")
            continue

        for tid in tids:
            s_path = _find_summary(tid)
            if not s_path.exists():
                messages.append(f"[FAIL] {tid}: summary.json not found at {s_path}")
                any_fail = True
                continue
            try:
                summary = _load_json(s_path)
            except Exception as e:
                messages.append(f"[FAIL] {tid}: could not parse summary.json ({e})")
                any_fail = True
                continue

            # Primary metric check
            primary_ok, metric_key, value, thr = check_primary_metric(meta, tid, summary)
            if not primary_ok:
                if thr is None:
                    messages.append(f"[FAIL] {tid}: primary metric '{metric_key}' failed (value={value})")
                else:
                    messages.append(f"[FAIL] {tid}: primary metric '{metric_key}'={value:.6g} exceeds threshold {thr:.6g}")
                any_fail = True
            else:
                if thr is None:
                    messages.append(f"[PASS] {tid}: primary metric '{metric_key}' OK")
                else:
                    messages.append(f"[PASS] {tid}: primary '{metric_key}'={value:.6g} (thr={thr:.6g})")

            # Energy conservation (if present in summary)
            ed = None
            for key in ("energy_drift", "relative_energy_drift"):
                if key in summary:
                    try:
                        ed = float(summary[key])
                        break
                    except Exception:
                        pass
            if ed is not None:
                thr_e = get_energy_threshold(meta, tid, default=0.02)
                if ed >= thr_e:
                    messages.append(f"[FAIL] {tid}: energy_drift={ed:.6g} exceeds {thr_e:.6g}")
                    any_fail = True
                else:
                    messages.append(f"[PASS] {tid}: energy_drift={ed:.6g} (thr={thr_e:.6g})")

    return (not any_fail), messages


def validate_tier(tier: int) -> Tuple[bool, List[str]]:
    """Validate all tests found in results folder for a given tier."""
    tier_dir = _workspace_root().parent / "results" / _tier_dir_name(tier)
    if not tier_dir.exists():
        return False, [f"[ERROR] Results directory not found: {tier_dir}"]
    tests = []
    for child in tier_dir.iterdir():
        if child.is_dir() and (child / "summary.json").exists():
            tests.append(child.name)
    return validate_tests(tests)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate test results against tier metadata")
    ap.add_argument("--tests", type=str, default=None, help="Comma-separated list of test IDs to validate")
    ap.add_argument("--tier", type=int, default=None, help="Validate all results found for a specific tier (1-7)")
    args = ap.parse_args()

    if not args.tests and not args.tier:
        print("Usage: --tests ID1,ID2 or --tier N")
        return 2

    if args.tests:
        ids = [s.strip() for s in args.tests.split(",") if s.strip()]
        ok, msgs = validate_tests(ids)
    else:
        ok, msgs = validate_tier(int(args.tier))

    for m in msgs:
        print(m)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
