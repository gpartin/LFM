#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com

"""
lfm_results.py — Result handling and structured output for all LFM tiers.
Handles safe directory creation, summary writing, CSV utilities, and
metadata bundling. Works with lfm_logger and lfm_plotting.
"""

import csv, json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------
def ensure_dirs(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------
def write_json(path, data):
    """Write structured JSON safely (with timestamp)."""
    ensure_dirs(Path(path).parent)
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    # Convert Python and NumPy types to JSON-serializable values
    def convert_types(obj):
        import numpy as np
        # Handle NumPy scalar types (includes np.bool_, np.integer, np.floating, etc.)
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                # Fallback: convert to str if item() fails for some exotic scalar
                return str(obj)
        # NumPy arrays -> lists (recursively converted by calling convert_types again)
        if isinstance(obj, np.ndarray):
            return convert_types(obj.tolist())
        # Python built-in bool, int, float, str are JSON-serializable as-is
        if isinstance(obj, (bool, int, float, str)):
            return obj
        # Containers: recurse
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_types(x) for x in obj]
        # Unknown objects: use str() as a safe fallback to avoid json encoder errors
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert_types(data), f, indent=2)

def read_json(path):
    """Read JSON file (if exists) else return None."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def write_csv(path, rows, header=None):
    """Write CSV with optional header."""
    ensure_dirs(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def read_csv(path):
    """Read CSV into list of rows."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))

# ---------------------------------------------------------------------
# Result bundle helpers
# ---------------------------------------------------------------------
def save_summary(base_dir, test_id, summary_data, metrics=None):
    """
    Save both summary.json and metrics.csv in standard LFM format.
    base_dir: root folder (e.g. results/Tier1/REL-01/)
    summary_data: dict with metadata, parameters, status, tolerances, etc.
    metrics: list of (name, value) pairs for metrics.csv
    """
    base = Path(base_dir)
    ensure_dirs(base)

    summary_path = base / "summary.json"
    write_json(summary_path, summary_data)

    if metrics:
        write_csv(base / "metrics.csv", metrics, header=["metric", "value"])

    return str(summary_path)

# ---------------------------------------------------------------------
# Proof-bundle metadata
# ---------------------------------------------------------------------
def write_metadata_bundle(base_dir, test_id, tier, category, hardware_info=None):
    """
    Create a lightweight metadata.json for reproducibility/auditing.
    Includes date, test id, tier, category, and optional hardware info.
    """
    bundle = {
        "test_id": test_id,
        "tier": tier,
        "category": category,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": hardware_info or {}
    }
    write_json(Path(base_dir) / "metadata.json", bundle)
    return bundle

# ---------------------------------------------------------------------
# Master test status tracking
# ---------------------------------------------------------------------
def update_master_test_status(results_dir: Path = None):
    """
    Scan results directory and update MASTER_TEST_STATUS.csv with current test results.
    Should be called after any test completes (individual, tier, or parallel suite).
    
    Args:
        results_dir: Path to results directory (default: ./results)
    """
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)
    
    # Test categories and expected counts
    categories = {
        1: {"name": "Relativistic", "prefix": "REL", "expected": 15, "dir": "Relativistic"},
        2: {"name": "Gravity Analogue", "prefix": "GRAV", "expected": 25, "dir": "Gravity"},
        3: {"name": "Energy Conservation", "prefix": "ENRG", "expected": 11, "dir": "Energy"},
        4: {"name": "Quantization", "prefix": "QUANT", "expected": 9, "dir": "Quantization"},
    }
    
    # Scan all summary.json files
    all_tests = {}
    for tier, cat_info in categories.items():
        cat_dir = results_dir / cat_info["dir"]
        if not cat_dir.exists():
            continue
            
        for test_dir in cat_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            summary_file = test_dir / "summary.json"
            if summary_file.exists():
                try:
                    # Read JSON summaries as UTF-8 to support non-ASCII characters
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    test_id = summary.get("test_id", summary.get("id", test_dir.name))
                    # Determine status from various possible field names
                    if "status" in summary:
                        status = summary["status"]
                    elif "passed" in summary:
                        status = "PASS" if summary["passed"] else "FAIL"
                    else:
                        status = "UNKNOWN"
                    description = summary.get("description", "")
                    notes = summary.get("notes", "")
                    
                    all_tests[test_id] = {
                        "tier": tier,
                        "description": description,
                        "status": status,
                        "notes": notes
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not read {summary_file}: {e}")
    
    # Generate CSV content
    lines = []
    lines.append("MASTER TEST STATUS REPORT - LFM Lattice Field Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Validation Rule: Suite marked NOT RUN if any test missing from CSV")
    lines.append("")
    lines.append("CATEGORY SUMMARY")
    lines.append("Tier,Category,Expected_Tests,Tests_In_CSV,Status,Pass_Rate")
    
    # Category summaries
    for tier in sorted(categories.keys()):
        cat_info = categories[tier]
        prefix = cat_info["prefix"]
        expected = cat_info["expected"]
        
        # Find all tests for this category
        tier_tests = {tid: info for tid, info in all_tests.items() if info["tier"] == tier}
        tests_in_csv = len(tier_tests)
        
        # Count statuses
        passed = sum(1 for t in tier_tests.values() if t["status"] == "PASS")
        failed = sum(1 for t in tier_tests.values() if t["status"] == "FAIL")
        skipped = sum(1 for t in tier_tests.values() if t["status"] in ["SKIP", "SKIPPED"])
        unknown = sum(1 for t in tier_tests.values() if t["status"] not in ["PASS", "FAIL", "SKIP", "SKIPPED"])
        missing = expected - tests_in_csv
        
        # Determine overall status
        if tests_in_csv == 0 or missing > expected // 2:
            status = "NOT RUN"
        elif passed == tests_in_csv:
            status = "PASS"
        elif passed > 0:
            status = "PARTIAL"
        else:
            status = "FAIL"
        
        # Build pass rate string
        parts = []
        if passed > 0:
            parts.append(f"{passed}/{tests_in_csv} passed")
        if skipped > 0:
            parts.append(f"{skipped} skipped")
        if missing > 0:
            parts.append(f"{missing} missing")
        if unknown > 0:
            parts.append(f"{unknown} unknown")
        
        pass_rate = " - ".join(parts) if parts else f"{passed}/{tests_in_csv} (100%)"
        
        lines.append(f"Tier {tier},{cat_info['name']},{expected},{tests_in_csv},{status},{pass_rate}")
    
    lines.append("")
    lines.append("DETAILED TEST RESULTS")
    lines.append("")
    
    # Detailed results by tier
    for tier in sorted(categories.keys()):
        cat_info = categories[tier]
        tier_tests = {tid: info for tid, info in all_tests.items() if info["tier"] == tier}
        
        if not tier_tests:
            continue
            
        lines.append(f"TIER {tier} - {cat_info['name'].upper()} ({len(tier_tests)}/{cat_info['expected']} tests)")
        lines.append("Test_ID,Description,Status,Notes")
        
        # Sort by test ID
        for test_id in sorted(tier_tests.keys()):
            info = tier_tests[test_id]
            desc = info["description"].replace(",", ";")  # Escape commas
            notes = info["notes"].replace(",", ";")  # Escape commas
            lines.append(f"{test_id},{desc},{info['status']},{notes}")
        
        lines.append("")
    
    # Write to file
    output_file = results_dir / "MASTER_TEST_STATUS.csv"
    # Write CSV with UTF-8 BOM so it's Excel-friendly on Windows and avoids cp1252 encode errors
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('\n'.join(lines))
    
    return output_file
