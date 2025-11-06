#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
lfm_results.py — Result handling and structured output for all LFM tiers.
Handles safe directory creation, summary writing, CSV utilities, and
metadata bundling. Works with lfm_logger and lfm_plotting.
"""

import csv, json, os, tempfile, time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------
def ensure_dirs(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_workspace_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor directory named 'workspace' from start or CWD.

    Falls back to the current working directory if no such ancestor exists.
    """
    start_path = Path(start) if start is not None else Path.cwd()
    if start_path.name.lower() == "workspace":
        return start_path
    for p in [start_path] + list(start_path.parents):
        if p.name.lower() == "workspace":
            return p
    # Fallback: if a 'workspace' child exists under start_path, use it
    candidate = start_path / "workspace"
    if candidate.is_dir():
        return candidate
    return start_path

def get_results_root() -> Path:
    """Return the canonical results root under the workspace directory."""
    ws = get_workspace_root()
    results = ws / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results

# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------
def write_json(path, data):
    """Write structured JSON safely (with timestamp) using atomic replace."""
    ensure_dirs(Path(path).parent)
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    # Convert Python and NumPy types to JSON-serializable values
    def convert_types(obj):
        import numpy as np
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, np.ndarray):
            return convert_types(obj.tolist())
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_types(x) for x in obj]
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(convert_types(data), f, indent=2)
        Path(tmp_name).replace(path)
    finally:
        try:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass

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
    """Write CSV with optional header using atomic replace."""
    ensure_dirs(Path(path).parent)
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            for r in rows:
                w.writerow(r)
        Path(tmp_name).replace(path)
    finally:
        try:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass

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

    # Ensure schema and license metadata are present for downstream consumers
    summary_data.setdefault("schema_version", "1.0")
    summary_data.setdefault("license", "CC BY-NC-ND 4.0")

    summary_path = base / "summary.json"
    write_json(summary_path, summary_data)

    if metrics:
        write_csv(base / "metrics.csv", metrics, header=["metric", "value"])

    # Auto-generate a lightweight readme.txt summarizing this result folder
    try:
        _generate_result_readme(base)
    except Exception as e:
        # Do not fail the run if readme generation has a transient issue
        # Callers and tests can still enforce presence separately.
        print(f"Warning: failed to generate readme.txt for {base}: {e}")

    return str(summary_path)

# ---------------------------------------------------------------------
# Per-result README generation (lightweight, no external deps)
# ---------------------------------------------------------------------
def _generate_result_readme(result_dir: Path):
    """
    Create or overwrite result_dir/readme.txt with a short summary of contents.
    Mirrors the essentials of tools/generate_results_readmes.py for a single folder
    to keep test harnesses self-contained.
    """
    result_dir = Path(result_dir)
    ensure_dirs(result_dir)

    summary_path = result_dir / 'summary.json'
    metrics_csv = result_dir / 'metrics.csv'
    diagnostics_dir = result_dir / 'diagnostics'
    plots_dir = result_dir / 'plots'

    has_summary = summary_path.exists()
    has_metrics_csv = metrics_csv.exists()
    csv_files = list(result_dir.glob('*.csv'))
    # Exclude metrics.csv from the general CSV count for clarity
    csv_count = len([p for p in csv_files if p.name != 'metrics.csv'])
    plot_count = len(list(plots_dir.glob('*.png'))) if plots_dir.exists() else 0

    # Extract a few simple scalars from summary.json if present
    metric_lines = []
    if has_summary:
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            metrics = summary.get('metrics', {})
            for key in [
                'runtime_sec', 'exit_code', 'peak_cpu_percent',
                'peak_memory_mb', 'peak_gpu_memory_mb'
            ]:
                if key in metrics and isinstance(metrics[key], (int, float, str)):
                    metric_lines.append(f"- {key}: {metrics[key]}")
            if not metric_lines:
                metric_lines.append("- (summary.json present but no simple scalar metrics to display)")
        except Exception as e:
            metric_lines.append(f"- Failed to parse summary.json: {e}")
    else:
        metric_lines.append("- (no summary.json found)")

    readme_text = []
    readme_text.append(f"LFM Results — {result_dir.name}")
    readme_text.append("")
    readme_text.append("This folder contains the outputs for a single LFM test run.")
    readme_text.append("")
    readme_text.append("## Overview")
    readme_text.append(f"- Contains summary.json: {has_summary}")
    readme_text.append(f"- Contains metrics.csv: {has_metrics_csv}")
    readme_text.append(f"- CSV files (excluding metrics.csv): {csv_count}")
    readme_text.append(f"- Plot images: {plot_count}")
    readme_text.append("")
    readme_text.append("## Key Metrics (from summary.json)")
    readme_text.extend(metric_lines)
    readme_text.append("")
    readme_text.append("Generated automatically by lfm_results._generate_result_readme().")
    readme_text.append("")
    readme_text.append("Legal: CC BY-NC-ND 4.0 — Non-commercial use only; no derivatives. See LICENSE and NOTICE.")

    with open(result_dir / 'readme.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(readme_text))

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
        results_dir = get_results_root()
    else:
        results_dir = Path(results_dir)
    
    # Test categories from central registry (fallback to legacy if unavailable)
    try:
        from harness.lfm_tiers import get_tiers
        _tiers = get_tiers()
        categories = {
            int(t["tier"]): {
                "name": t.get("category_name", t.get("name", t.get("dir"))),
                "prefix": t.get("prefix"),
                "expected": int(t.get("expected", 0)),
                "dir": t.get("dir"),
            } for t in _tiers
        }
    except Exception:
        categories = {
            1: {"name": "Relativistic", "prefix": "REL", "expected": 15, "dir": "Relativistic"},
            2: {"name": "Gravity Analogue", "prefix": "GRAV", "expected": 25, "dir": "Gravity"},
            3: {"name": "Energy Conservation", "prefix": "ENER", "expected": 11, "dir": "Energy"},
            4: {"name": "Quantization", "prefix": "QUAN", "expected": 9, "dir": "Quantization"},
        }
    
    # Load presentation overrides (optional)
    overrides_path = Path("config/presentation_overrides.json")
    overrides = {}
    try:
        if overrides_path.exists():
            with open(overrides_path, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
    except Exception:
        overrides = {}
    skip_overrides = set(overrides.get("skip_tests", []))
    note_overrides = overrides.get("notes", {})

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
                    if summary.get("skipped") is True:
                        status = "SKIP"
                    elif "status" in summary:
                        status = summary["status"]
                    elif "passed" in summary:
                        # Handle None/Null specifically (treat as UNKNOWN unless skipped is True)
                        passed_val = summary["passed"]
                        if isinstance(passed_val, bool):
                            status = "PASS" if passed_val else "FAIL"
                        else:
                            status = "UNKNOWN"
                    else:
                        status = "UNKNOWN"
                    
                    # Apply presentation overrides: force SKIP for listed tests
                    if test_id in skip_overrides:
                        status = "SKIP"

                    # Normalize status values to uppercase for consistent counting
                    # (handles "Passed", "passed", "PASS", "Pass", etc.)
                    status_upper = status.upper()
                    if status_upper in ["PASSED", "PASS", "TRUE"]:
                        status = "PASS"
                    elif status_upper in ["FAILED", "FAIL", "FALSE"]:
                        status = "FAIL"
                    elif status_upper in ["SKIPPED", "SKIP"]:
                        status = "SKIP"
                    # else keep original status (UNKNOWN, etc.)
                    
                    description = summary.get("description", "")
                    notes = summary.get("notes", "")
                    if summary.get("skipped") is True and summary.get("skip_reason"):
                        notes = (notes + "; " if notes else "") + str(summary.get("skip_reason"))
                    if test_id in note_overrides:
                        notes = (notes + "; " if notes else "") + str(note_overrides.get(test_id))
                    
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
    
    # Write to file atomically with a simple lock to avoid concurrent clobber
    output_file = results_dir / "MASTER_TEST_STATUS.csv"
    lock_file = results_dir / "MASTER_TEST_STATUS.lock"

    def _acquire_lock(timeout_s=3.0, interval_s=0.05):
        start = time.time()
        while True:
            try:
                with open(lock_file, 'x', encoding='utf-8') as _:
                    return True
            except FileExistsError:
                if time.time() - start > timeout_s:
                    print("Warning: timeout acquiring MASTER_TEST_STATUS lock; proceeding without lock")
                    return False
                time.sleep(interval_s)

    def _release_lock():
        try:
            if lock_file.exists():
                lock_file.unlink()
        except Exception:
            pass

    _acquire_lock()
    try:
        content = '\n'.join(lines)
        fd, tmp_name = tempfile.mkstemp(prefix=output_file.name + ".", dir=str(output_file.parent))
        try:
            # Write CSV with UTF-8 BOM so it's Excel-friendly on Windows
            with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(content)
            Path(tmp_name).replace(output_file)
        finally:
            try:
                if Path(tmp_name).exists():
                    Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
    finally:
        _release_lock()

    return output_file

# ---------------------------------------------------------------------
# Human-friendly inspection
# ---------------------------------------------------------------------
def inspect_result_dir(result_dir: Path) -> str:
    """Return a human-readable summary for a single result directory."""
    d = Path(result_dir)
    summary_path = d / 'summary.json'
    metrics_csv = d / 'metrics.csv'
    plots_dir = d / 'plots'
    diagnostics_dir = d / 'diagnostics'
    lines = []
    lines.append(f"Result: {d}")
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            lines.append(f"- test_id: {s.get('test_id')}")
            lines.append(f"- tier/category: {s.get('tier')}/{s.get('category')}")
            lines.append(f"- status: {s.get('status')}")
            metrics = s.get('metrics', {})
            for k in ['runtime_sec','exit_code','peak_cpu_percent','peak_memory_mb','peak_gpu_memory_mb']:
                if k in metrics:
                    lines.append(f"- {k}: {metrics[k]}")
        except Exception as e:
            lines.append(f"- Failed to read summary.json: {e}")
    else:
        lines.append("- summary.json: missing")

    lines.append(f"- metrics.csv: {'present' if metrics_csv.exists() else 'missing'}")
    lines.append(f"- diagnostics/: {'present' if diagnostics_dir.exists() else 'missing'}")
    if plots_dir.exists():
        pngs = list(plots_dir.glob('*.png'))
        lines.append(f"- plots/: {len(pngs)} .png file(s)")
    else:
        lines.append("- plots/: missing")
    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LFM results utilities")
    sub = parser.add_subparsers(dest="cmd")
    p_inspect = sub.add_parser("inspect", help="Inspect a result directory")
    p_inspect.add_argument("path", help="Path to results/<Category>/<TEST_ID> directory")
    args = parser.parse_args()
    if args.cmd == "inspect":
        print(inspect_result_dir(Path(args.path)))
    else:
        parser.print_help()
