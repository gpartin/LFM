#!/usr/bin/env python3
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

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
