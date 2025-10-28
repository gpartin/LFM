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
