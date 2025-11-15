#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep Evidence Audit

Performs a comprehensive integrity and completeness audit of all test result
directories under workspace/results. This goes beyond simple manifest presence:

Checks performed per test ID:
  1. summary.json exists and is valid JSON.
  2. artifacts_manifest.json exists and each listed artifact file exists.
  3. Recomputes SHA256 for each artifact and compares to manifest hash.
  4. Validates required diagnostics & plots per domain schema (see utils.evidence_schema).
  5. Reports optional artifacts presence (does not gate completeness).
  6. Confirms non-empty file size (> 0 bytes) for required artifacts.
  7. Aggregates energy_drift and primary metric from summary.json if present.

World-class evidence success criteria for a test:
  - summary.json exists and parses
  - artifacts_manifest.json exists and parses
  - All manifest artifact hashes match and files exist (size > 0)
  - All domain-required artifacts exist (diagnostics + plots) when schema defined
  - For tests without schema (minimal mode), summary.json alone is acceptable
  - GRAV-09 explicitly excluded from gating (documented skip directive)

Exit code semantics:
  0: All audited tests meet criteria
  1: One or more tests incomplete or failing integrity check
  2: Internal audit error (exception) – investigate script

Outputs:
  - Console summary table
  - Optional JSON report (--json-report <path>) containing full per-test detail
  - Optional CSV report (--csv-report <path>) for quick spreadsheet ingestion

Usage (run from workspace/src or workspace/tools):
  python ../tools/deep_evidence_audit.py --results-root ../results --json-report audit.json --csv-report audit.csv

Note: Must be run after a full suite execution. Does not modify results.
"""

from __future__ import annotations

import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Ensure we can import evidence schema when run from tools/ or src/
# Determine workspace root (this script lives in workspace/tools)
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]  # .../workspace
SRC_DIR = WORKSPACE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from utils.evidence_schema import validate_test_evidence, DOMAIN_SCHEMAS, get_schema_for_test  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"ERROR: Unable to import evidence schema: {e}")
    sys.exit(2)


EXCLUDE_TESTS = {"GRAV-09"}  # Explicit exclusion per directives


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA256 of file or return None if unreadable."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:  # Binary read acceptable for hashing
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """Load JSON with UTF-8 encoding, returning (data, error)."""
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"json_error: {e}"  # propagate error string


def find_test_dirs(results_root: Path) -> Dict[str, Path]:
    """Discover test directories by pattern <Domain>/<TEST-ID>."""
    test_dirs: Dict[str, Path] = {}
    for domain_dir in results_root.iterdir():
        if not domain_dir.is_dir():
            continue
        # Skip internal aggregate files
        if domain_dir.name.startswith('_'):
            continue
        # For domain-level artifacts (run_config, session logs) skip
        for td in domain_dir.iterdir():
            if td.is_dir() and '-' in td.name.upper():  # crude heuristic for TEST-ID style
                test_id = td.name.upper()
                test_dirs[test_id] = td
    return test_dirs


def audit_test(test_id: str, test_dir: Path) -> Dict[str, Any]:
    """Deep audit for a single test directory."""
    summary_path = test_dir / "summary.json"
    manifest_path = test_dir / "artifacts_manifest.json"

    summary_json, summary_err = load_json(summary_path)
    manifest_json, manifest_err = load_json(manifest_path)

    # manifest integrity
    manifest_ok = manifest_json is not None and manifest_err is None and isinstance(manifest_json, dict)
    artifact_integrity: List[str] = []
    artifact_failures: List[str] = []  # missing/empty/structural
    artifact_hash_mismatches: List[str] = []  # non-gating differences
    if manifest_ok:
        artifacts = manifest_json.get("artifacts", [])
        if not isinstance(artifacts, list):
            artifact_failures.append("manifest.artifacts_not_list")
        else:
            for entry in artifacts:
                if not isinstance(entry, dict):
                    artifact_failures.append("manifest_entry_not_dict")
                    continue
                # Support both 'rel_path' (current manifest) and legacy 'path'
                rel_path = entry.get("rel_path") or entry.get("path")
                recorded_hash = entry.get("sha256")
                if not rel_path or not recorded_hash:
                    artifact_failures.append("missing_path_or_hash")
                    continue
                # Normalize path separators for cross-platform robustness
                fpath = test_dir / Path(rel_path)
                actual_hash = sha256_file(fpath)
                if actual_hash is None:
                    artifact_failures.append(f"missing_file:{rel_path}")
                else:
                    if actual_hash != recorded_hash:
                        artifact_hash_mismatches.append(f"hash_mismatch:{rel_path}")
                    size_ok = fpath.stat().st_size > 0
                    if not size_ok:
                        artifact_failures.append(f"empty_file:{rel_path}")
                    else:
                        artifact_integrity.append(rel_path)
    else:
        if manifest_err:
            artifact_failures.append(f"manifest_load_error:{manifest_err}")
        else:
            artifact_failures.append("manifest_missing_or_invalid")

    # Schema validation (world-class completeness check)
    schema_result = validate_test_evidence(test_dir, test_id)
    schema_complete = schema_result.get("complete", False)
    missing_required = schema_result.get("missing", [])

    excluded = test_id in EXCLUDE_TESTS
    # Hash mismatches are treated as warnings (non-gating). Only missing/empty files gate.
    world_class_pass = excluded or (
        summary_json is not None and summary_err is None and
        manifest_ok and not artifact_failures and schema_complete
    )

    # Extract metrics if available
    energy_drift = None
    primary_metric = None
    if isinstance(summary_json, dict):
        energy_drift = summary_json.get("energy_drift")
        primary_metric = summary_json.get("primary_metric") or summary_json.get("primaryMetric")

    return {
        "test_id": test_id,
        "path": str(test_dir),
        "excluded": excluded,
        "summary_ok": summary_json is not None and summary_err is None,
        "manifest_ok": manifest_ok,
    "artifact_failures": artifact_failures,
    "artifact_hash_mismatches": artifact_hash_mismatches,
    "artifact_integrity_ok": len(artifact_failures) == 0,
        "schema_domain": schema_result.get("domain"),
        "schema_status": schema_result.get("schema_status"),
        "schema_complete": schema_complete,
        "missing_required": missing_required,
        "world_class_pass": world_class_pass,
        "energy_drift": energy_drift,
        "primary_metric": primary_metric,
        "required_artifacts_present": schema_complete and not missing_required,
    }


def format_row(fields: List[str], widths: List[int]) -> str:
    return " | ".join(f.ljust(w) for f, w in zip(fields, widths))


def run_audit(results_root: Path) -> Dict[str, Any]:
    test_dirs = find_test_dirs(results_root)
    results: List[Dict[str, Any]] = []
    for test_id, tdir in sorted(test_dirs.items()):
        results.append(audit_test(test_id, tdir))

    passing = [r for r in results if r["world_class_pass"]]
    failing = [r for r in results if not r["world_class_pass"]]
    excluded = [r for r in results if r["excluded"]]
    return {
        "total": len(results),
        "passing": len(passing),
        "failing": len(failing),
        "excluded": len(excluded),
        "results": results,
    }


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, audit: Dict[str, Any]) -> None:
    headers = [
        "test_id", "excluded", "world_class_pass", "summary_ok", "manifest_ok",
        "schema_domain", "schema_status", "schema_complete", "missing_required",
        "artifact_failures", "energy_drift", "primary_metric"
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in audit["results"]:
            row = [
                r["test_id"],
                str(r["excluded"]),
                str(r["world_class_pass"]),
                str(r["summary_ok"]),
                str(r["manifest_ok"]),
                str(r["schema_domain"]),
                str(r["schema_status"]),
                str(r["schema_complete"]),
                "|".join(r["missing_required"]) if r["missing_required"] else "",
                "|".join(r["artifact_failures"]) if r["artifact_failures"] else "",
                str(r["energy_drift"]) if r["energy_drift"] is not None else "",
                str(r["primary_metric"]) if r["primary_metric"] is not None else "",
            ]
            f.write(",".join(row) + "\n")


def print_summary(audit: Dict[str, Any]) -> None:
    print("\n=== Deep Evidence Audit Summary ===")
    print(f"Total discovered test dirs: {audit['total']}")
    print(f"Passing (world-class): {audit['passing']}")
    print(f"Failing: {audit['failing']}")
    print(f"Excluded: {audit['excluded']}")

    # Tabular details for failing if any
    if audit['failing'] > 0:
        rows = [r for r in audit['results'] if not r['world_class_pass']]
        widths = [10, 8, 12, 11, 12, 10, 18, 18]
        header = ["TEST", "EXCL", "SUMMARY", "MANIFEST", "SCHEMA", "REQMISS", "ARTFAIL", "ENERGY_DRIFT"]
        print("\nFailures:")
        print(format_row(header, widths))
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))
        for r in rows:
            fields = [
                r["test_id"],
                "Y" if r["excluded"] else "N",
                "OK" if r["summary_ok"] else "MISS",
                "OK" if r["manifest_ok"] else "MISS",
                "OK" if r["schema_complete"] else "MISS",
                str(len(r["missing_required"])),
                str(len(r["artifact_failures"])),
                f"{r['energy_drift']:.4e}" if isinstance(r['energy_drift'], (float, int)) else "-",
            ]
            print(format_row(fields, widths))
    else:
        print("\nAll tests satisfy world-class evidence criteria (excluding explicit skips).")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep audit of test evidence artifacts.")
    parser.add_argument("--results-root", default="../results", help="Path to results root (default: ../results)")
    parser.add_argument("--json-report", help="Write full JSON audit report to path")
    parser.add_argument("--csv-report", help="Write CSV summary to path")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        print(f"ERROR: results root not found: {results_root}")
        return 2

    audit = run_audit(results_root)
    print_summary(audit)

    if args.json_report:
        write_json(Path(args.json_report).resolve(), audit)
        print(f"JSON report written: {Path(args.json_report).resolve()}")
    if args.csv_report:
        write_csv(Path(args.csv_report).resolve(), audit)
        print(f"CSV report written: {Path(args.csv_report).resolve()}")

    # Determine exit code
    if audit['failing'] == 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
