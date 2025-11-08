#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
"""Discovery Provenance Gate
============================

Phase 1 requirement: Every validated test under ``workspace/tests/`` must have
an associated discovery entry in ``workspace/docs/discoveries/discoveries.json``.

This script enforces and (optionally) helps populate missing discovery entries.

Why this matters:
  * Patent/IP defensibility demands traceable provenance from tests → discoveries.
  * Prevents silent introduction of un-documented physics validation.
  * Enables downstream upload/package builders to include complete discovery set.

Modes:
  --check (default)       : Fails (exit 1) if any test file lacks a discovery entry.
  --populate              : Adds entries ONLY for staged test files missing provenance.
  --populate-all          : Adds entries for ALL missing tests (use once to bootstrap).
  --list-missing          : Prints missing test paths then exits 0 (informational).

Environment overrides:
  LFM_ALLOW_MISSING_DISCOVERIES=1  → Converts missing provenance from hard error to warning.

Discovery Entry Schema (minimal placeholder auto-generated):
  {
    "date": "YYYY-MM-DD",               # ISO-8601 (today by default for placeholders)
    "tier": "Tier N - Auto Inferred",   # Inferred from test path (tier folder)
    "title": "PLACEHOLDER discovery for test_<name>.py",
    "summary": "Placeholder discovery entry auto-generated to satisfy provenance gate. Replace with validated discovery summary.",
    "evidence": "test_<name>.py",
    "links": ["tests/tierN/test_<name>.py"]
  }

IMPORTANT: Auto-generated placeholders MUST be manually refined with
authentic scientific content before release. Their presence only satisfies
the structural provenance gate.

Exit Codes:
  0: Success / all provenance satisfied
  1: Missing provenance (unless override active) or unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = REPO_ROOT / "workspace"
DISCOVERY_FILE = WORKSPACE_DIR / "docs" / "discoveries" / "discoveries.json"
TESTS_ROOT = WORKSPACE_DIR / "tests"


def _git_staged_test_files() -> Set[Path]:
    """Return staged test file paths (absolute) under tests/ matching test_*.py."""
    try:
        out = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"], cwd=REPO_ROOT, text=True, encoding="utf-8")
    except Exception:
        return set()
    staged: Set[Path] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        p = (REPO_ROOT / line).resolve()
        try:
            rel = p.relative_to(WORKSPACE_DIR)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "tests" and p.name.startswith("test_") and p.suffix == ".py":
            staged.add(p)
    return staged


def _load_discoveries() -> List[Dict]:
    if not DISCOVERY_FILE.exists():
        return []
    # Try standard UTF-8, then gracefully fall back to UTF-8 with BOM
    for enc in ("utf-8", "utf-8-sig"):
        try:
            text = DISCOVERY_FILE.read_text(encoding=enc)
            data = json.loads(text)
            if not isinstance(data, list):
                print("ERROR: discoveries.json is not a JSON array.")
                return []
            return data
        except Exception as e:
            last_err = e
            continue
    print(f"ERROR: Failed to parse discoveries.json: {last_err}")
    return []


def _extract_test_links(discoveries: List[Dict]) -> Set[str]:
    links: Set[str] = set()
    for entry in discoveries:
        for link in entry.get("links", []):
            if isinstance(link, str) and link.startswith("tests/tier"):
                links.add(link)
    return links


def _all_test_files() -> List[Path]:
    if not TESTS_ROOT.exists():
        return []
    return sorted([p for p in TESTS_ROOT.rglob("test_*.py") if p.is_file()])


def _infer_tier(path: Path) -> str:
    """Infer tier string from path like tests/tier5/..."""
    rel = path.relative_to(WORKSPACE_DIR)
    parts = rel.parts
    tier_folder = next((p for p in parts if p.startswith("tier") and len(p) > 4), None)
    if tier_folder and tier_folder[4:].isdigit():
        return f"Tier {tier_folder[4:]} - Auto Inferred"
    return "Unknown Tier"


def _create_placeholder_entry(test_path: Path) -> Dict:
    rel = test_path.relative_to(WORKSPACE_DIR)
    link = str(rel).replace("\\", "/")  # normalize to forward slashes
    tier = _infer_tier(test_path)
    return {
        "date": date.today().isoformat(),
        "tier": tier,
        "title": f"PLACEHOLDER discovery for {test_path.name}",
        "summary": "Placeholder discovery entry auto-generated to satisfy provenance gate. Replace with validated discovery summary.",
        "evidence": test_path.name,
        "links": [link],
    }


def _write_discoveries(entries: List[Dict]) -> None:
    DISCOVERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DISCOVERY_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    discoveries = _load_discoveries()
    existing_links = _extract_test_links(discoveries)
    all_tests = _all_test_files()
    missing_tests: List[Path] = []
    for t in all_tests:
        rel = t.relative_to(WORKSPACE_DIR)
        rel_str = str(rel).replace("\\", "/")
        if rel_str not in existing_links:
            missing_tests.append(t)

    if args.list_missing:
        if missing_tests:
            print("Missing discovery provenance for:")
            for m in missing_tests:
                print(f" - {m.relative_to(WORKSPACE_DIR)}")
        else:
            print("All tests have discovery provenance.")
        return 0

    if args.populate or args.populate_all:
        # Determine which set to populate
        if args.populate_all:
            to_populate = missing_tests
        else:
            staged = _git_staged_test_files()
            to_populate = [t for t in missing_tests if t in staged]
        if not to_populate:
            print("No missing tests to populate (mode: populate).")
        else:
            print(f"Auto-populating {len(to_populate)} discovery placeholder entries...")
            for test_path in to_populate:
                entry = _create_placeholder_entry(test_path)
                discoveries.append(entry)
            _write_discoveries(discoveries)
            print(f"Updated discoveries.json with {len(to_populate)} new placeholder entries.")
        # Recompute missing after population
        discoveries = _load_discoveries()
        existing_links = _extract_test_links(discoveries)
        missing_tests = [t for t in _all_test_files() if str(t.relative_to(WORKSPACE_DIR)).replace("\\", "/") not in existing_links]

    if missing_tests:
        allow = os.environ.get("LFM_ALLOW_MISSING_DISCOVERIES")
        print(f"ERROR: Missing discovery provenance for {len(missing_tests)} test(s):")
        for m in missing_tests[:100]:  # limit output
            print(f" - {m.relative_to(WORKSPACE_DIR)}")
        if allow:
            print("(override) Missing discoveries allowed via LFM_ALLOW_MISSING_DISCOVERIES=1")
            return 0
        print("Add discovery entries or run with --populate-all to generate placeholders.")
        return 1

    print("All tests have discovery provenance entries.")
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate discovery provenance for tests.")
    p.add_argument("--check", action="store_true", help="Explicitly run in check mode (default).")
    p.add_argument("--populate", action="store_true", help="Populate placeholders for staged missing tests only.")
    p.add_argument("--populate-all", action="store_true", help="Populate placeholders for ALL missing tests.")
    p.add_argument("--list-missing", action="store_true", help="List missing tests without failing.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    # Default mode is check if neither populate nor list provided
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
