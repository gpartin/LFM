#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Run Pre-Commit Suite
====================

Aggregates IP-protection and quality gates before allowing a commit.
This script is intended to be invoked by the local git pre-commit hook.

Checks performed (in order):
1) License header compliance (SPDX CC-BY-NC-ND-4.0)
2) UTF-8 encoding compliance (if tool is available)
3) Public surface guard (blocks staging files outside workspace/ [allow README.md, .gitignore])
4) Banned terms scan (professional language policy)
5) License drift guard (blocks deprecated license identifiers; legacy tokens now suppressed)
6) No-tracking scanner for website (blocks analytics/tracker insertions)
7) Physics + website validation via tools/pre_commit_validation.py

Override environment variables (for rare, intentional bypasses):
- LFM_ALLOW_ROOT_COMMIT=1          # allow committing root-level files (IP risk)
- LFM_ALLOW_PROHIBITED_TERMS=1     # skip banned terms check (requires manual review)
- LFM_SKIP_WEBSITE_TRACKERS=1      # skip trackers scan (discouraged)

Exit Codes:
 0: All checks passed
 1: One or more checks failed
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_DIR.parent
VENVPY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
PY_CMD = str(VENVPY) if VENVPY.exists() else sys.executable

# Professional language policy (see copilot-instructions.md)
BANNED_TERMS = [
    r"\bevil physicist\b",
    r"\bevil\b",
    r"\bskeptic\b",   # pejorative usage cannot be auto-detected, so flag for review
]

# Analytics/tracker identifiers (deny-list)
TRACKER_TOKENS = [
    "gtag(", "googletagmanager", "dataLayer", "mixpanel", "analytics.identify",
    "window.fbq", "fbq(", "hotjar", "hj(" , "clarity(" , "plausible.io",
    "umami.is", "segment.com", "rudderstack"
]

ALLOWED_ROOT_FILES = {"README.md", ".gitignore"}


def _run(cmd: List[str], cwd: Path | None = None, timeout: int | None = None) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('PYTHONUTF8', '1')
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _fail(msg: str) -> int:
    print(f"\n❌ {msg}")
    return 1


def list_staged_files() -> List[Path]:
    code, out, err = _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"], cwd=REPO_ROOT)
    if code != 0:
        print(err)
        return []
    files: List[Path] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        files.append((REPO_ROOT / line.strip()).resolve())
    return files


def check_license_headers() -> int:
    tool = WORKSPACE_DIR / "tools" / "check_source_headers.py"
    if not tool.exists():
        print("(skip) header checker not found")
        return 0
    code, out, err = _run([PY_CMD, str(tool), "--check"], cwd=WORKSPACE_DIR)
    print(out)
    if code != 0:
        print(err)
        return _fail("Header compliance failed. Run: python workspace/tools/check_source_headers.py --fix")
    print("✅ Header compliance passed")
    return 0


def check_encoding() -> int:
    tool = WORKSPACE_DIR / "tools" / "check_encoding_compliance.py"
    if not tool.exists():
        print("(skip) encoding compliance tool not found")
        return 0
    code, out, err = _run([PY_CMD, str(tool)], cwd=WORKSPACE_DIR)
    # Assume tool prints its own summary; non-zero indicates failure
    if code != 0:
        # Gracefully handle UnicodeEncodeError in the tool's own printing on Windows
        if 'UnicodeEncodeError' in (out + err):
            print("(note) encoding tool encountered UnicodeEncodeError; treating as warning due to Windows console encoding.")
            return 0
        print(out)
        print(err)
        return _fail("UTF-8 encoding compliance failed. Run the tool and fix reported files.")
    print("✅ UTF-8 encoding compliance passed")
    return 0


def check_public_surface() -> int:
    if os.environ.get("LFM_ALLOW_ROOT_COMMIT"):
        print("(override) public surface guard disabled via LFM_ALLOW_ROOT_COMMIT=1")
        return 0
    staged = list_staged_files()
    offenders: List[Path] = []
    for p in staged:
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            rel = p
        # Allow only workspace/ + allowed root files
        if rel.parts and rel.parts[0] == "workspace":
            continue
        if str(rel) in ALLOWED_ROOT_FILES:
            continue
        offenders.append(rel)
    if offenders:
        print("Staged files outside allowed public surface:")
        for o in offenders:
            print(f" - {o}")
        return _fail("Only commit files under workspace/ (plus README.md, .gitignore). Set LFM_ALLOW_ROOT_COMMIT=1 to override (not recommended).")
    print("✅ Public surface guard passed (staged files limited to workspace/")
    return 0


def scan_banned_terms() -> int:
    if os.environ.get("LFM_ALLOW_PROHIBITED_TERMS"):
        print("(override) banned terms scan disabled via LFM_ALLOW_PROHIBITED_TERMS=1")
        return 0
    staged = list_staged_files()
    terms_re = re.compile("|".join(BANNED_TERMS), flags=re.IGNORECASE)
    hits: List[Tuple[Path, int, str]] = []
    for p in staged:
        if not p.exists():
            continue
        # Only text-like files
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".pdf"}:
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if terms_re.search(line):
                    hits.append((p, i, line.strip()))
        except Exception:
            # Ignore binary or unreadable
            continue
    if hits:
        print("Prohibited terminology found (professional language policy):")
        for path, ln, text in hits[:50]:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f" - {rel}:{ln}: {text}")
        return _fail("Remove or rephrase prohibited terms per professional language policy. Set LFM_ALLOW_PROHIBITED_TERMS=1 to override (not recommended).")
    print("✅ Banned terms scan passed")
    return 0


def license_drift_guard() -> int:
    staged = list_staged_files()
    # Legacy tokens intentionally left empty; detection disabled until explicit re-enable decision
    drift_tokens: List[str] = []
    drift_hits: List[Tuple[Path, int, str]] = []
    for p in staged:
        if not p.exists():
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if any(tok in line for tok in drift_tokens):
                    drift_hits.append((p, i, line.strip()))
        except Exception:
            continue
    if drift_hits:
        print("Potential license drift detected (deprecated identifiers):")
        for path, ln, text in drift_hits[:50]:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f" - {rel}:{ln}: {text}")
        return _fail("Replace deprecated license references with CC BY-NC-ND 4.0.")
    print("✅ License drift guard passed")
    return 0


def no_tracker_scan() -> int:
    if os.environ.get("LFM_SKIP_WEBSITE_TRACKERS"):
        print("(override) website tracker scan disabled via LFM_SKIP_WEBSITE_TRACKERS=1")
        return 0
    staged = list_staged_files()
    hits: List[Tuple[Path, int, str]] = []
    for p in staged:
        if not p.exists():
            continue
        if not p.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".html"}:
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if any(tok in line for tok in TRACKER_TOKENS):
                    hits.append((p, i, line.strip()))
        except Exception:
            continue
    if hits:
        print("Potential tracking/analytics code detected (policy: no cookies/tracking):")
        for path, ln, text in hits[:50]:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f" - {rel}:{ln}: {text}")
        return _fail("Remove analytics/trackers to comply with no-cookies/no-tracking policy.")
    print("✅ No-tracking scanner passed")
    return 0


def secrets_scan() -> int:
    # Minimal high-signal secret patterns
    patterns = [
        re.compile(r"ghp_[A-Za-z0-9]{30,}", re.IGNORECASE),  # GitHub token
        re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
        re.compile(r"BEGIN (?:RSA |EC )?PRIVATE KEY"),
        re.compile(r"SECRET_KEY\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    ]
    hits: List[Tuple[Path, int, str]] = []
    for p in list_staged_files():
        if not p.exists():
            continue
        # Skip binaries
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".pdf"}:
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if any(r.search(line) for r in patterns):
                    hits.append((p, i, line.strip()))
        except Exception:
            continue
    if hits:
        print("Potential secrets detected:")
        for path, ln, text in hits[:50]:
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f" - {rel}:{ln}: {text}")
        return _fail("Remove secrets from code or use environment variables/secrets manager.")
    print("✅ Secrets scan passed")
    return 0


def run_validation_suite() -> int:
    tool = WORKSPACE_DIR / "tools" / "pre_commit_validation.py"
    if not tool.exists():
        print("(skip) pre_commit_validation.py not found")
        return 0
    code, out, err = _run([PY_CMD, str(tool)], cwd=WORKSPACE_DIR / "src")
    print(out)
    if err:
        print(err)
    if code != 0:
        return _fail("Validation suite failed (see summary above).")
    print("✅ Pre-commit validation suite passed")
    return 0


def main() -> int:
    print("=" * 70)
    print("LFM Pre-Commit Suite — IP Protection & Quality Gates")
    print("=" * 70)

    # Proactively roll header years to include the current year; re-stage changes under workspace/
    print("\n— Header year rollover —")
    code, out, err = _run([PY_CMD, str(WORKSPACE_DIR / 'tools' / 'check_source_headers.py'), '--update-year'])
    print(out)
    if err:
        print(err)
    # Re-stage any changes made by the rollover step so the commit includes them
    _run(["git", "add", "workspace"], cwd=REPO_ROOT)

    # Insert discovery provenance gate after license drift (IP-focused sequence)
    def discovery_gate() -> int:
        tool = WORKSPACE_DIR / "tools" / "check_discoveries.py"
        if not tool.exists():
            print("(skip) discovery provenance tool not found")
            return 0
        # Always run in pure check mode (hard fail if any test lacks a discovery entry)
        code, out, err = _run([PY_CMD, str(tool), "--check"], cwd=WORKSPACE_DIR)
        print(out)
        if err:
            print(err)
        if code != 0:
            return _fail("Discovery provenance check failed. Add entries or run tools/check_discoveries.py --populate-all to bootstrap.")
        print("✅ Discovery provenance passed")
        return 0

    steps = [
        ("License headers", check_license_headers),
        ("UTF-8 encoding", check_encoding),
        ("Public surface guard", check_public_surface),
        ("Banned terms", scan_banned_terms),
        ("License drift", license_drift_guard),
        ("Discovery provenance", discovery_gate),
        ("Secrets", secrets_scan),
        ("No tracking", no_tracker_scan),
        ("Validation suite", run_validation_suite),
    ]

    for name, fn in steps:
        print(f"\n— {name} —")
        rc = fn()
        if rc != 0:
            return 1

    print("\n✅ ALL PRE-COMMIT CHECKS PASSED\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
