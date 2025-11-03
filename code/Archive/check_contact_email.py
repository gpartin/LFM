#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
check_contact_email.py — Guardrail to prevent old contact email from reappearing in source.

Scans the repository for occurrences of the old contact email and fails if any are found
outside of explicitly allowed generated/snapshot directories.

Exit codes:
- 0: No violations found
- 1: Violations detected

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OLD = b"gpartin@gmail.com"  # Use bytes for exact matching

# Allowed paths (prefix match) where old email may legitimately persist (snapshots, evidence, outputs)
ALLOWED_PREFIXES = [
    "docs/upload/",            # current upload staging (regenerated periodically)
    "docs/upload_backup_",     # immutable backups
    "docs/upload_ref",         # immutable references
    "docs/evidence/",          # extracted evidence and templates
    "results/",                # run outputs
    ".git/",
    "__pycache__/",
]


def is_allowed(path: Path) -> bool:
    rel = path.as_posix()
    for p in ALLOWED_PREFIXES:
        if rel.startswith(p):
            return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = []
    scanned = 0
    self_rel_path = Path(__file__).resolve().relative_to(root).as_posix()
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        # Normalize for relative path
        for fname in filenames:
            p = Path(dirpath_p / fname)
            rel = p.relative_to(root)
            # Only check likely text files quickly
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                data = p.read_bytes()
                scanned += 1
                if OLD in data:
                    rel_posix = rel.as_posix()
                    # Skip this checker file itself (contains the OLD literal by design)
                    if rel_posix == self_rel_path:
                        continue
                    if not is_allowed(rel):
                        violations.append(rel_posix)
            except Exception:
                continue

    if violations:
        print("Old contact email found in restricted locations:")
        for v in violations:
            print(" -", v)
        print(f"Total files scanned: {scanned}; violations: {len(violations)}")
        return 1
    else:
        print(f"No restricted occurrences of old email. Total files scanned: {scanned}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
