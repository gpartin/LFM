#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
check_source_headers.py
=======================

Scans source files under the workspace directory and verifies a standard
license/copyright header is present. Can also auto-fix files.

Header policy (concise, non‑commercial emphasis):
- SPDX: CC-BY-NC-ND-4.0
- Non-commercial only; no distribution of modified material; attribution required.
- Year auto-fills to current year.

Supported file types: .ts, .tsx, .js, .jsx, .py

Usage (run from anywhere):
    python workspace/tools/check_source_headers.py --check
    python workspace/tools/check_source_headers.py --fix

Exit codes:
    0 - All files compliant (or fixed successfully with --fix)
    1 - Non-compliant files found (in --check mode), or errors occurred

Notes:
- For Next.js client/server components, the header is inserted AFTER any
  initial "use client" / "use server" directive to avoid breaking semantics.
- For Python files, a UTF-8 encoding line is ensured at the very top; the header
  appears after the shebang and encoding line, but before code.
- Files matched by .headerignore are skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple
import fnmatch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTS = {".ts", ".tsx", ".js", ".jsx", ".py"}
SPDX_TAG = "SPDX-License-Identifier: CC-BY-NC-ND-4.0"

HEADERIGNORE_FILE = WORKSPACE_ROOT / ".headerignore"
DEFAULT_IGNORES = [
    "**/.git/**",
    "**/.vscode/**",
    "**/node_modules/**",
    "**/.next/**",
    "**/coverage/**",
    "**/dist/**",
    "**/build/**",
    "**/out/**",
    "**/results/**",
    "**/uploads/**",
]


def _load_ignore_patterns() -> List[str]:
    patterns: List[str] = []
    if HEADERIGNORE_FILE.exists():
        try:
            for line in HEADERIGNORE_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
        except Exception:
            pass
    return patterns + DEFAULT_IGNORES


def _is_ignored(path: Path, patterns: Iterable[str]) -> bool:
    s = str(path.as_posix())
    for pat in patterns:
        if fnmatch.fnmatch(s, pat):
            return True
    return False


@dataclass
class FileCheckResult:
    path: Path
    compliant: bool
    fixed: bool = False
    error: str | None = None


def _build_headers(ext: str) -> Tuple[List[str], List[str]]:
    """Return (prefix_lines_before_header, header_lines) for a given extension.

    For TS/JS/TSX/JSX, we insert after any leading 'use client'/'use server' directives.
    For Python, ensure encoding line; header comes after shebang+encoding.
    """
    year = datetime.now().year

    if ext in {".ts", ".tsx", ".js", ".jsx"}:
        header_lines = [
            "/*",
            f" * © {year} Emergent Physics Lab. All rights reserved.",
            " * Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).",
            " * Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.",
            f" * {SPDX_TAG}",
            " */",
            "",
        ]
        return ([], header_lines)

    elif ext == ".py":
        header_lines = [
            f"# © {year} Emergent Physics Lab. All rights reserved.",
            "# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).",
            "# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.",
            f"# {SPDX_TAG}",
            "",
        ]
        return ([], header_lines)

    else:
        return ([], [])


def _has_spdx(lines: List[str]) -> bool:
    top = lines[:20]
    return any(SPDX_TAG in ln for ln in top)


def _insert_header_ts_js(lines: List[str], header_lines: List[str]) -> List[str]:
    # Keep leading directives like 'use client' or 'use server' (single quotes or double quotes)
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped in {"'use client';", '"use client";', "'use server';", '"use server";'}:
            idx += 1
            continue
        # Allow an initial shebang in JS (rare) or empty lines
        if stripped.startswith("#!") or stripped == "":
            idx += 1
            continue
        break
    return lines[:idx] + header_lines + lines[idx:]


def _insert_header_py(lines: List[str], header_lines: List[str]) -> List[str]:
    new_lines = list(lines)
    idx = 0
    # Shebang handling
    if idx < len(new_lines) and new_lines[idx].startswith("#!"):
        idx += 1
    # Ensure encoding line exists at the very top (after shebang if present)
    enc_line = "# -*- coding: utf-8 -*-"
    if not (idx < len(new_lines) and enc_line in new_lines[idx]):
        new_lines[idx:idx] = [enc_line + "\n"]
        idx += 1
    else:
        idx += 1  # skip existing encoding line
    # Insert header after shebang+encoding
    return new_lines[:idx] + header_lines + new_lines[idx:]


def check_or_fix_file(path: Path, fix: bool) -> FileCheckResult:
    try:
        ext = path.suffix.lower()
        prefix, header_lines = _build_headers(ext)
        if not header_lines:
            return FileCheckResult(path=path, compliant=True)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(True)  # keepends
        if _has_spdx(lines):
            return FileCheckResult(path=path, compliant=True)
        if not fix:
            return FileCheckResult(path=path, compliant=False)
        # Perform insertion
        if ext in {".ts", ".tsx", ".js", ".jsx"}:
            new_lines = _insert_header_ts_js(lines, [l + ("\n" if not l.endswith("\n") else "") for l in header_lines])
        elif ext == ".py":
            new_lines = _insert_header_py(lines, [l + ("\n" if not l.endswith("\n") else "") for l in header_lines])
        else:
            return FileCheckResult(path=path, compliant=True)
        path.write_text("".join(new_lines), encoding="utf-8")
        return FileCheckResult(path=path, compliant=True, fixed=True)
    except Exception as e:
        return FileCheckResult(path=path, compliant=False, error=str(e))


def _update_year_range_in_lines(lines: List[str]) -> Tuple[List[str], bool]:
    """Update © year or year range to include current year if needed.

    Looks for a line with '© YYYY' or '© YYYY-YYYY' near the top and updates the
    end year to the current year when appropriate. Returns (new_lines, changed).
    """
    cur_year = datetime.now().year
    year_re = re.compile(r"(©\s*)(\d{4})(\s*-\s*(\d{4}))?(\b)")
    changed = False
    new_lines = list(lines)
    for i in range(min(30, len(new_lines))):
        m = year_re.search(new_lines[i])
        if not m:
            continue
        start = int(m.group(2))
        end_str = m.group(4)
        # If start year is in the future, skip
        if start > cur_year:
            break
        # Determine new replacement
        if end_str is None:
            if start < cur_year:
                repl = f"{m.group(1)}{start}-{cur_year}{m.group(5)}"
            else:
                break
        else:
            end = int(end_str)
            if end < cur_year:
                repl = f"{m.group(1)}{start}-{cur_year}{m.group(5)}"
            else:
                break
        new_lines[i] = year_re.sub(repl, new_lines[i], count=1)
        changed = True
        break
    return new_lines, changed


def iter_source_files(root: Path, exts: Iterable[str], ignore_patterns: Iterable[str]) -> Iterable[Path]:
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if _is_ignored(p, ignore_patterns):
            continue
        yield p


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify or insert license headers across the workspace.")
    ap.add_argument("--check", action="store_true", help="Check mode: non-compliant files cause non-zero exit.")
    ap.add_argument("--fix", action="store_true", help="Fix mode: insert headers where missing (idempotent).")
    ap.add_argument("--update-year", action="store_true", help="Update © year to a YYYY-<current> range when needed.")
    ap.add_argument("--ext", action="append", dest="exts", help="Limit to file extensions (e.g. --ext .ts --ext .py)")
    args = ap.parse_args(argv)

    if not (args.check or args.fix or args.update_year):
        ap.error("Specify --check or --fix or --update-year")

    exts = set(a.lower() for a in (args.exts or [])) or set(DEFAULT_EXTS)
    ignore_patterns = _load_ignore_patterns()

    root = WORKSPACE_ROOT
    non_compliant: List[FileCheckResult] = []
    fixed_count = 0
    error_count = 0
    total = 0

    updated_years = 0
    for f in iter_source_files(root, exts, ignore_patterns):
        total += 1
        # Year update runs regardless of check/fix; do it first when requested
        if args.update_year:
            try:
                text = f.read_text(encoding="utf-8")
                lines = text.splitlines(True)
                new_lines, changed = _update_year_range_in_lines(lines)
                if changed:
                    f.write_text("".join(new_lines), encoding="utf-8")
                    updated_years += 1
            except Exception:
                # Non-fatal: continue with other checks
                pass

        if args.check or args.fix:
            res = check_or_fix_file(f, fix=args.fix)
            if not res.compliant:
                non_compliant.append(res)
            if res.fixed:
                fixed_count += 1
            if res.error:
                error_count += 1

    mode_parts = []
    if args.check:
        mode_parts.append("CHECK")
    if args.fix:
        mode_parts.append("FIX")
    if args.update_year:
        mode_parts.append("UPDATE-YEAR")
    mode = "/".join(mode_parts) if mode_parts else "—"
    print(f"\n[{mode}] Scanned {total} files | Non-compliant: {len(non_compliant)} | Fixed: {fixed_count} | Year-updated: {updated_years} | Errors: {error_count}")

    if non_compliant and args.check:
        print("\nNon-compliant files (sample up to 50):")
        for r in non_compliant[:50]:
            print(f" - {r.path.relative_to(root)}{f'  (error: {r.error})' if r.error else ''}")
        print("\nHint: run with --fix to insert headers automatically.")
        return 1

    if error_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
