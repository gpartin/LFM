#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
replace_contact_email.py — Safely update the repository-wide contact email.

Default behavior:
- Replaces all occurrences of the old email with the new one across source and docs
  while skipping generated artifacts and backups (docs/upload*, docs/evidence, results, etc.).
- Operates in dry-run mode unless --write is provided.

Usage examples:
  python tools/replace_contact_email.py                 # dry-run, show what would change
  python tools/replace_contact_email.py --write         # apply changes
  python tools/replace_contact_email.py --include-generated --write  # also update generated copies

Notes:
- Only text-like files are considered by extension and by successful UTF-8 decoding.
- Extensions scanned by default: .py, .md, .txt, .json, .csv, .yml, .yaml, .ini, .cfg, .toml, .rst, .license, .notice

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_OLD = "latticefieldmediumresearch@gmail.com"
DEFAULT_NEW = "latticefieldmediumresearch@gmail.com"

# Directories to skip by default (generated artifacts, caches, large outputs)
DEFAULT_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "results",
    os.path.join("docs", "upload"),
    os.path.join("docs", "upload_backup_20251102_153410"),
    os.path.join("docs", "upload_ref_20251102_153947"),
    os.path.join("docs", "upload_ref2_20251102_154019"),
    os.path.join("docs", "evidence"),
}

# File extensions to consider as text and safe to edit
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".csv",
    ".yml",
    ".yaml",
    ".ini",
    ".cfg",
    ".toml",
    ".rst",
    ".license",
    ".notice",
}


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    # Try UTF-8 sniff as a fallback (small files only to avoid large reads)
    try:
        if path.stat().st_size > 2_000_000:  # 2 MB guard; skip large unknowns
            return False
        with path.open("rb") as f:
            data = f.read(4096)
        data.decode("utf-8")
        return True
    except Exception:
        return False


def should_skip_dir(root: Path, dir_name: str, include_generated: bool) -> bool:
    rel = os.path.relpath((root / dir_name).as_posix(), Path.cwd().as_posix())
    # Normalize path separators for matching
    rel = rel.replace("\\", "/")
    for d in DEFAULT_SKIP_DIRS:
        d_norm = d.replace("\\", "/")
        if rel.endswith(d_norm) or rel == d_norm:
            # Allow opting into generated folders
            if include_generated and (d_norm.startswith("docs/upload") or d_norm.startswith("docs/evidence")):
                return False
            return True
    return False


def replace_in_file(path: Path, old: str, new: str, write: bool) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return 0
    count = text.count(old)
    if count > 0 and write:
        new_text = text.replace(old, new)
        path.write_text(new_text, encoding="utf-8", newline="\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely replace contact email across repository")
    parser.add_argument("--old", default=DEFAULT_OLD, help="Old email string to replace")
    parser.add_argument("--new", default=DEFAULT_NEW, help="New email string to use")
    parser.add_argument("--write", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also update generated artifacts (docs/upload*, docs/evidence, etc.)",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root directory to operate on (defaults to repository root)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    old = args.old
    new = args.new
    write = args.write
    include_generated = args.include_generated

    total_files = 0
    total_replacements = 0
    changed_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        root_path = Path(dirpath)
        # Prune dirs in-place
        pruned = []
        for d in list(dirnames):
            if should_skip_dir(root_path, d, include_generated):
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for fname in filenames:
            path = root_path / fname
            if not is_text_file(path):
                continue
            total_files += 1
            c = replace_in_file(path, old, new, write)
            if c:
                changed_files += 1
                total_replacements += c

    mode = "WRITE" if write else "DRY-RUN"
    print(f"Mode          : {mode}")
    print(f"Root          : {root}")
    print(f"Include gen   : {include_generated}")
    print(f"Old -> New    : {old} -> {new}")
    print(f"Files scanned : {total_files}")
    print(f"Files changed : {changed_files}")
    print(f"Replacements  : {total_replacements}")
    if not write:
        print("(no files were modified; re-run with --write to apply changes)")


if __name__ == "__main__":
    main()
