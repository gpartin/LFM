#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
fix_license_headers.py — Update all license references from CC BY-NC-ND 4.0 to CC BY-NC-ND 4.0

This script replaces all occurrences of the old license identifier with the correct
non-commercial, no-derivatives license across the repository.
"""
from __future__ import annotations

import os
from pathlib import Path

# Pattern replacements
REPLACEMENTS = [
    ("CC BY-NC-ND 4.0", "CC BY-NC-ND 4.0"),
    ("Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)",
     "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)"),
    ("Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International",
     "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International"),
    ("Attribution-NonCommercial-NoDerivatives 4.0", "Attribution-NonCommercial-NoDerivatives 4.0"),
    ("https://creativecommons.org/licenses/by-nc-nd/4.0/",
     "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
    ("License-CC%20BY--NC--ND%204.0", "License-CC%20BY--NC--ND%204.0"),
    ("CC BY--NC--ND 4.0", "CC BY--NC--ND 4.0"),  # Markdown badge variant
]

# Skip patterns where we want to preserve "CC BY-NC-ND 4.0" as historical references
SKIP_PATTERNS = [
    "Earlier releases were distributed under CC BY-NC 4.0",
    "Earlier releases (v1.x and v2.x) were distributed under CC BY-NC 4.0",
    "Earlier releases (v2.x and prior) were distributed under CC BY-NC 4.0",
]

# Directories to completely skip (backups/snapshots)
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "docs/upload_backup_20251102_153410",
    "docs/upload_ref_20251102_153947",
    "docs/upload_ref2_20251102_154019",
}

# File extensions to process
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".rst", ".ini"}


def should_skip_dir(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    for d in SKIP_DIRS:
        if rel.startswith(d):
            return True
    return False


def process_file(path: Path) -> tuple[int, bool]:
    """Process a file and return (replacements_made, modified)"""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return 0, False

    original = content
    replacements_made = 0
    
    for old, new in REPLACEMENTS:
        if old in content:
            # Check if this line should be skipped (historical reference)
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                # Skip replacement if line contains a historical reference pattern
                skip_line = any(skip in line for skip in SKIP_PATTERNS)
                if not skip_line and old in line:
                    line = line.replace(old, new)
                    replacements_made += 1
                new_lines.append(line)
            content = "\n".join(new_lines)
    
    if content != original:
        path.write_text(content, encoding="utf-8", newline="\n")
        return replacements_made, True
    return 0, False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    total_files = 0
    modified_files = 0
    total_replacements = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        
        # Skip directories in-place
        pruned = []
        for d in list(dirnames):
            if should_skip_dir(dirpath_p / d, root):
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for fname in filenames:
            path = dirpath_p / fname
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            
            total_files += 1
            replacements, modified = process_file(path)
            if modified:
                modified_files += 1
                total_replacements += replacements
                print(f"✓ {path.relative_to(root)} ({replacements} replacements)")

    print(f"\n=== Summary ===")
    print(f"Files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")


if __name__ == "__main__":
    main()
