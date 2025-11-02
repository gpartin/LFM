#!/usr/bin/env python3
"""
Script to add copyright and license headers to all Python files in LFM project.

This script:
1. Finds all .py files in the project
2. Checks if they already have copyright headers
3. Adds standardized copyright/license headers if missing
4. Preserves shebang lines and existing docstrings

Usage: python add_copyright_headers.py
"""

import os
import sys
from pathlib import Path

# Copyright header template
COPYRIGHT_HEADER = """# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: gpartin@gmail.com
"""

def has_copyright_header(content: str) -> bool:
    """Check if file already has copyright header."""
    return "Copyright (c) 2025 Greg D. Partin" in content or \
           "Licensed under CC BY-NC 4.0" in content

def add_header_to_file(filepath: Path) -> tuple[bool, str]:
    """
    Add copyright header to a Python file.
    
    Returns:
        (modified, message) tuple
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, f"Error reading: {e}"
    
    # Skip if already has copyright
    if has_copyright_header(content):
        return False, "Already has copyright header"
    
    lines = content.split('\n')
    new_lines = []
    insert_index = 0
    
    # Handle shebang
    if lines and lines[0].startswith('#!'):
        new_lines.append(lines[0])
        insert_index = 1
    
    # Add copyright header
    header_lines = COPYRIGHT_HEADER.strip().split('\n')
    new_lines.extend(header_lines)
    new_lines.append('')  # Blank line after header
    
    # Add rest of original content
    new_lines.extend(lines[insert_index:])
    
    # Write back
    try:
        new_content = '\n'.join(new_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, "Header added successfully"
    except Exception as e:
        return False, f"Error writing: {e}"

def find_python_files(root_dir: Path) -> list[Path]:
    """Find all Python files in directory tree."""
    python_files = []
    
    # Skip these directories
    skip_dirs = {'__pycache__', '.pytest_cache', '.git', 'venv', 'env'}
    
    for path in root_dir.rglob('*.py'):
        # Skip if in excluded directory
        if any(skip in path.parts for skip in skip_dirs):
            continue
        python_files.append(path)
    
    return sorted(python_files)

def main():
    """Main execution function."""
    # Get project root (where this script is located)
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    print("=" * 80)
    print("LFM Copyright Header Addition Tool")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print()
    
    # Find all Python files
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files")
    print()
    
    # Process each file
    modified_count = 0
    skipped_count = 0
    error_count = 0
    
    for filepath in python_files:
        relative_path = filepath.relative_to(project_root)
        modified, message = add_header_to_file(filepath)
        
        if modified:
            print(f"✅ {relative_path}")
            modified_count += 1
        elif "Error" in message:
            print(f"❌ {relative_path}: {message}")
            error_count += 1
        else:
            print(f"⏭️  {relative_path}: {message}")
            skipped_count += 1
    
    # Summary
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  Total files processed: {len(python_files)}")
    print(f"  ✅ Headers added: {modified_count}")
    print(f"  ⏭️  Skipped (already had headers): {skipped_count}")
    print(f"  ❌ Errors: {error_count}")
    print("=" * 80)
    
    if modified_count > 0:
        print()
        print("✅ Copyright headers successfully added!")
        print("   All Python files now have legal protection.")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
