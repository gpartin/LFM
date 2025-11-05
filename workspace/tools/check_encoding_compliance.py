#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Python files for missing encoding='utf-8' in file operations.

Usage:
    python tools/check_encoding_compliance.py
    python tools/check_encoding_compliance.py --fix  # Auto-fix issues (coming soon)
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Check a Python file for encoding violations.
    
    Returns list of (line_number, line_content, violation_type) tuples.
    """
    violations = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return violations
    
    lines = content.split('\n')
    
    # Patterns to detect
    patterns = [
        # open() without encoding
        (r"open\([^)]*\)", r"encoding\s*=", "open() missing encoding="),
        
        # Path.read_text() without encoding
        (r"\.read_text\(\)", r"encoding\s*=", "Path.read_text() missing encoding="),
        
        # Path.write_text() without encoding  
        (r"\.write_text\(", r"encoding\s*=", "Path.write_text() missing encoding="),
        
        # pd.read_csv() without encoding
        (r"pd\.read_csv\(", r"encoding\s*=", "pd.read_csv() missing encoding="),
        
        # pd.to_csv() without encoding
        (r"\.to_csv\(", r"encoding\s*=", "DataFrame.to_csv() missing encoding="),
    ]
    
    for line_num, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
            
        for pattern, encoding_pattern, violation_type in patterns:
            if re.search(pattern, line):
                # Check if encoding= is present in this line or continuation
                if not re.search(encoding_pattern, line):
                    # Check if this is part of multi-line call
                    # (simple heuristic: if line ends with comma or no closing paren)
                    if not (line.rstrip().endswith(',') or ')' not in line):
                        violations.append((line_num, line.strip(), violation_type))
    
    return violations


def scan_directory(root_dir: Path, exclude_dirs: List[str] = None) -> dict:
    """
    Scan all Python files in directory tree.
    
    Returns dict mapping file paths to their violations.
    """
    if exclude_dirs is None:
        exclude_dirs = ['.venv', 'venv', '__pycache__', '.git', 'node_modules', 'build', 'dist']
    
    results = {}
    
    for py_file in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        
        violations = check_file(py_file)
        if violations:
            results[py_file] = violations
    
    return results


def print_report(results: dict):
    """Print formatted report of violations."""
    if not results:
        print("✓ No encoding violations found!")
        return
    
    print(f"\n⚠️  Found encoding violations in {len(results)} file(s):\n")
    
    total_violations = 0
    for file_path, violations in sorted(results.items()):
        print(f"\n{file_path}:")
        for line_num, line_content, violation_type in violations:
            print(f"  Line {line_num}: {violation_type}")
            print(f"    {line_content}")
            total_violations += 1
    
    print(f"\nTotal violations: {total_violations}")
    print("\nFix by adding encoding='utf-8' to file operations.")
    print("See CODING_STANDARDS.md for details.\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check Python files for missing encoding='utf-8' parameters"
    )
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Directory to scan (default: workspace root)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix violations (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    if args.fix:
        print("Auto-fix not implemented yet. Please fix manually.")
        return 1
    
    print(f"Scanning {args.dir} for encoding violations...")
    results = scan_directory(args.dir)
    print_report(results)
    
    return 1 if results else 0


if __name__ == '__main__':
    sys.exit(main())
