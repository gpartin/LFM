# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
IP Audit Tool - Verifies IP protection elements across LFM codebase.

Checks:
- Copyright notices present in key files
- License files exist and are up-to-date
- NOTICE file contains IP protection statement
- No leaked credentials or sensitive information
- Consistent versioning across documents
- Zenodo/OSF links are current
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def check_copyright_notices() -> Tuple[bool, List[str]]:
    """Check that copyright notices are present in key files."""
    issues = []
    
    # Key files that should have copyright
    key_files = [
        Path("workspace/LICENSE"),
        Path("workspace/NOTICE"),
        Path("README.md"),
    ]
    
    for file_path in key_files:
        if not file_path.exists():
            issues.append(f"Missing required file: {file_path}")
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8')
            if "Copyright" not in content and "copyright" not in content:
                issues.append(f"No copyright notice in {file_path}")
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
    
    return len(issues) == 0, issues


def check_license_files() -> Tuple[bool, List[str]]:
    """Verify LICENSE and NOTICE files exist and contain proper content."""
    issues = []
    
    license_path = Path("workspace/LICENSE")
    notice_path = Path("workspace/NOTICE")
    
    # Check LICENSE exists
    if not license_path.exists():
        issues.append("LICENSE file missing")
    else:
        try:
            content = license_path.read_text(encoding='utf-8')
            if "CC BY-NC-ND" not in content:
                issues.append("LICENSE missing CC BY-NC-ND designation")
        except Exception as e:
            issues.append(f"Error reading LICENSE: {e}")
    
    # Check NOTICE exists
    if not notice_path.exists():
        issues.append("NOTICE file missing")
    else:
        try:
            content = notice_path.read_text(encoding='utf-8')
            if "All rights reserved" not in content:
                issues.append("NOTICE missing 'All rights reserved' statement")
            if "Greg D. Partin" not in content:
                issues.append("NOTICE missing author name")
        except Exception as e:
            issues.append(f"Error reading NOTICE: {e}")
    
    return len(issues) == 0, issues


def check_version_consistency() -> Tuple[bool, List[str]]:
    """Check version numbers are consistent across files."""
    issues = []
    
    version_file = Path("workspace/VERSION")
    if not version_file.exists():
        issues.append("VERSION file missing")
        return False, issues
    
    try:
        version_data = json.loads(version_file.read_text(encoding='utf-8'))
        version = version_data.get("version", "UNKNOWN")
        
        # Check key files mention the version
        files_to_check = [
            Path("workspace/uploads/zenodo/README.md"),
            Path("workspace/uploads/osf/README.md"),
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                if version not in content:
                    issues.append(f"Version {version} not found in {file_path}")
        
    except Exception as e:
        issues.append(f"Error checking version consistency: {e}")
    
    return len(issues) == 0, issues


def check_zenodo_links() -> Tuple[bool, List[str]]:
    """Verify Zenodo links are consistent and use new record URL."""
    issues = []
    
    # Should NOT contain old DOI
    old_doi = "10.5281/zenodo.17510124"
    
    # Should contain new record URL
    new_record = "https://zenodo.org/records/17618474"
    
    key_files = [
        Path("workspace/uploads/zenodo/README.md"),
        Path("workspace/uploads/osf/README.md"),
        Path("README.md"),
        Path("workspace/CITATION.cff"),
    ]
    
    for file_path in key_files:
        if not file_path.exists():
            issues.append(f"Missing file: {file_path}")
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check for old DOI (should not be present)
            if old_doi in content:
                issues.append(f"Old DOI {old_doi} still present in {file_path}")
            
            # Check for new record URL (should be present)
            if new_record not in content and file_path.suffix in ['.md', '.cff']:
                issues.append(f"New Zenodo record URL not found in {file_path}")
                
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
    
    return len(issues) == 0, issues


def check_sensitive_info() -> Tuple[bool, List[str]]:
    """Check for potential leaks of sensitive information."""
    issues = []
    
    # Patterns that should NOT appear in committed files
    sensitive_patterns = [
        (r'password\s*=\s*["\'].*["\']', "Hardcoded password"),
        (r'api[_-]?key\s*=\s*["\'].*["\']', "API key"),
        (r'secret\s*=\s*["\'].*["\']', "Secret token"),
    ]
    
    # Check key config and source files
    search_paths = [
        Path("workspace/config"),
        Path("workspace/src"),
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        for file_path in search_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for pattern, desc in sensitive_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"Potential {desc} in {file_path}")
                        
            except Exception:
                pass  # Skip binary or unreadable files
    
    return len(issues) == 0, issues


def run_audit() -> Dict[str, Tuple[bool, List[str]]]:
    """Run all audit checks and return results."""
    print("=" * 70)
    print("LFM IP AUDIT")
    print("=" * 70)
    print()
    
    checks = {
        "Copyright Notices": check_copyright_notices,
        "License Files": check_license_files,
        "Version Consistency": check_version_consistency,
        "Zenodo Links": check_zenodo_links,
        "Sensitive Information": check_sensitive_info,
    }
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks.items():
        print(f"Checking: {check_name}...")
        passed, issues = check_func()
        results[check_name] = (passed, issues)
        
        if passed:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
            all_passed = False
            for issue in issues:
                print(f"    - {issue}")
        print()
    
    print("=" * 70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - IP protection is properly maintained")
    else:
        print("✗ SOME CHECKS FAILED - Review issues above")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    import os
    
    # Change to repository root
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)
    
    results = run_audit()
    
    # Exit with error code if any check failed
    all_passed = all(passed for passed, _ in results.values())
    sys.exit(0 if all_passed else 1)
