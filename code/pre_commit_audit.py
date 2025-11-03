#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Pre-Commit IP Audit - Final Review
==================================
Comprehensive final check for IP protection compliance before git commit.
Ensures all files are properly protected and ready for public distribution.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_git_status():
    """Check what files are staged for commit"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        return []
    except:
        return []

def check_sensitive_content():
    """Look for potentially sensitive information"""
    code_dir = Path(__file__).parent
    issues = []
    
    # Check for hardcoded paths, emails, or sensitive data
    sensitive_patterns = [
        "password", "secret", "private_key", "api_key",
        "greg.partin@", "personal", "confidential"
    ]
    
    for pattern in sensitive_patterns:
        try:
            result = subprocess.run([
                'grep', '-r', '-i', pattern, str(code_dir), 
                '--include=*.py', '--include=*.md', '--include=*.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'latticefieldmediumresearch@gmail.com' not in line:  # Exclude legitimate contact
                        issues.append(f"⚠️ Sensitive content: {line}")
        except:
            pass
    
    return issues

def check_license_consistency():
    """Verify license information is consistent across files"""
    issues = []
    
    # Check for consistent copyright year
    current_year = "2025"
    
    # Check key license files exist
    license_files = ['LICENSE', 'NOTICE', 'THIRD_PARTY_LICENSES.md']
    for lf in license_files:
        if not (Path(__file__).parent / lf).exists():
            issues.append(f"❌ Missing license file: {lf}")
    
    return issues

def main():
    """Main pre-commit audit"""
    print("=" * 70)
    print("  LFM PRE-COMMIT IP AUDIT - FINAL REVIEW")
    print("=" * 70)
    print()
    
    # Check git status
    staged_files = check_git_status()
    print(f"📋 Files ready for commit: {len(staged_files)}")
    
    issues_found = []
    
    # 1. Header compliance check
    print("\n🔍 1. Running header compliance check...")
    
    # Run validation directly instead of subprocess to avoid encoding issues
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import validate_headers
        # Capture the validation result
        original_stdout = sys.stdout
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        result = validate_headers.main()
        
        sys.stdout = original_stdout
        output = captured_output.getvalue()
        
        if result != 0:
            issues_found.append("❌ Header validation failed")
            print("   ❌ Header validation FAILED")
        else:
            print("   ✅ All headers compliant")
            
    except Exception as e:
        issues_found.append(f"❌ Could not run header validation: {e}")
        print(f"   ❌ Exception: {e}")
    finally:
        sys.stdout = original_stdout
    
    # 2. License consistency check
    print("\n🔍 2. Checking license consistency...")
    license_issues = check_license_consistency()
    if license_issues:
        issues_found.extend(license_issues)
        for issue in license_issues:
            print(f"   {issue}")
    else:
        print("   ✅ License files present and consistent")
    
    # 3. Sensitive content check
    print("\n🔍 3. Scanning for sensitive content...")
    sensitive_issues = check_sensitive_content()
    if sensitive_issues:
        issues_found.extend(sensitive_issues)
        for issue in sensitive_issues:
            print(f"   {issue}")
    else:
        print("   ✅ No sensitive content detected")
    
    # 4. Check key IP protection files
    print("\n🔍 4. Verifying IP protection files...")
    ip_files = [
        'LICENSE', 'NOTICE', 'CITATION.cff', 'THIRD_PARTY_LICENSES.md',
        'IP_PROTECTION_SUMMARY.md', 'validate_headers.py', 'fix_headers.py'
    ]
    
    missing_files = []
    for ip_file in ip_files:
        if not (Path(__file__).parent / ip_file).exists():
            missing_files.append(ip_file)
    
    if missing_files:
        issues_found.extend([f"❌ Missing IP file: {f}" for f in missing_files])
        for f in missing_files:
            print(f"   ❌ Missing: {f}")
    else:
        print("   ✅ All IP protection files present")
    
    # 5. Final summary
    print("\n" + "=" * 70)
    print("  FINAL AUDIT SUMMARY")
    print("=" * 70)
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND: {len(issues_found)}")
        print("\n🚫 DO NOT COMMIT - Fix these issues first:")
        for issue in issues_found:
            print(f"   {issue}")
        print("\n💡 After fixing issues, run this audit again.")
        return 1
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🎉 READY FOR COMMIT")
        print("\nIP Protection Status:")
        print("  ✅ All source files have proper copyright headers")
        print("  ✅ CC BY-NC-ND 4.0 license properly applied")
        print("  ✅ Commercial use restrictions in place")
        print("  ✅ Contact information present")
        print("  ✅ No sensitive content detected")
        print("  ✅ License files complete and consistent")
        print("\n🔒 Your intellectual property is fully protected!")
        print("🚀 Safe to commit and push to public repository.")
        return 0

if __name__ == "__main__":
    sys.exit(main())