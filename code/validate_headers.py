#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Header Validation Script
========================
Checks that all LFM source files have proper copyright headers and license information.
Ensures IP protection compliance across the entire codebase.
"""

import os
import sys
from pathlib import Path

def check_python_header(filepath):
    """Check if Python file has proper copyright header"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Check for required components in first 10 lines
        header_text = ''.join(lines[:10])
        
        required_elements = [
            "Copyright (c) 2025 Greg D. Partin",
            "All rights reserved",
            "CC BY-NC-ND 4.0",
            "Commercial use prohibited",
            "latticefieldmediumresearch@gmail.com"
        ]
        
        missing = []
        for element in required_elements:
            if element not in header_text:
                missing.append(element)
                
        return missing
        
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_markdown_header(filepath):
    """Check if Markdown file has proper copyright notice"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Check for copyright notice in comments in first 10 lines
        header_text = ''.join(lines[:10])
        
        if "Copyright (c) 2025 Greg D. Partin" in header_text:
            return []
        else:
            return ["Missing copyright notice"]
            
    except Exception as e:
        return [f"Error reading file: {e}"]

def check_script_header(filepath):
    """Check if batch/shell script has proper copyright header"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Check for required components in first 10 lines
        header_text = ''.join(lines[:10])
        
        if "Copyright (c) 2025 Greg D. Partin" in header_text:
            return []
        else:
            return ["Missing copyright notice"]
            
    except Exception as e:
        return [f"Error reading file: {e}"]

def main():
    """Main header validation"""
    print("=" * 60)
    print("  LFM COPYRIGHT HEADER VALIDATION")
    print("=" * 60)
    print()
    
    code_dir = Path(__file__).parent
    issues_found = []
    files_checked = 0
    
    # Check Python files
    print("\n🐍 Checking Python files..." if sys.stdout.encoding != 'cp1252' else "\nChecking Python files...")
    for py_file in code_dir.glob("*.py"):
        if py_file.name == "__pycache__":
            continue
            
        files_checked += 1
        missing = check_python_header(py_file)
        if missing:
            issues_found.append(f"❌ {py_file.name}: {', '.join(missing)}")
        else:
            print(f"   ✅ {py_file.name}")
    
    # Check subdirectories for Python files
    for subdir in ['tests', 'tools', 'archive']:
        subdir_path = code_dir / subdir
        if subdir_path.exists():
            for py_file in subdir_path.rglob("*.py"):
                files_checked += 1
                missing = check_python_header(py_file)
                if missing:
                    issues_found.append(f"❌ {py_file.relative_to(code_dir)}: {', '.join(missing)}")
                else:
                    print(f"   ✅ {py_file.relative_to(code_dir)}")
    
    # Check Markdown files
    print("\n📝 Checking Markdown files...")
    for md_file in code_dir.glob("*.md"):
        files_checked += 1
        missing = check_markdown_header(md_file)
        if missing:
            issues_found.append(f"❌ {md_file.name}: {', '.join(missing)}")
        else:
            print(f"   ✅ {md_file.name}")
    
    # Check root README
    root_readme = code_dir.parent / "README.md"
    if root_readme.exists():
        files_checked += 1
        missing = check_markdown_header(root_readme)
        if missing:
            issues_found.append(f"❌ ../README.md: {', '.join(missing)}")
        else:
            print(f"   ✅ ../README.md")
    
    # Check script files
    print("\n🔧 Checking setup scripts...")
    for script_file in code_dir.glob("*.bat"):
        files_checked += 1
        missing = check_script_header(script_file)
        if missing:
            issues_found.append(f"❌ {script_file.name}: {', '.join(missing)}")
        else:
            print(f"   ✅ {script_file.name}")
    
    for script_file in code_dir.glob("*.sh"):
        files_checked += 1
        missing = check_script_header(script_file)
        if missing:
            issues_found.append(f"❌ {script_file.name}: {', '.join(missing)}")
        else:
            print(f"   ✅ {script_file.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files checked: {files_checked}")
    print(f"Issues found: {len(issues_found)}")
    
    if issues_found:
        print("\n❌ ISSUES TO FIX:")
        for issue in issues_found:
            print(f"   {issue}")
        print("\nAll source files must have proper copyright headers!")
        print("Required elements:")
        print("  • Copyright (c) 2025 Greg D. Partin")
        print("  • All rights reserved")
        print("  • CC BY-NC-ND 4.0 license")
        print("  • Commercial use prohibition")
        print("  • Contact email")
        return 1
    else:
        print("\n✅ ALL FILES HAVE PROPER COPYRIGHT HEADERS!")
        print("\nIP protection compliance: VERIFIED")
        return 0

if __name__ == "__main__":
    sys.exit(main())