#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Automated Header Fixer
======================
Automatically adds proper copyright headers to LFM source files.
Ensures IP protection compliance across the entire codebase.
"""

import os
import sys
from pathlib import Path

# Standard copyright header for Python files
PYTHON_HEADER = '''#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

'''

# Standard copyright header for Markdown files
MARKDOWN_HEADER = '''<!-- Copyright (c) 2025 Greg D. Partin. All rights reserved. -->
<!-- Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International). -->
<!-- See LICENSE file in project root for full license text. -->
<!-- Commercial use prohibited without explicit written permission. -->
<!-- Contact: latticefieldmediumresearch@gmail.com -->

'''

def has_full_copyright_header(content):
    """Check if content already has full copyright header"""
    required_elements = [
        "Copyright (c) 2025 Greg D. Partin",
        "All rights reserved",
        "CC BY-NC-ND 4.0",
        "Commercial use prohibited",
        "latticefieldmediumresearch@gmail.com"
    ]
    
    # Check first 10 lines
    first_lines = '\n'.join(content.split('\n')[:10])
    
    for element in required_elements:
        if element not in first_lines:
            return False
    return True

def fix_python_file(filepath):
    """Fix Python file header"""
    print(f"   Fixing {filepath.name}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if has_full_copyright_header(content):
        print(f"   ✅ {filepath.name} already has full header")
        return False
    
    lines = content.split('\n')
    
    # Find where the actual code starts (after shebang and existing headers)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#!/usr/bin/python') or line.startswith('#!/usr/bin/env python'):
            start_idx = i + 1
            continue
        elif line.startswith('#') and ('copyright' in line.lower() or 'licensed' in line.lower()):
            # Skip existing copyright/license lines
            continue
        elif line.strip() == '' or line.startswith('#'):
            # Skip empty lines and comments at top
            continue
        else:
            # Found start of actual content
            start_idx = i
            break
    
    # Extract the content after headers
    remaining_content = '\n'.join(lines[start_idx:])
    
    # Write new file with proper header
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(PYTHON_HEADER)
        f.write(remaining_content)
    
    return True

def fix_markdown_file(filepath):
    """Fix Markdown file header"""
    print(f"   Fixing {filepath.name}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "Copyright (c) 2025 Greg D. Partin" in content[:500]:
        print(f"   ✅ {filepath.name} already has copyright notice")
        return False
    
    # Add header at the beginning
    with open(filepath, 'w', encoding='utf-8') as f:
        # Keep the first line (title) if it starts with #
        lines = content.split('\n')
        if lines and lines[0].startswith('#'):
            f.write(lines[0] + '\n\n')
            f.write(MARKDOWN_HEADER)
            f.write('\n'.join(lines[1:]))
        else:
            f.write(MARKDOWN_HEADER)
            f.write(content)
    
    return True

def main():
    """Main header fixing process"""
    print("=" * 60)
    print("  LFM COPYRIGHT HEADER FIXER")
    print("=" * 60)
    print()
    
    code_dir = Path(__file__).parent
    files_fixed = 0
    
    # Fix Python files that need headers
    files_to_fix = [
        'tools/build_comprehensive_pdf.py',
        'tools/build_master_docs.py', 
        'tools/build_upload_package.py',
        'tools/check_contact_email.py',
        'tools/compile_results_report.py',
        'tools/diagnostics_policy.py',
        'tools/dirhash_compare.py',
        'tools/docx_text_extract.py',
        'tools/docx_to_pdf.py',
        'tools/generate_results_readmes.py',
        'tools/post_run_hooks.py',
        'tools/validate_results_pipeline.py',
        'archive/add_copyright_headers.py',
        'archive/check_contact_email.py',
        'archive/fix_license_headers.py',
        'archive/replace_contact_email.py'
    ]
    
    print("🐍 Fixing Python files...")
    for file_path in files_to_fix:
        full_path = code_dir / file_path
        if full_path.exists():
            if fix_python_file(full_path):
                files_fixed += 1
        else:
            print(f"   ⚠️  File not found: {file_path}")
    
    # Fix Markdown files
    print("\n📝 Fixing Markdown files...")
    md_files_to_fix = [
        'PRE_PUBLIC_AUDIT_REPORT.md',
        'THIRD_PARTY_LICENSES.md'
    ]
    
    for file_path in md_files_to_fix:
        full_path = code_dir / file_path
        if full_path.exists():
            if fix_markdown_file(full_path):
                files_fixed += 1
        else:
            print(f"   ⚠️  File not found: {file_path}")
    
    print(f"\n✅ Fixed {files_fixed} files")
    print("\nRun 'python validate_headers.py' to verify all headers are now correct.")

if __name__ == "__main__":
    main()