#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PDFs from core markdown documents for OSF/Zenodo uploads.

Converts markdown files to PDF using pandoc with xelatex engine.
Generates PDFs for:
- Executive_Summary.md
- LFM_Core_Equations.md  
- LFM_Master.md
- LFM_Phase1_Test_Design.md

Usage:
    python tools/generate_core_pdfs.py [--dest {osf|zenodo|both}]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Project root and paths
ROOT = Path(__file__).resolve().parent.parent
DEST_OSF = ROOT / 'uploads' / 'osf'
DEST_ZENODO = ROOT / 'uploads' / 'zenodo'

# Core documents to convert (MD -> PDF)
CORE_DOCS = [
    'EXECUTIVE_SUMMARY.md',
    'CORE_EQUATIONS.md',
    'MASTER_DOCUMENT.md',
    'TEST_DESIGN.md',
]

# Output PDF names
PDF_NAMES = {
    'EXECUTIVE_SUMMARY.md': 'Executive_Summary.pdf',
    'CORE_EQUATIONS.md': 'LFM_Core_Equations.pdf',
    'MASTER_DOCUMENT.md': 'LFM_Master.pdf',
    'TEST_DESIGN.md': 'LFM_Phase1_Test_Design.pdf',
}


def check_pandoc() -> bool:
    """Check if pandoc with xelatex is available."""
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ pandoc not found. Install from https://pandoc.org/")
            return False
        
        result = subprocess.run(['xelatex', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ xelatex not found. Install TeX Live or MiKTeX.")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error checking PDF tools: {e}")
        return False


def generate_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Generate PDF from markdown using pandoc with xelatex."""
    try:
        print(f"  Converting: {md_path.name} → {pdf_path.name}")
        
        # Use pandoc with xelatex engine for better formatting
        cmd = [
            'pandoc',
            str(md_path),
            '-o', str(pdf_path),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'linestretch=1.2',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
        
        if result.returncode != 0:
            print(f"    ❌ pandoc failed:")
            if result.stderr:
                print(f"    {result.stderr[:500]}")
            return False
        
        if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
            print(f"    ❌ PDF too small or missing")
            return False
            
        print(f"    ✅ Generated: {pdf_path.stat().st_size:,} bytes")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"    ❌ Timeout generating PDF")
        return False
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate PDFs from core markdown documents')
    parser.add_argument('--dest', choices=['osf', 'zenodo', 'both'], default='both',
                       help='Destination(s) for generated PDFs')
    args = parser.parse_args()
    
    print("PDF Generator for LFM Core Documents")
    print("=" * 50)
    
    # Check for required tools
    if not check_pandoc():
        print("\n❌ Required PDF generation tools not found")
        return 1
    
    print("✅ pandoc and xelatex available\n")
    
    # Determine destination directories
    destinations: List[Path] = []
    if args.dest in ['osf', 'both']:
        destinations.append(DEST_OSF)
    if args.dest in ['zenodo', 'both']:
        destinations.append(DEST_ZENODO)
    
    # Generate PDFs
    success_count = 0
    failure_count = 0
    
    for dest in destinations:
        print(f"Target: {dest.relative_to(ROOT)}/")
        print("-" * 50)
        
        if not dest.exists():
            print(f"  ⚠️  Directory does not exist: {dest}")
            continue
        
        for md_name in CORE_DOCS:
            md_path = dest / md_name
            pdf_name = PDF_NAMES[md_name]
            pdf_path = dest / pdf_name
            
            if not md_path.exists():
                print(f"  ⚠️  Source not found: {md_name}")
                failure_count += 1
                continue
            
            if generate_pdf(md_path, pdf_path):
                success_count += 1
            else:
                failure_count += 1
        
        print()
    
    # Summary
    print("=" * 50)
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failure_count}")
    
    if failure_count > 0:
        return 1
    
    print("\n✅ All PDFs generated successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
