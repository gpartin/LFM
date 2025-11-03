#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Build a comprehensive PDF combining:
- All governing DOCX contents (Executive Summary, Master, Core Equations, Phase 1 Test Design)
- Test results rollup from MASTER_TEST_STATUS.csv
- Tier descriptions
- Per-test descriptions with pass/fail status

Usage:
  python tools/build_comprehensive_pdf.py

Output: docs/upload/LFM_Comprehensive_Report_<YYYYMMDD>.pdf
"""
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
UPLOAD = DOCS / 'upload'
RESULTS = ROOT / 'results'
EVIDENCE = DOCS / 'evidence' / 'docx'

TODAY = datetime.now().strftime('%Y%m%d')
OUT_BASENAME = f'LFM_Comprehensive_Report_{TODAY}'


def find_pandoc() -> Path | None:
    exe = shutil.which('pandoc')
    if exe:
        return Path(exe)
    for candidate in [
        Path('C:/Program Files/Pandoc/pandoc.exe'),
        Path('C:/Program Files (x86)/Pandoc/pandoc.exe'),
        Path.home() / 'AppData/Local/Pandoc/pandoc.exe',
    ]:
        if candidate.exists():
            return candidate
    return None


def read_master_status() -> str:
    """Read and format the master test status CSV."""
    csv_path = RESULTS / 'MASTER_TEST_STATUS.csv'
    if not csv_path.exists():
        return '> Master test status not available.\n'
    lines = ['# Test Results Rollup', '']
    txt = csv_path.read_text(encoding='utf-8')
    lines.append('```')
    lines.append(txt)
    lines.append('```')
    return '\n'.join(lines)


def read_tier_descriptions() -> str:
    """Read tier and per-test descriptions from results directories."""
    lines = ['# Tier and Test Descriptions', '']
    # Presentation overrides (optional): tests to be shown as SKIP and extra notes
    overrides_path = ROOT / 'config' / 'presentation_overrides.json'
    overrides = {}
    try:
        if overrides_path.exists():
            overrides = json.loads(overrides_path.read_text(encoding='utf-8'))
    except Exception:
        overrides = {}
    skip_overrides = set(overrides.get('skip_tests', []))
    note_overrides = overrides.get('notes', {})
    
    tier_dirs = {
        'Relativistic': 'Tier 1 — Relativistic (Lorentz invariance, isotropy, causality)',
        'Gravity': 'Tier 2 — Gravity Analogue (χ-field gradients, redshift, lensing)',
        'Energy': 'Tier 3 — Energy Conservation (Hamiltonian partitioning, dissipation)',
        'Quantization': 'Tier 4 — Quantization (Discrete exchange, spectral linearity, uncertainty)',
    }
    
    for tier_name, tier_desc in tier_dirs.items():
        tier_dir = RESULTS / tier_name
        if not tier_dir.exists():
            continue
        lines.append(f'## {tier_desc}')
        lines.append('')
        
        # Read test directories and their readme.txt or summary.json
        for test_dir in sorted(tier_dir.glob('*')):
            if not test_dir.is_dir():
                continue
            test_id = test_dir.name
            
            # Try to read description from readme.txt or summary.json
            readme = test_dir / 'readme.txt'
            summary_json = test_dir / 'summary.json'
            
            desc = 'No description available'
            status = 'UNKNOWN'
            
            if summary_json.exists():
                try:
                    data = json.loads(summary_json.read_text(encoding='utf-8'))
                    desc = data.get('description', desc)
                    # Handle various status formats
                    if data.get('skipped') is True:
                        status_raw = 'SKIP'
                    elif 'status' in data:
                        status_raw = data['status']
                    elif 'passed' in data:
                        status_raw = 'PASS' if data['passed'] else 'FAIL'
                    else:
                        status_raw = 'UNKNOWN'
                    
                    # Normalize status to uppercase
                    status_upper = str(status_raw).upper()
                    if status_upper in ['PASSED', 'PASS', 'TRUE']:
                        status = 'PASS'
                    elif status_upper in ['FAILED', 'FAIL', 'FALSE']:
                        status = 'FAIL'
                    elif status_upper in ['SKIPPED', 'SKIP']:
                        status = 'SKIP'
                    else:
                        status = status_raw
                    # Apply presentation overrides
                    if test_id in skip_overrides:
                        status = 'SKIP'
                        if note_overrides.get(test_id):
                            desc += f" (Skipped: {note_overrides.get(test_id)})"
                    # Include skip reason if provided
                    if status == 'SKIP' and data.get('skip_reason'):
                        desc += f" (Skipped: {data.get('skip_reason')})"
                except Exception:
                    pass
            
            if readme.exists():
                try:
                    txt = readme.read_text(encoding='utf-8')
                    # Extract description from readme
                    for line in txt.splitlines():
                        if line.startswith('- description:'):
                            desc = line.split(':', 1)[1].strip()
                            break
                except Exception:
                    pass
            
            lines.append(f'### {test_id}: {desc}')
            lines.append(f'**Status:** {status}')
            lines.append('')
    
    return '\n'.join(lines)


def convert_docx_to_md(docx_path: Path, pandoc_exe: Path) -> str:
    """Convert a DOCX to Markdown text using Pandoc."""
    try:
        result = subprocess.run(
            [str(pandoc_exe), str(docx_path), '-t', 'markdown'],
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            check=True
        )
        return result.stdout
    except Exception as e:
        return f'> Error converting {docx_path.name}: {e}\n'


def build_combined_markdown(pandoc_exe: Path) -> Path:
    """Build a comprehensive Markdown combining all sources."""
    UPLOAD.mkdir(parents=True, exist_ok=True)
    parts = []
    
    # Cover
    parts.append('# LFM Comprehensive Report')
    parts.append('')
    parts.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    parts.append('License: CC BY-NC-ND 4.0 — Non-commercial use; no derivatives')
    parts.append('')
    parts.append('This document combines:')
    parts.append('- Governing documents (Executive Summary, Master, Core Equations, Phase 1 Test Design)')
    parts.append('- Test results rollup')
    parts.append('- Tier and per-test descriptions with pass/fail status')
    parts.append('')
    parts.append('---')
    parts.append('')
    
    # Use canonical text sources instead of DOCX files
    text_sources = [
        ('Executive_Summary.txt', 'Executive Summary'),
        ('LFM_Master.txt', 'Master Document'),
        ('LFM_Core_Equations.txt', 'Core Equations'),
        ('LFM_Phase1_Test_Design.txt', 'Phase 1 Test Design'),
    ]
    
    text_dir = ROOT / 'docs' / 'text'
    
    for filename, title in text_sources:
        text_path = text_dir / filename
        if text_path.exists():
            parts.append(f'# {title}')
            parts.append('')
            # Read the text content directly
            text_content = text_path.read_text(encoding='utf-8')
            parts.append(text_content)
            parts.append('')
            parts.append('---')
            parts.append('')
        else:
            parts.append(f'# {title}')
            parts.append('')
            parts.append(f'> Error: Source file {filename} not found')
            parts.append('')
            parts.append('---')
            parts.append('')
    
    # Test results rollup
    parts.append(read_master_status())
    parts.append('')
    parts.append('---')
    parts.append('')
    
    # Tier and test descriptions
    parts.append(read_tier_descriptions())
    
    combined = '\n'.join(parts)
    out_md = UPLOAD / f'{OUT_BASENAME}.md'
    out_md.write_text(combined, encoding='utf-8')
    return out_md


def main():
    pandoc_exe = find_pandoc()
    if pandoc_exe is None:
        print('ERROR: Pandoc not found. Install from https://pandoc.org/install.html')
        sys.exit(1)
    
    print('Building comprehensive Markdown...')
    md = build_combined_markdown(pandoc_exe)
    print(f'Wrote {md}')
    
    # Build PDF from the combined Markdown
    pdf_out = UPLOAD / f'{OUT_BASENAME}.pdf'
    print('Generating PDF...')
    try:
        subprocess.run(
            [str(pandoc_exe), str(md), '-o', str(pdf_out),
             '--from', 'markdown', '--toc', '--standalone',
             '--pdf-engine=xelatex'],  # Try xelatex first
            check=True
        )
        print(f'SUCCESS: {pdf_out}')
    except subprocess.CalledProcessError:
        print('xelatex failed, trying fallback via DOCX...')
        # Fallback: build DOCX first, then convert to PDF
        docx_tmp = UPLOAD / f'{OUT_BASENAME}.docx'
        try:
            subprocess.run(
                [str(pandoc_exe), str(md), '-o', str(docx_tmp),
                 '--from', 'markdown', '--toc', '--standalone'],
                check=True
            )
            print(f'Created DOCX: {docx_tmp}')
            # Try docx2pdf
            try:
                from docx2pdf import convert as docx2pdf_convert
                docx2pdf_convert(str(docx_tmp), str(pdf_out))
                print(f'SUCCESS via docx2pdf: {pdf_out}')
            except Exception as e:
                print(f'docx2pdf failed: {e}')
                print(f'PDF generation failed. DOCX available at {docx_tmp}')
        except Exception as e:
            print(f'DOCX generation failed: {e}')
            sys.exit(1)


if __name__ == '__main__':
    main()
