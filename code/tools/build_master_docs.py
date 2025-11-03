#!/usr/bin/env python3
"""
Build the Master document (DOCX/PDF) for upload from canonical Markdown sources.
- Concatenate docs/Executive_Summary.md, LFM_Master.md, LFM_Core_Equations.md, LFM_Phase1_Test_Design.md
- Append docs/upload/RESULTS_REPORT.md (if present)
- Produce docs/upload/LFM_Master_<YYYYMMDD>_v1.docx and .pdf via pandoc when available
- Always write docs/upload/_LFM_Master_combined.md as the canonical combined source

Usage:
  python tools/build_master_docs.py

Notes:
- Requires Pandoc on PATH to produce DOCX/PDF. If absent, only the combined .md is produced.
- You can install Pandoc from https://pandoc.org/install.html
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil
import os

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
TEXT = DOCS / 'text'
UPLOAD = DOCS / 'upload'

# Deterministic build controls
DETERMINISTIC = os.environ.get('LFM_DETERMINISTIC', '0') == '1'

def _deterministic_date_str() -> str:
    sde = os.environ.get('SOURCE_DATE_EPOCH')
    if sde and sde.isdigit():
        try:
            return datetime.utcfromtimestamp(int(sde)).strftime('%Y%m%d')
        except Exception:
            pass
    return '19700101'

TODAY = _deterministic_date_str() if DETERMINISTIC else datetime.now().strftime('%Y%m%d')
OUT_BASENAME = f'LFM_Master_{TODAY}_v1'

MD_SOURCE_FILES = [
    DOCS / 'Executive_Summary.md',
    DOCS / 'LFM_Master.md',
    DOCS / 'LFM_Core_Equations.md',
    DOCS / 'LFM_Phase1_Test_Design.md',
]
TEXT_SOURCE_FILES = [
    TEXT / 'Executive_Summary.txt',
    TEXT / 'LFM_Master.txt',
    TEXT / 'LFM_Core_Equations.txt',
    TEXT / 'LFM_Phase1_Test_Design.txt',
]
RESULTS_REPORT = UPLOAD / 'RESULTS_REPORT.md'


def read_text_if_exists(p: Path) -> str:
    return p.read_text(encoding='utf-8') if p.exists() else ''


def build_combined_markdown() -> Path:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    parts = []
    # Cover
    parts.append('# Lorentzian Field Model (LFM) — Master Document')
    parts.append('')
    if DETERMINISTIC:
        parts.append('Generated: 1970-01-01 00:00:00Z')
    else:
        parts.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    parts.append('License: CC BY-NC-ND 4.0 — Non-commercial use; no derivatives')
    parts.append('Related: OSF https://osf.io/6agn8 · Zenodo https://zenodo.org/records/17478758')
    parts.append('')

    # Body sections (prefer canonical plain-text in docs/text/, fallback to Markdown in docs/)
    paired = list(zip(TEXT_SOURCE_FILES, MD_SOURCE_FILES))
    for txt, md in paired:
        if txt.exists():
            parts.append(f'\n\n<!-- Source: {txt.relative_to(ROOT)} -->\n\n')
            parts.append(read_text_if_exists(txt))
        elif md.exists():
            parts.append(f'\n\n<!-- Source: {md.relative_to(ROOT)} -->\n\n')
            parts.append(read_text_if_exists(md))
        else:
            parts.append(f'\n\n> NOTE: Missing source files: {txt} and {md}\n\n')

    # Append results report if present
    if RESULTS_REPORT.exists():
        parts.append('\n\n---\n\n')
        parts.append('# Aggregated Results Report (excerpt)')
        rr = read_text_if_exists(RESULTS_REPORT)
        parts.append(rr)

    combined = '\n'.join(parts)
    out_md = UPLOAD / '_LFM_Master_combined.md'
    out_md.write_text(combined, encoding='utf-8')
    return out_md


def find_pandoc() -> Path | None:
    """Return path to pandoc executable if found on PATH or default install location."""
    # First try PATH
    exe = shutil.which('pandoc')
    if exe:
        return Path(exe)
    # Try common Windows install paths
    for candidate in [
        Path('C:/Program Files/Pandoc/pandoc.exe'),
        Path('C:/Program Files (x86)/Pandoc/pandoc.exe'),
        Path.home() / 'AppData/Local/Pandoc/pandoc.exe',
    ]:
        if candidate.exists():
            return candidate
    return None


def run_pandoc(pandoc_exe: Path, md_in: Path, out_path: Path):
    args = [
        str(pandoc_exe), str(md_in), '-o', str(out_path),
        '--from', 'markdown', '--toc', '--standalone'
    ]
    subprocess.run(args, check=True)


def main():
    md = build_combined_markdown()
    pandoc_exe = find_pandoc()
    if pandoc_exe is None:
        print('Pandoc not found on PATH; wrote combined markdown only:', md)
        print('Install Pandoc to produce DOCX/PDF: https://pandoc.org/install.html')
        sys.exit(0)

    # In deterministic mode, skip DOCX/PDF generation to avoid non-deterministic metadata
    if DETERMINISTIC:
        print('Deterministic mode enabled; skipping DOCX/PDF build. Wrote combined markdown only:', md)
        sys.exit(0)

    # Produce DOCX and PDF
    docx_out = UPLOAD / f'{OUT_BASENAME}.docx'
    pdf_out = UPLOAD / f'{OUT_BASENAME}.pdf'
    try:
        run_pandoc(pandoc_exe, md, docx_out)
        print('Wrote', docx_out)
    except subprocess.CalledProcessError as e:
        print('Error building DOCX via pandoc:', e)

    try:
        run_pandoc(pandoc_exe, md, pdf_out)
        print('Wrote', pdf_out)
    except subprocess.CalledProcessError as e:
        print('Error building PDF via pandoc:', e)
        # Fallback: try docx2pdf if available
        try:
            from docx2pdf import convert as docx2pdf_convert
            # Convert the just-built DOCX to PDF in place
            docx2pdf_convert(str(docx_out), str(pdf_out))
            print('Fallback via docx2pdf succeeded:', pdf_out)
        except Exception as fe:
            print('Fallback PDF conversion failed (docx2pdf):', fe)

if __name__ == '__main__':
    main()
