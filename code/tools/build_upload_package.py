#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Build OSF/Zenodo upload package (dry-run)
- Refresh results artifacts (MASTER_TEST_STATUS.csv, RESULTS_REPORT.md)
- Stage required documents into docs/upload/
- Compute SHA256 checksums and sizes
- Generate MANIFEST.md (overwrites)
- Emit simple metadata templates for Zenodo/OSF (JSON/MD)

Usage (from repo root):
  python tools/build_upload_package.py

This is a dry-run: no network calls; outputs are staged under docs/upload/.
"""
import hashlib
import shutil
import argparse
import zipfile
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import platform

# Local imports (repo)
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from lfm_results import update_master_test_status  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
UPLOAD = ROOT / 'docs' / 'upload'

# Files we expect to stage (some may already exist in docs/upload)
CORE_DOCS = [
    'LICENSE',
    'NOTICE',
    'README.md',
    'README_FOR_REVIEWERS.md',
    'PRE_PUBLIC_AUDIT_REPORT.md',
    'RESULTS_REPORT.md',
    'evidence_prior_art.txt',
    '.zenodo_upload_checklist.md',
]

OPTIONAL_DOCS = [
    'LFM_Master_20251102_v1.pdf',
    'LFM_Master_20251102_v1.docx',
    'diagnostics_energy_drift.csv',
]

STAGED_GENERATED = [
    'results_MASTER_TEST_STATUS.csv',
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_now_str() -> str:
    sde = os.environ.get('SOURCE_DATE_EPOCH')
    if sde and sde.isdigit():
        try:
            return datetime.utcfromtimestamp(int(sde)).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
    return '1970-01-01 00:00:00'


def _deterministic_date_stamp() -> str:
    """Return YYYYMMDD from SOURCE_DATE_EPOCH or 19700101 as fallback."""
    sde = os.environ.get('SOURCE_DATE_EPOCH')
    if sde and sde.isdigit():
        try:
            return datetime.utcfromtimestamp(int(sde)).strftime('%Y%m%d')
        except Exception:
            pass
    return '19700101'


def _collect_provenance(deterministic: bool = False) -> list[str]:
    """Collect provenance: git SHA, Python, NumPy/CuPy, OS, deterministic flag."""
    lines: list[str] = []
    # Git SHA
    sha = None
    try:
        import subprocess as _sp
        proc = _sp.run(['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=3)
        if proc.returncode == 0:
            sha = proc.stdout.strip()
    except Exception:
        sha = None
    lines.append(f"- Git SHA: {sha or 'unknown'}")
    # Python & packages
    lines.append(f"- Python: {platform.python_version()}")
    try:
        import numpy as _np
        lines.append(f"- NumPy: {_np.__version__}")
    except Exception:
        lines.append("- NumPy: unavailable")
    try:
        import cupy as _cp  # type: ignore
        _ = _cp.__version__
        lines.append(f"- CuPy: {_cp.__version__}")
    except Exception:
        lines.append("- CuPy: unavailable")
    # OS
    lines.append(f"- OS: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"- Deterministic mode: {'on' if deterministic else 'off'}")
    return lines


def refresh_results_artifacts(deterministic: bool = False, build_master: bool = False):
    # Always rebuild master docs before staging, so DOCX/PDF are current
    master_builder = ROOT / 'tools' / 'build_master_docs.py'
    if build_master and master_builder.exists():
        import runpy
        try:
            # Propagate deterministic mode to master builder via env var
            if deterministic:
                os.environ['LFM_DETERMINISTIC'] = '1'
            runpy.run_path(str(master_builder), run_name='__main__')
        except SystemExit:
            # The builder may call sys.exit(); treat as normal completion
            pass
        except Exception as e:
            print(f"[WARN] Master docs build failed: {e}")

    # Ensure master status is up to date
    out = update_master_test_status(RESULTS)
    # Copy to docs/upload as results_MASTER_TEST_STATUS.csv
    target = UPLOAD / 'results_MASTER_TEST_STATUS.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(out).read_bytes())
    # Normalize timestamp line for determinism
    if deterministic and target.exists():
        try:
            txt = target.read_text(encoding='utf-8').splitlines()
            fixed = []
            for line in txt:
                if line.startswith('Generated: '):
                    fixed.append('Generated: ' + _deterministic_now_str())
                else:
                    fixed.append(line)
            target.write_text('\n'.join(fixed) + ('\n' if txt and not txt[-1].endswith('\n') else ''), encoding='utf-8')
        except Exception:
            pass

    # Compile RESULTS_REPORT.md if not present; otherwise leave as-is
    # Prefer existing tool if available
    compiler = ROOT / 'tools' / 'compile_results_report.py'
    if compiler.exists():
        # Execute compiler via Python to regenerate the report
        import runpy
        # Set sys.argv so defaults (results root and output md) apply
        sys_argv_bak = sys.argv
        try:
            sys.argv = [str(compiler), str(RESULTS), str(UPLOAD / 'RESULTS_REPORT.md')]
            runpy.run_path(str(compiler), run_name='__main__')
        finally:
            sys.argv = sys_argv_bak

    # Generate a plain-text README (README_LFM.txt) from docs/README.md when possible
    try:
        readme_src = ROOT / 'docs' / 'README.md'
        readme_txt = UPLOAD / 'README_LFM.txt'
        if readme_src.exists():
            _convert_md_to_txt(readme_src, readme_txt)
    except Exception:
        pass
    # Stage legal/IP docs into upload if present at repo root or docs/
    stage_legal_docs()
    # Note: We no longer stage canonical txt sources; .txt will be derived from evidence DOCX


def stage_evidence_docx(include: bool = False) -> List[str]:
    """Optionally stage original source DOCX files under upload/evidence_docx.

    Returns list of relative paths staged.
    """
    staged: List[str] = []
    if not include:
        return staged
    src_dir = ROOT / 'docs' / 'evidence' / 'docx'
    if not src_dir.exists():
        return staged
    dst_dir = UPLOAD / 'evidence_docx'
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob('*.docx'):
        dst = dst_dir / p.name
        try:
            shutil.copyfile(p, dst)
            staged.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
        except Exception:
            continue
    # Optional: write a short README about these artifacts
    readme = dst_dir / 'README.txt'
    if not readme.exists():
        readme.write_text(
            'Original source DOCX files for provenance and archival purposes.\n'
            'These mirror the canonical Markdown sources used to build the Master document.\n',
            encoding='utf-8'
        )
        staged.append(str(readme.relative_to(UPLOAD)).replace('\\', '/'))
    return staged


def export_txt_from_evidence(include: bool = True) -> List[str]:
    """Export .txt versions of key DOCX sources for archival (Executive_Summary, LFM_*).

    Writes to upload/txt/*.txt and returns relative paths added.
    """
    emitted: List[str] = []
    if not include:
        return emitted
    src_dir = UPLOAD / 'evidence_docx'
    if not src_dir.exists():
        return emitted
    txt_dir = UPLOAD / 'txt'
    txt_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(src_dir.glob('*.docx')):
        dst = txt_dir / (p.stem + '.txt')
        # Incremental: regenerate only if source is newer than destination
        try:
            if dst.exists() and (dst.stat().st_mtime >= p.stat().st_mtime):
                # Up to date
                continue
        except Exception:
            pass
        try:
            _convert_docx_to_txt(p, dst)
            emitted.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
        except Exception as e:
            print(f"[WARN] TXT export failed for {p.name}: {e}")
            continue
    return emitted
def _convert_docx_to_md(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    pandoc = _find_pandoc()
    if pandoc:
        import subprocess as _sp
        _sp.run([str(pandoc), str(src), '-t', 'gfm', '-o', str(dst)], check=True)
        return
    # Fallback: use plain text and wrap as code block markers minimal
    tmp_txt = dst.with_suffix('.txt.tmp')
    _convert_docx_to_txt(src, tmp_txt)
    txt = tmp_txt.read_text(encoding='utf-8')
    dst.write_text(txt, encoding='utf-8')
    try:
        tmp_txt.unlink()
    except Exception:
        pass


def export_md_from_evidence() -> List[str]:
    """Export .md versions of governing DOCX into upload/md and create a top-level Executive_Summary.md."""
    emitted: List[str] = []
    src_dir = UPLOAD / 'evidence_docx'
    if not src_dir.exists():
        return emitted
    md_dir = UPLOAD / 'md'
    md_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob('*.docx')):
        dst = md_dir / (p.stem + '.md')
        try:
            # Incremental: convert only if source newer than destination
            do_convert = True
            if dst.exists():
                try:
                    do_convert = dst.stat().st_mtime < p.stat().st_mtime
                except Exception:
                    do_convert = True
            if do_convert:
                _convert_docx_to_md(p, dst)
            emitted.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
        except Exception as e:
            print(f"[WARN] MD export failed for {p.name}: {e}")
            continue
        # Also create a top-level Executive_Summary.md for convenience
        if p.stem == 'Executive_Summary':
            top = UPLOAD / 'Executive_Summary.md'
            try:
                # Incremental: copy only if newer
                if (not top.exists()) or (dst.stat().st_mtime > top.stat().st_mtime):
                    shutil.copyfile(dst, top)
                emitted.append(str(top.relative_to(UPLOAD)).replace('\\', '/'))
            except Exception:
                pass
    return emitted


def stage_legal_docs() -> List[str]:
    """Copy legal/IP docs from repo (./ or ./docs) into upload.

    Returns list of relative paths staged.
    """
    staged: List[str] = []
    candidates = [
        (ROOT, 'LICENSE'),
        (ROOT, 'NOTICE'),
        (ROOT, 'PRE_PUBLIC_AUDIT_REPORT.md'),
        (ROOT / 'docs', 'README_FOR_REVIEWERS.md'),
        (ROOT / 'docs', '.zenodo_upload_checklist.md'),
        (ROOT / 'docs', 'README.md'),
        (ROOT / 'docs', 'evidence_prior_art.txt'),
    ]
    for base, name in candidates:
        src = base / name
        if src.exists():
            dst = UPLOAD / name
            try:
                shutil.copyfile(src, dst)
                staged.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
            except Exception:
                pass
    return staged


def _stage_canonical_txt_sources() -> List[str]:
    """Copy canonical docs/text/*.txt into docs/upload/txt/.

    Returns list of relative paths staged.
    """
    out: List[str] = []
    src_dir = ROOT / 'docs' / 'text'
    if not src_dir.exists():
        return out
    dst_dir = UPLOAD / 'txt'
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ['Executive_Summary.txt', 'LFM_Master.txt', 'LFM_Core_Equations.txt', 'LFM_Phase1_Test_Design.txt']:
        src = src_dir / name
        if src.exists():
            dst = dst_dir / name
            shutil.copyfile(src, dst)
            out.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
    return out


def validate_core_docs(strict: bool = False, deterministic: bool = False, build_master: bool = False) -> List[str]:
    """Validate presence and basic integrity of critical documents.

    Returns a list of warnings/errors. In strict mode, raises on errors.
    """
    issues: List[str] = []
    # Evidence DOCX set should exist (canonical sources): 4 governing documents
    required_docx = [
        UPLOAD / 'evidence_docx' / 'Executive_Summary.docx',
        UPLOAD / 'evidence_docx' / 'LFM_Master.docx',
        UPLOAD / 'evidence_docx' / 'LFM_Core_Equations.docx',
        UPLOAD / 'evidence_docx' / 'LFM_Phase1_Test_Design.docx',
    ]
    missing = [str(p.name) for p in required_docx if not p.exists()]
    if missing:
        issues.append('Missing evidence DOCX files: ' + ', '.join(missing))

    # Master DOCX/PDF expectations vary: in deterministic mode we may skip them
    if build_master and not deterministic:
        docx_name = None
        for p in UPLOAD.glob('LFM_Master_*_v1.docx'):
            docx_name = p.name
            break
        if not docx_name:
            issues.append('Master DOCX not found (LFM_Master_YYYYMMDD_v1.docx)')
        pdf_found = any(UPLOAD.glob('LFM_Master_*_v1.pdf'))
        if not pdf_found:
            issues.append('Master PDF not found; will rely on DOCX only')

    if strict and issues:
        raise RuntimeError('Upload validation failed: ' + '; '.join(issues))
    return issues


# ---------------------------- Converters (Pandoc-first) ----------------------------
def _find_pandoc() -> Path | None:
    import shutil as _sh
    exe = _sh.which('pandoc')
    if exe:
        return Path(exe)
    # Common Windows installs
    for candidate in [
        Path('C:/Program Files/Pandoc/pandoc.exe'),
        Path('C:/Program Files (x86)/Pandoc/pandoc.exe'),
        Path.home() / 'AppData/Local/Pandoc/pandoc.exe',
    ]:
        if candidate.exists():
            return candidate
    return None


def _convert_docx_to_txt(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    pandoc = _find_pandoc()
    if pandoc:
        import subprocess as _sp
        _sp.run([str(pandoc), str(src), '-t', 'plain', '-o', str(dst)], check=True)
        return
    # Fallback: very simple extraction using python-docx if installed
    try:
        from docx import Document  # type: ignore
        doc = Document(str(src))
        text = []
        for p in doc.paragraphs:
            text.append(p.text)
        dst.write_text('\n'.join(text), encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Cannot convert {src.name} to txt: {e}")


def _convert_md_to_txt(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    pandoc = _find_pandoc()
    if pandoc:
        import subprocess as _sp
        _sp.run([str(pandoc), str(src), '-t', 'plain', '-o', str(dst)], check=True)
        return
    # Fallback: naive markdown strip
    txt = src.read_text(encoding='utf-8')
    # Remove common markdown markers
    import re as _re
    txt = _re.sub(r"`{1,3}.*?`{1,3}", "", txt, flags=_re.S)
    txt = _re.sub(r"^#+\\s*", "", txt, flags=_re.M)
    txt = _re.sub(r"\*\*|__|\*|_", "", txt)
    dst.write_text(txt, encoding='utf-8')


def stage_and_list_files() -> List[Tuple[str, int, str]]:
    """Return list of (relative_path_under_upload, size, sha256)."""
    entries: List[Tuple[str, int, str]] = []

    # Ensure upload dir exists
    UPLOAD.mkdir(parents=True, exist_ok=True)

    # Collect core and optional docs if present
    for name in CORE_DOCS + OPTIONAL_DOCS + STAGED_GENERATED:
        p = UPLOAD / name
        if p.exists() and p.is_file():
            entries.append((name, p.stat().st_size, sha256_file(p)))

    # Include any PDF files at the root level (comprehensive reports, etc.)
    for pdf_file in UPLOAD.glob('*.pdf'):
        if pdf_file.is_file():
            name = pdf_file.name
            if name not in [doc for doc in CORE_DOCS + OPTIONAL_DOCS]:
                entries.append((name, pdf_file.stat().st_size, sha256_file(pdf_file)))

    # Include any DOCX files at the root level (intermediate comprehensive reports)
    for docx_file in UPLOAD.glob('*.docx'):
        if docx_file.is_file():
            name = docx_file.name
            if name not in [doc for doc in CORE_DOCS + OPTIONAL_DOCS]:
                entries.append((name, docx_file.stat().st_size, sha256_file(docx_file)))

    # Include any MD files at the root level (intermediate comprehensive reports)
    for md_file in UPLOAD.glob('*.md'):
        if md_file.is_file():
            name = md_file.name
            if name not in [doc for doc in CORE_DOCS + OPTIONAL_DOCS + STAGED_GENERATED]:
                entries.append((name, md_file.stat().st_size, sha256_file(md_file)))

    # Include any staged evidence DOCX files (if present)
    evidence_dir = UPLOAD / 'evidence_docx'
    if evidence_dir.exists() and evidence_dir.is_dir():
        for f in sorted(evidence_dir.rglob('*')):
            if f.is_file():
                rel = str(f.relative_to(UPLOAD)).replace('\\', '/')
                entries.append((rel, f.stat().st_size, sha256_file(f)))

    # Include any staged canonical TXT files (if present)
    txt_dir = UPLOAD / 'txt'
    if txt_dir.exists() and txt_dir.is_dir():
        for f in sorted(txt_dir.rglob('*')):
            if f.is_file():
                rel = str(f.relative_to(UPLOAD)).replace('\\', '/')
                entries.append((rel, f.stat().st_size, sha256_file(f)))

    # Include any staged MD files (if present)
    md_dir = UPLOAD / 'md'
    if md_dir.exists() and md_dir.is_dir():
        for f in sorted(md_dir.rglob('*')):
            if f.is_file():
                rel = str(f.relative_to(UPLOAD)).replace('\\', '/')
                entries.append((rel, f.stat().st_size, sha256_file(f)))

    # Include plots (if present)
    plots_dir = UPLOAD / 'plots'
    if plots_dir.exists() and plots_dir.is_dir():
        for f in sorted(plots_dir.rglob('*')):
            if f.is_file():
                rel = str(f.relative_to(UPLOAD)).replace('\\', '/')
                entries.append((rel, f.stat().st_size, sha256_file(f)))

    return sorted(entries, key=lambda x: x[0].lower())


def write_manifest(entries: List[Tuple[str, int, str]], deterministic: bool = False):
    now = _deterministic_now_str() if deterministic else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines: List[str] = []
    lines.append('# Upload Manifest')
    lines.append('')
    lines.append(f'- Generated: {now}')
    lines.append('- Scope: OSF/Zenodo dry-run package (docs/upload)')
    lines.append('')
    lines.append('## Provenance')
    lines.extend(_collect_provenance(deterministic))
    lines.append('')
    lines.append('| File | Size (bytes) | SHA256 |')
    lines.append('|------|--------------:|--------|')
    for rel, size, digest in entries:
        lines.append(f'| {rel} | {size} | `{digest}` |')
    lines.append('')
    lines.append('Notes:')
    lines.append('- MASTER_TEST_STATUS is refreshed from results/.')
    lines.append('- RESULTS_REPORT.md is compiled from results/ readmes and summaries.')
    lines.append('- This is a dry-run; verify placeholders (master PDF/DOCX) before upload.')
    (UPLOAD / 'MANIFEST.md').write_text('\n'.join(lines), encoding='utf-8')


def write_zenodo_metadata(entries: List[Tuple[str, int, str]], deterministic: bool = False):
    pub_date = _deterministic_now_str().split(' ')[0] if deterministic else datetime.now().strftime('%Y-%m-%d')
    meta = {
        "title": "Lorentzian Field Model (LFM) — Results Package",
        "upload_type": "dataset",
        "publication_date": pub_date,
        "creators": [
            {"name": "Partin, Greg D.", "affiliation": "Independent"}
        ],
        "description": (
            "Consolidated results and diagnostics for the LFM lattice field model. "
            "Includes MASTER_TEST_STATUS, aggregated report, and legal documents."
        ),
        "access_right": "open",
        "license": "cc-by-nc-nd-4.0",
        "keywords": ["Lorentzian Field Model", "lattice", "simulation", "energy", "quantization"],
        "related_identifiers": [
            {"relation": "isSupplementTo", "identifier": "https://osf.io/6agn8"},
            {"relation": "isSupplementTo", "identifier": "https://zenodo.org/records/17478758"}
        ],
        "notes": "Dry-run metadata generated locally; review before deposition."
    }
    (UPLOAD / 'zenodo_metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')


def write_osf_metadata(entries: List[Tuple[str, int, str]]):
    # OSF often done manually; include a helper JSON + MD overview.
    meta = {
        "title": "LFM Results Package (dry-run)",
        "category": "data",
        "tags": ["LFM", "lattice", "results"],
        "description": "Staged documents for OSF upload; generated locally.",
        "license": "CC BY-NC-ND 4.0"
    }
    (UPLOAD / 'osf_metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')

    md = [
        '# OSF Upload Overview',
        '',
        'This is a dry-run staging area. Upload the following files to OSF:',
        '',
        '| File | Size (bytes) | SHA256 |',
        '|------|--------------:|--------|',
    ]
    for rel, size, digest in entries:
        md.append(f'| {rel} | {size} | `{digest}` |')
    (UPLOAD / 'OSF_UPLOAD_OVERVIEW.md').write_text('\n'.join(md), encoding='utf-8')


def stage_result_plots(limit_per_dir: int = 6) -> List[str]:
    """Collect .png plots from results/**/plots and stage to upload/plots mirroring substructure.

    limit_per_dir limits the number of PNGs per source plot directory to keep package concise.
    Returns list of relative paths staged.
    """
    staged: List[str] = []
    src_root = ROOT / 'results'
    dst_root = UPLOAD / 'plots'
    if not src_root.exists():
        return staged
    for plot_dir in sorted(src_root.rglob('plots')):
        if not plot_dir.is_dir():
            continue
        # mirror path under upload/plots
        rel_parent = plot_dir.parent.relative_to(src_root)
        out_dir = dst_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for png in sorted(plot_dir.glob('*.png')):
            dst = out_dir / png.name
            try:
                shutil.copyfile(png, dst)
                staged.append(str(dst.relative_to(UPLOAD)).replace('\\', '/'))
                count += 1
                if count >= limit_per_dir:
                    break
            except Exception:
                continue
    # Write a simple overview index
    if staged:
        lines = ['# Plots Overview', '', 'Staged representative PNG plots from results/ (limited per directory).', '']
        for rel in staged:
            if rel.lower().endswith('.png'):
                lines.append(f'- ![{rel}]({rel})')
        (UPLOAD / 'PLOTS_OVERVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    return staged


def create_zip_bundle(entries: List[Tuple[str, int, str]], label: str | None = None, deterministic: bool = False) -> Tuple[str, int, str]:
    """Create a ZIP bundle containing the payload entries only.

    Returns (relative_path_under_upload, size, sha256) for the created zip.
    """
    # Determine bundle name
    stamp = _deterministic_date_stamp() if deterministic else datetime.now().strftime('%Y%m%d')
    base = f"LFM_upload_bundle_{stamp}_v1" if not label else f"{label}_{stamp}_v1"
    zip_rel = f"{base}.zip"
    zip_path = UPLOAD / zip_rel

    # Write zip
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Sort entries again to ensure deterministic ordering regardless of caller
        for rel, _size, _digest in sorted(entries, key=lambda x: x[0].lower()):
            src = UPLOAD / rel
            if not src.exists():
                continue
            if deterministic:
                # Write with fixed timestamp for reproducibility
                zi = zipfile.ZipInfo(rel)
                zi.date_time = (1980, 1, 1, 0, 0, 0)
                zi.compress_type = zipfile.ZIP_DEFLATED
                data = src.read_bytes()
                zf.writestr(zi, data)
            else:
                zf.write(src, arcname=rel)

    # Compute size and sha256
    size = zip_path.stat().st_size
    digest = sha256_file(zip_path)
    return (zip_rel, size, digest)


def generate_comprehensive_pdf() -> str | None:
    """Generate comprehensive PDF combining all governing docs and test results.
    
    Returns the relative path to the PDF under upload/ if successful, None otherwise.
    """
    script_path = ROOT / 'tools' / 'build_comprehensive_pdf.py'
    if not script_path.exists():
        print('[WARN] build_comprehensive_pdf.py not found; skipping comprehensive PDF')
        return None
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            encoding='utf-8',
            check=True
        )
        # Look for the generated PDF
        for line in result.stdout.splitlines():
            if line.startswith('SUCCESS') and '.pdf' in line:
                # Extract path from success message: "SUCCESS via docx2pdf: <path>"
                parts = line.split(':')
                if len(parts) >= 2:
                    pdf_path_str = ':'.join(parts[1:]).strip()  # Handle Windows paths with drive letters
                    pdf_path = Path(pdf_path_str)
                    if pdf_path.exists() and pdf_path.is_relative_to(UPLOAD):
                        rel = pdf_path.relative_to(UPLOAD)
                        return str(rel).replace('\\', '/')
        return None
    except Exception as e:
        print(f'[WARN] Comprehensive PDF generation failed: {e}')
        return None


def verify_manifest(strict: bool = False) -> tuple[int, int]:
    """Verify MANIFEST.md hashes against current files in docs/upload.

    Returns (checked, mismatched). In strict mode, raises on mismatches.
    """
    manifest = UPLOAD / 'MANIFEST.md'
    if not manifest.exists():
        print('[WARN] MANIFEST.md not found; nothing to verify')
        return (0, 0)
    text = manifest.read_text(encoding='utf-8').splitlines()
    checked = 0
    mismatched = 0
    ignored = {"MANIFEST.md", "OSF_UPLOAD_OVERVIEW.md"}
    for line in text:
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.strip('|').split('|')]
        if len(parts) != 3:
            continue
        rel, size_str, digest_md = parts
        # digest is wrapped in backticks in manifest
        if digest_md.startswith('`') and digest_md.endswith('`'):
            digest_md = digest_md[1:-1]
        # Skip volatile self-referential or post-generated files
        if rel in ignored:
            continue
        rel_path = (UPLOAD / rel).resolve()
        if not rel_path.exists() or not rel_path.is_file():
            # Non-file entries (headers) or missing files: skip quietly
            continue
        try:
            digest_now = sha256_file(rel_path)
            size_now = rel_path.stat().st_size
            size_ok = True
            try:
                size_ok = int(size_str) == size_now
            except Exception:
                size_ok = True  # don't fail on size parse; hash is authoritative
            if (digest_now != digest_md) or (not size_ok):
                mismatched += 1
                print(f"[ERROR] MANIFEST mismatch for {rel}: size {size_str} vs {size_now}, sha {digest_md} vs {digest_now}")
            checked += 1
        except Exception as e:
            mismatched += 1
            checked += 1
            print(f"[ERROR] Verification failed for {rel}: {e}")
    if strict and mismatched:
        raise RuntimeError(f"Manifest verification failed: {mismatched} mismatched of {checked} checked")
    return (checked, mismatched)


def generate_comprehensive_pdf() -> str | None:
    """Generate comprehensive PDF using the separate build_comprehensive_pdf.py tool.
    
    Returns relative path of generated PDF or None if generation failed.
    """
    tool_path = ROOT / 'tools' / 'build_comprehensive_pdf.py'
    if not tool_path.exists():
        print("[WARN] build_comprehensive_pdf.py tool not found")
        return None
        
    try:
        # Run the comprehensive PDF builder
        import runpy
        runpy.run_path(str(tool_path), run_name='__main__')
        
        # Find the generated PDF
        stamp = datetime.now().strftime('%Y%m%d')
        pdf_name = f'LFM_Comprehensive_Report_{stamp}.pdf'
        pdf_path = UPLOAD / pdf_name
        
        if pdf_path.exists():
            return pdf_name
        else:
            print(f"[WARN] Expected comprehensive PDF not found: {pdf_name}")
            return None
            
    except Exception as e:
        print(f"[WARN] Comprehensive PDF generation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Build OSF/Zenodo upload package (dry-run)')
    parser.add_argument('--no-zip', action='store_true', help='Do not create a ZIP bundle of payload files')
    parser.add_argument('--label', type=str, default=None, help='Custom label prefix for the ZIP bundle name')
    parser.add_argument('--include-evidence-docx', action='store_true', help='(Deprecated) Evidence DOCX are now always included')
    parser.add_argument('--no-txt', action='store_true', help='Do not generate .txt exports of evidence DOCX into upload/txt/')
    parser.add_argument('--strict', action='store_true', help='Fail if critical docs are missing or contain placeholders')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic outputs (fixed timestamps, reproducible zip, skip non-deterministic artifacts)')
    parser.add_argument('--build-master', action='store_true', help='Build a combined Master DOCX/PDF (non-deterministic)')
    parser.add_argument('--verify-manifest', action='store_true', help='After writing MANIFEST.md, verify all hashes and sizes against current files')
    parser.add_argument('--validate-pipeline', action='store_true', help='Run tools/validate_results_pipeline.py --all --quiet after staging')
    args = parser.parse_args()

    refresh_results_artifacts(deterministic=args.deterministic, build_master=args.build_master)
    # Always stage original DOCX evidence (governing documents)
    staged_evidence = stage_evidence_docx(include=True)
    if staged_evidence:
        print(f"Staged {len(staged_evidence)} evidence DOCX file(s)")
    # Export .txt mirrors (default on)
    if not args.no_txt:
        txt_list = export_txt_from_evidence(include=True)
        if txt_list:
            print(f"Exported {len(txt_list)} TXT file(s) from evidence DOCX")

    # Export .md mirrors from evidence DOCX and stage result plots
    md_list = export_md_from_evidence()
    if md_list:
        print(f"Exported {len(md_list)} MD file(s) from evidence DOCX")
    plots_list = stage_result_plots(limit_per_dir=6)
    if plots_list:
        print(f"Staged {len(plots_list)} PNG plot(s) from results")

    # Generate comprehensive PDF combining all governing docs and test results
    if not args.deterministic:
        pdf_rel = generate_comprehensive_pdf()
        if pdf_rel:
            print(f"Generated comprehensive PDF: {pdf_rel}")

    issues = validate_core_docs(strict=args.strict, deterministic=args.deterministic)
    for msg in issues:
        print('[WARN]', msg)
    entries = stage_and_list_files()

    # Optionally add ZIP bundle of payload files (default: create it)
    zip_info: Tuple[str, int, str] | None = None
    if not args.no_zip:
        zip_info = create_zip_bundle(entries, label=args.label, deterministic=args.deterministic)
        # Append zip to published entries for manifest/overview convenience
        entries_with_zip = entries + [zip_info]
    else:
        entries_with_zip = entries

    write_manifest(entries_with_zip, deterministic=args.deterministic)
    write_zenodo_metadata(entries_with_zip, deterministic=args.deterministic)
    write_osf_metadata(entries_with_zip)

    # Optional: verify manifest
    if args.verify_manifest:
        checked, mismatched = verify_manifest(strict=args.strict)
        print(f"Verified MANIFEST entries: {checked} checked; mismatches: {mismatched}")

    # Optional: run pipeline validator
    if args.validate_pipeline:
        try:
            env = os.environ.copy()
            env.setdefault('PYTHONIOENCODING', 'utf-8')
            proc = subprocess.run(
                [sys.executable, str(ROOT / 'tools' / 'validate_results_pipeline.py'), '--all', '--quiet'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
            )
            if proc.returncode != 0:
                print('[ERROR] Results pipeline validation failed')
                if not args.strict:
                    print(proc.stdout)
                    print(proc.stderr)
                if args.strict:
                    raise RuntimeError('Pipeline validation failed')
            else:
                print('Results pipeline validation: PASS')
        except Exception as e:
            if args.strict:
                raise
            print(f"[WARN] Could not run results pipeline validator: {e}")

    print(f"Staged {len(entries)} payload file(s) in {UPLOAD}")
    if zip_info:
        print(f"Created bundle: {zip_info[0]} ({zip_info[1]} bytes)")
    print("Wrote MANIFEST.md, zenodo_metadata.json, osf_metadata.json, OSF_UPLOAD_OVERVIEW.md")

if __name__ == '__main__':
    main()
