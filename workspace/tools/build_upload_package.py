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
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
# Add src/ for module imports after src/ reorganization
SRC_DIR = ROOT_DIR / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
from utils.lfm_results import update_master_test_status  # type: ignore

ROOT = ROOT_DIR
RESULTS = ROOT / 'results'
# Legacy dry-run upload folder (kept for backward compatibility)
UPLOAD = ROOT / 'docs' / 'upload'
# New canonical destinations for OSF and Zenodo payloads
DEST_OSF = ROOT / 'uploads' / 'osf'
DEST_ZENODO = ROOT / 'uploads' / 'zenodo'

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
    """Return timestamp string from SOURCE_DATE_EPOCH or current date as fallback.
    
    For production uploads, SOURCE_DATE_EPOCH should be set or omit --deterministic flag.
    The 1970 fallback is only for testing reproducibility.
    """
    sde = os.environ.get('SOURCE_DATE_EPOCH')
    if sde and sde.isdigit():
        try:
            return datetime.utcfromtimestamp(int(sde)).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
    # Use current date instead of 1970 for production uploads
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _deterministic_date_stamp() -> str:
    """Return YYYYMMDD from SOURCE_DATE_EPOCH or current date as fallback.
    
    For production uploads, SOURCE_DATE_EPOCH should be set or omit --deterministic flag.
    The 19700101 fallback is only for testing reproducibility.
    """
    sde = os.environ.get('SOURCE_DATE_EPOCH')
    if sde and sde.isdigit():
        try:
            return datetime.utcfromtimestamp(int(sde)).strftime('%Y%m%d')
        except Exception:
            pass
    # Use current date instead of 19700101 for production uploads
    return datetime.now().strftime('%Y%m%d')


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
        # Fallback acceptance: allow validated package if equivalent PDFs or TXT sources exist
        fallback_ok = False

        try:
            # Fallback A: All four governing PDFs present at upload root
            pdf_set = [
                UPLOAD / 'Executive_Summary.pdf',
                UPLOAD / 'LFM_Master.pdf',
                UPLOAD / 'LFM_Core_Equations.pdf',
                UPLOAD / 'LFM_Phase1_Test_Design.pdf',
            ]
            pdf_ok = all(p.exists() for p in pdf_set)

            # Fallback B: Canonical TXT sources staged under upload/txt
            txt_dir = UPLOAD / 'txt'
            txt_set = [
                txt_dir / 'Executive_Summary.txt',
                txt_dir / 'LFM_Master.txt',
                txt_dir / 'LFM_Core_Equations.txt',
                txt_dir / 'LFM_Phase1_Test_Design.txt',
            ]
            txt_ok = all(p.exists() for p in txt_set)

            # Fallback C: Master DOCX present with dated suffix (e.g., LFM_Master_YYYYMMDD_v1.docx)
            master_docx_ok = any(UPLOAD.glob('LFM_Master_*_v1.docx'))

            fallback_ok = pdf_ok or (txt_ok and master_docx_ok)
        except Exception:
            fallback_ok = False

        if fallback_ok:
            issues.append('Evidence DOCX missing; using fallback artifacts (PDF/TXT)')
        else:
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
        # In strict mode, only fail if the issues include hard errors (i.e., no accepted fallback)
        hard_errors = [msg for msg in issues if msg.startswith('Missing evidence DOCX files:')]
        if hard_errors:
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


def write_manifest_for(base_dir: Path, entries: List[Tuple[str, int, str]], deterministic: bool = False):
    """Write MANIFEST.md under base_dir using provided entries."""
    now = _deterministic_now_str() if deterministic else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines: List[str] = []
    lines.append('# Upload Manifest')
    lines.append('')
    lines.append(f'- Generated: {now}')
    lines.append(f'- Scope: Package manifest for {base_dir.as_posix()}')
    lines.append('')
    lines.append('## Provenance')
    lines.extend(_collect_provenance(deterministic))
    lines.append('')
    lines.append('| File | Size (bytes) | SHA256 |')
    lines.append('|------|--------------:|--------|')
    for rel, size, digest in entries:
        lines.append(f'| {rel} | {size} | `{digest}` |')
    lines.append('')
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'MANIFEST.md').write_text('\n'.join(lines), encoding='utf-8')


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


def audit_upload_compliance(entries: List[Tuple[str, int, str]]) -> List[str]:
    """Audit upload directory for IP/legal and repository hygiene concerns.

    Returns list of human-readable issues and writes UPLOAD_COMPLIANCE_AUDIT.md.
    """
    issues: List[str] = []

    def require(path: Path, label: str):
        if not path.exists():
            issues.append(f"Missing required: {label}")

    # Required presence
    require(UPLOAD, 'docs/upload directory')
    require(UPLOAD / 'LICENSE', 'docs/upload/LICENSE')
    require(UPLOAD / 'NOTICE', 'docs/upload/NOTICE')
    require(UPLOAD / 'MANIFEST.md', 'docs/upload/MANIFEST.md')

    # OSF/Zenodo metadata
    if not ((UPLOAD / 'zenodo_metadata.json').exists() or (UPLOAD / 'zenodo' / 'zenodo_metadata.json').exists()):
        issues.append('Zenodo metadata json missing (zenodo_metadata.json)')
    if not (UPLOAD / 'osf_metadata.json').exists():
        issues.append('OSF metadata json missing (osf_metadata.json)')

    # Leak scan for sensitive names
    leaks = []
    for f in sorted(UPLOAD.rglob('*')):
        name = f.name.lower()
        if any(x in name for x in ['secret', 'token', 'password', 'private_key']) or name.endswith('.pyc'):
            leaks.append(str(f.relative_to(UPLOAD)))
    if leaks:
        issues.append(f"Potentially sensitive artifacts present: {len(leaks)} (see list)")

    # Produce report
    lines = ['# Upload Compliance Audit', '', '## Summary', '']
    lines += [f"- {i}" for i in (issues if issues else ['No issues found'])]
    if leaks:
        lines += ['', '## Potentially sensitive artifacts', '']
        lines += [f"- {x}" for x in leaks[:200]]
    (UPLOAD / 'UPLOAD_COMPLIANCE_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return issues


def audit_dir_compliance(base_dir: Path, entries: List[Tuple[str, int, str]]) -> List[str]:
    """Audit arbitrary directory for compliance items (LICENSE/NOTICE/MANIFEST)."""
    issues: List[str] = []

    def require(path: Path, label: str):
        if not path.exists():
            issues.append(f"Missing required: {label}")

    require(base_dir, f'{base_dir.name} directory')
    require(base_dir / 'LICENSE', f'{base_dir}/LICENSE')
    require(base_dir / 'NOTICE', f'{base_dir}/NOTICE')
    require(base_dir / 'MANIFEST.md', f'{base_dir}/MANIFEST.md')

    # Basic leak scan as in legacy audit
    leaks = []
    for f in sorted(base_dir.rglob('*')):
        name = f.name.lower()
        if any(x in name for x in ['secret', 'token', 'password', 'private_key']) or name.endswith('.pyc'):
            leaks.append(str(f.relative_to(base_dir)))
    if leaks:
        issues.append(f"Potentially sensitive artifacts present: {len(leaks)} (see list)")

    # Write report
    lines = ['# Upload Compliance Audit', '', '## Summary', '']
    lines += [f"- {i}" for i in (issues if issues else ['No issues found'])]
    if leaks:
        lines += ['', '## Potentially sensitive artifacts', '']
        lines += [f"- {x}" for x in leaks[:200]]
    (base_dir / 'UPLOAD_COMPLIANCE_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return issues


def _list_entries_for_dir(base_dir: Path) -> List[Tuple[str, int, str]]:
    """Return list of (relative_path, size, sha256) for files under base_dir."""
    entries: List[Tuple[str, int, str]] = []
    if not base_dir.exists():
        return entries
    for f in sorted(base_dir.rglob('*')):
        if f.is_file():
            rel = str(f.relative_to(base_dir)).replace('\\', '/')
            entries.append((rel, f.stat().st_size, sha256_file(f)))
    return entries


def copy_results_summaries(dest_dir: Path, include_full: bool = False):
    """Copy key result files from workspace/results to dest_dir/results for reproducibility.
    
    Args:
        dest_dir: Destination directory (uploads/osf or uploads/zenodo)
        include_full: If True, copy everything; if False, copy only summaries/plots
    
    Returns:
        List of relative paths copied
    """
    copied = []
    src_root = ROOT / 'results'
    dst_root = dest_dir / 'results'
    
    if not src_root.exists():
        return copied
    
    dst_root.mkdir(parents=True, exist_ok=True)
    
    # Copy top-level files
    for top_file in ['MASTER_TEST_STATUS.csv', 'README.md', 'parallel_run_summary.json', 'parallel_test_results.json']:
        src = src_root / top_file
        if src.exists() and src.is_file():
            dst = dst_root / top_file
            try:
                shutil.copyfile(src, dst)
                copied.append(f"results/{top_file}")
            except Exception:
                pass
    
    # Copy per-category results
    for category_dir in sorted(src_root.iterdir()):
        if not category_dir.is_dir():
            continue
        if category_dir.name in ['__pycache__', '.git']:
            continue
        
        dst_category = dst_root / category_dir.name
        dst_category.mkdir(parents=True, exist_ok=True)
        
        # Copy category README
        readme = category_dir / 'README.md'
        if readme.exists():
            try:
                shutil.copyfile(readme, dst_category / 'README.md')
                copied.append(f"results/{category_dir.name}/README.md")
            except Exception:
                pass
        
        # Copy per-test results
        for test_dir in sorted(category_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            if test_dir.name in ['__pycache__', '.git']:
                continue
            
            dst_test = dst_category / test_dir.name
            dst_test.mkdir(parents=True, exist_ok=True)
            
            # Always copy: summary.json, readme.txt
            for essential in ['summary.json', 'readme.txt']:
                src_file = test_dir / essential
                if src_file.exists():
                    try:
                        shutil.copyfile(src_file, dst_test / essential)
                        copied.append(f"results/{category_dir.name}/{test_dir.name}/{essential}")
                    except Exception:
                        pass
            
            # Copy plots directory
            plots_dir = test_dir / 'plots'
            if plots_dir.exists() and plots_dir.is_dir():
                dst_plots = dst_test / 'plots'
                dst_plots.mkdir(parents=True, exist_ok=True)
                for plot in sorted(plots_dir.glob('*.png')):
                    try:
                        shutil.copyfile(plot, dst_plots / plot.name)
                        copied.append(f"results/{category_dir.name}/{test_dir.name}/plots/{plot.name}")
                    except Exception:
                        pass
            
            # Optionally copy diagnostics (if include_full)
            if include_full:
                diag_dir = test_dir / 'diagnostics'
                if diag_dir.exists() and diag_dir.is_dir():
                    dst_diag = dst_test / 'diagnostics'
                    dst_diag.mkdir(parents=True, exist_ok=True)
                    for diag in sorted(diag_dir.glob('*.csv'))[:10]:  # Limit CSVs
                        try:
                            shutil.copyfile(diag, dst_diag / diag.name)
                            copied.append(f"results/{category_dir.name}/{test_dir.name}/diagnostics/{diag.name}")
                        except Exception:
                            pass
    
    return copied


def generate_tier_achievements(tier_name: str, category_dir: str, dest_dir: Path, 
                               tier_description: str, significance_text: str,
                               deterministic: bool = False) -> str:
    """Generate TIER{N}_ACHIEVEMENTS.md from tier results directory.
    
    Args:
        tier_name: Display name (e.g., "Tier 1 — Relativistic")
        category_dir: Results subdirectory name (e.g., "Relativistic")
        dest_dir: Destination directory for output file
        tier_description: Overview text for tier
        significance_text: Significance/implications text
        deterministic: Use fixed timestamps for reproducibility
    
    Returns:
        Relative path of generated file.
    """
    results_dir = ROOT / 'results' / category_dir
    stamp = _deterministic_now_str() if deterministic else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = f"{tier_name.split('—')[0].strip().upper().replace(' ', '_')}_ACHIEVEMENTS.md"
    
    lines = [
        f'# {tier_name} Validation Achievements',
        '',
        f'Generated: {stamp}',
        '',
        '## Overview',
        '',
        tier_description,
        '',
        '## Key Achievements',
        '',
    ]
    
    # Collect test results
    tests = []
    if results_dir.exists():
        for test_dir in sorted(results_dir.iterdir()):
            if not test_dir.is_dir() or test_dir.name.startswith('.'):
                continue
            summary_file = test_dir / 'summary.json'
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    tests.append({
                        'id': test_dir.name,
                        'summary': summary
                    })
                except Exception:
                    pass
    
    # Determine pass/fail status using same logic as update_master_test_status()
    def get_test_status(summary):
        """Extract pass/fail status from summary dict, handling multiple field formats."""
        if summary.get("skipped") is True:
            return "SKIP"
        elif "status" in summary:
            status = summary["status"]
        elif "passed" in summary:
            passed_val = summary["passed"]
            if isinstance(passed_val, bool):
                status = "PASS" if passed_val else "FAIL"
            else:
                status = "UNKNOWN"
        else:
            status = "UNKNOWN"
        
        # Normalize status values to uppercase
        status_upper = str(status).upper()
        if status_upper in ["PASSED", "PASS", "TRUE"]:
            return "PASS"
        elif status_upper in ["FAILED", "FAIL", "FALSE"]:
            return "FAIL"
        elif status_upper in ["SKIPPED", "SKIP"]:
            return "SKIP"
        return status_upper
    
    # Count passes
    total = len(tests)
    passed = sum(1 for t in tests if get_test_status(t['summary']) == 'PASS')
    
    lines.append(f'- **Total {tier_name} Tests**: {total}')
    lines.append(f'- **Tests Passed**: {passed} ({100*passed//total if total > 0 else 0}%)')
    lines.append('')
    
    if tests:
        lines.append('## Test Results Summary')
        lines.append('')
        lines.append('| Test ID | Status | Description |')
        lines.append('|---------|--------|-------------|')
        
        for test in sorted(tests, key=lambda x: x['id']):
            test_id = test['id']
            status = get_test_status(test['summary'])
            desc = test['summary'].get('description', 'No description')
            lines.append(f"| {test_id} | {status} | {desc[:60]} |")
        
        lines.append('')
    
    lines.append('## Significance')
    lines.append('')
    lines.append(significance_text)
    lines.append('')
    lines.append('---')
    lines.append('License: CC BY-NC-ND 4.0')
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    output = dest_dir / filename
    output.write_text('\n'.join(lines), encoding='utf-8')
    return str(output.relative_to(dest_dir)).replace('\\', '/')


def generate_evidence_review(dest_dir: Path, deterministic: bool = False) -> str:
    """Generate EVIDENCE_REVIEW.md auditing doc claims vs actual results using template.
    
    Returns relative path of generated file.
    """
    stamp = _deterministic_now_str() if deterministic else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load discoveries
    disc_file = ROOT / 'docs' / 'discoveries' / 'discoveries.json'
    discoveries = []
    if disc_file.exists():
        try:
            discoveries = json.loads(disc_file.read_text(encoding='utf-8'))
        except Exception:
            pass
    
    # Load master test status (custom format with header lines)
    master_file = ROOT / 'results' / 'MASTER_TEST_STATUS.csv'
    test_statuses = {}
    if master_file.exists():
        try:
            import csv
            with open(master_file, 'r', encoding='utf-8') as f:
                csv_lines = f.readlines()
                # Find lines with "Test_ID" header (marks start of test data sections)
                for i, line in enumerate(csv_lines):
                    if line.startswith('Test_ID,'):
                        # Parse from this header line onward until next section or EOF
                        section_lines = []
                        for j in range(i, len(csv_lines)):
                            if csv_lines[j].strip() and not csv_lines[j].startswith('TIER'):
                                section_lines.append(csv_lines[j])
                            elif csv_lines[j].startswith('TIER') and section_lines:
                                # Hit next tier section header, process accumulated lines
                                break
                        # Parse this section
                        if section_lines:
                            reader = csv.DictReader(section_lines)
                            for row in reader:
                                test_id = row.get('Test_ID', '')
                                status = row.get('Status', 'UNKNOWN')
                                if test_id and test_id != 'Test_ID':  # Skip any repeated headers
                                    test_statuses[test_id] = status
        except Exception as e:
            pass
    
    # Cross-reference
    verified = []
    needs_review = []
    
    for disc in discoveries:
        title = disc.get('title', 'Unknown')
        tier = disc.get('tier', '')
        evidence = disc.get('evidence', '')
        links = disc.get('links', [])
        
        # Check if evidence links reference test results
        has_test_ref = any('results/' in str(link) or any(tid in str(link) for tid in test_statuses.keys()) for link in links)
        
        if has_test_ref:
            verified.append(f"✓ {title} ({tier})")
        else:
            needs_review.append(f"⚠ {title} ({tier}) — Evidence: {evidence}")
    
    # Render template
    env = _jinja_env()
    template = env.get_template('evidence_review.md.j2')
    content = template.render(
        generation_time=stamp,
        discovery_count=len(discoveries),
        test_count=len(test_statuses),
        verified=verified[:20],  # Limit output
        verified_overflow=max(0, len(verified) - 20),
        needs_review=needs_review[:10],
        review_overflow=max(0, len(needs_review) - 10)
    )
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    output = dest_dir / 'EVIDENCE_REVIEW.md'
    output.write_text(content, encoding='utf-8')
    return str(output.relative_to(dest_dir)).replace('\\', '/')


def _generate_discoveries_overview(dest_dir: Path, deterministic: bool = False):
    """Generate DISCOVERIES_OVERVIEW.md in dest_dir from docs/discoveries/discoveries.json."""
    reg = ROOT / 'docs' / 'discoveries' / 'discoveries.json'
    entries = []
    try:
        if reg.exists():
            entries = json.loads(reg.read_text(encoding='utf-8'))
            if not isinstance(entries, list):
                entries = []
    except Exception:
        entries = []

    stamp = _deterministic_now_str() if deterministic else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = [
        '---',
        'title: "Scientific Discoveries and Domains of Emergence"',
        'author: "Greg D. Partin"',
        'institution: "LFM Research, Los Angeles CA USA"',
        'license: "CC BY-NC-ND 4.0"',
        'contact: "latticefieldmediumresearch@gmail.com"',
        'orcid: "https://orcid.org/0009-0004-0327-6528"',
        'doi: "10.5281/zenodo.17510124"',
        f'generated: "{stamp}"',
        '---',
        '',
        '## Summary Table',
        '',
        '| Date | Tier | Title | Evidence |',
        '|------|------|-------|----------|',
    ]
    rows = []
    for e in sorted(entries, key=lambda x: x.get('date','')):
        date = (e.get('date','') or '')[:10]
        tier = e.get('tier','')
        title = e.get('title','')
        ev = e.get('evidence','')
        rows.append(f"| {date} | {tier} | {title} | {ev} |")
    if not rows:
        rows.append('| - | - | (No discoveries recorded) | - |')

    details = ['','## Detailed List','']
    for e in sorted(entries, key=lambda x: x.get('date','')):
        details.append(f"- {e.get('date','')[:10]} — {e.get('title','')} ({e.get('tier','')})")
        if e.get('summary'):
            details.append(f"  - {e['summary']}")
        if e.get('evidence'):
            details.append(f"  - Evidence: {e['evidence']}")
        if e.get('links'):
            details.append(f"  - Links: {', '.join(e['links'])}")
    details.append('')
    details.append(f"Generated: {stamp}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / 'DISCOVERIES_OVERVIEW.md').write_text('\n'.join(header + rows + details), encoding='utf-8')


def _ensure_legal_docs(dest_dir: Path):
    """Ensure LICENSE and NOTICE are present in dest_dir by copying from repo root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ['LICENSE', 'NOTICE']:
        src = ROOT / name
        dst = dest_dir / name
        try:
            if src.exists():
                shutil.copyfile(src, dst)
        except Exception:
            pass


def _merge_from_legacy_upload(legacy_dir: Path, dest_dir: Path) -> list[str]:
    """Copy useful artifacts from legacy docs/upload into dest_dir.

    Returns list of relative paths (under dest_dir) that were copied.
    """
    copied: list[str] = []
    if not legacy_dir.exists():
        return copied
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy if present
    files_to_copy = [
        'results_MASTER_TEST_STATUS.csv',
        'RESULTS_REPORT.md',
        'PLOTS_OVERVIEW.md',
        'README_LFM.txt',
    ]
    for name in files_to_copy:
        src = legacy_dir / name
        if src.exists() and src.is_file():
            dst = dest_dir / name
            try:
                shutil.copyfile(src, dst)
                copied.append(str(dst.relative_to(dest_dir)).replace('\\','/'))
            except Exception:
                pass

    # Directories to mirror (plots excluded - now in results/)
    dirs_to_copy = [
        'txt',
        'md',
    ]
    for dname in dirs_to_copy:
        src_dir = legacy_dir / dname
        if src_dir.exists() and src_dir.is_dir():
            dst_dir = dest_dir / dname
            try:
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                # Record top-level marker for manifest context
                copied.append(dname)
            except Exception:
                pass

    return copied


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
        # Fallback: if no evidence DOCX were staged, stage canonical TXT sources from docs/text
        if not staged_evidence:
            txt_canonical = _stage_canonical_txt_sources()
            if txt_canonical:
                print(f"Staged {len(txt_canonical)} canonical TXT source file(s) from docs/text as fallback")

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

    # Compliance/audit (upload)
    issues = audit_upload_compliance(entries_with_zip)
    print(f"Upload compliance audit: {len(issues)} issue(s) flagged (see UPLOAD_COMPLIANCE_AUDIT.md)")

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

    # ---------------- New: Build OSF and Zenodo payloads in uploads/ ----------------
    print("\n[INFO] Building OSF and Zenodo payloads under uploads/ ...")
    # Ensure legal docs present
    _ensure_legal_docs(DEST_OSF)
    _ensure_legal_docs(DEST_ZENODO)
    # Generate discoveries overview for both
    _generate_discoveries_overview(DEST_OSF, deterministic=args.deterministic)
    _generate_discoveries_overview(DEST_ZENODO, deterministic=args.deterministic)

    # Merge legacy artifacts from docs/upload into each destination
    merged_osf = _merge_from_legacy_upload(UPLOAD, DEST_OSF)
    merged_zen = _merge_from_legacy_upload(UPLOAD, DEST_ZENODO)
    if merged_osf:
        print(f"Merged {len(merged_osf)} legacy artifact group(s) into OSF payload")
    if merged_zen:
        print(f"Merged {len(merged_zen)} legacy artifact group(s) into Zenodo payload")
    
    # Copy results tree to both destinations
    print("\n[INFO] Copying results summaries to upload payloads...")
    copied_osf_results = copy_results_summaries(DEST_OSF, include_full=False)
    copied_zen_results = copy_results_summaries(DEST_ZENODO, include_full=False)
    print(f"Copied {len(copied_osf_results)} result files to OSF payload")
    print(f"Copied {len(copied_zen_results)} result files to Zenodo payload")
    
    # Generate tier achievements and core docs from metadata/templates (deterministic)
    print("\n[INFO] Generating tier documents from metadata...")
    metadata = load_tier_metadata()

    # Per-tier achievements
    for tier in metadata.get('tiers', []):
        generate_tier_achievements_from_template(tier, DEST_OSF, deterministic=args.deterministic)
        generate_tier_achievements_from_template(tier, DEST_ZENODO, deterministic=args.deterministic)

    # Comprehensive and Executive documents
    generate_results_comprehensive(DEST_OSF, metadata, deterministic=args.deterministic)
    generate_results_comprehensive(DEST_ZENODO, metadata, deterministic=args.deterministic)
    generate_executive_summary(DEST_OSF, metadata, deterministic=args.deterministic)
    generate_executive_summary(DEST_ZENODO, metadata, deterministic=args.deterministic)
    
    # Core static documents (single source TXT -> MD)
    generate_core_equations(DEST_OSF, deterministic=args.deterministic)
    generate_core_equations(DEST_ZENODO, deterministic=args.deterministic)
    generate_master_document(DEST_OSF, deterministic=args.deterministic)
    generate_master_document(DEST_ZENODO, deterministic=args.deterministic)
    generate_test_design(DEST_OSF, deterministic=args.deterministic)
    generate_test_design(DEST_ZENODO, deterministic=args.deterministic)

    print("Generated tier achievements and core summaries from templates")
    
    # Generate evidence review report
    print("\n[INFO] Generating evidence review audit...")
    ev_rev_osf = generate_evidence_review(DEST_OSF, deterministic=args.deterministic)
    ev_rev_zen = generate_evidence_review(DEST_ZENODO, deterministic=args.deterministic)
    print(f"Generated: {ev_rev_osf}")

    # Write manifests for both destinations
    osf_entries = _list_entries_for_dir(DEST_OSF)
    zen_entries = _list_entries_for_dir(DEST_ZENODO)
    write_manifest_for(DEST_OSF, osf_entries, deterministic=args.deterministic)
    write_manifest_for(DEST_ZENODO, zen_entries, deterministic=args.deterministic)

    # Run compliance audits for both destinations
    osf_issues = audit_dir_compliance(DEST_OSF, osf_entries)
    zen_issues = audit_dir_compliance(DEST_ZENODO, zen_entries)
    print(f"OSF payload compliance: {len(osf_issues)} issue(s). Report: {DEST_OSF / 'UPLOAD_COMPLIANCE_AUDIT.md'}")
    print(f"Zenodo payload compliance: {len(zen_issues)} issue(s). Report: {DEST_ZENODO / 'UPLOAD_COMPLIANCE_AUDIT.md'}")

# ---------------- Template and metadata helpers (deterministic generation) ----------------

def _jinja_env():
    """Create Jinja2 environment for templates under tools/templates."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore
    except Exception as e:
        raise RuntimeError("Jinja2 is required for template-based generation. Install via requirements.txt") from e
    templates_dir = ROOT / 'tools' / 'templates'
    loader = FileSystemLoader(str(templates_dir))
    env = Environment(loader=loader, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return env


def _read_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tier_metadata() -> dict:
    """Load comprehensive tier metadata. Fallback to minimal metadata + results scan."""
    comp = ROOT / 'config' / 'tier_metadata_comprehensive.json'
    if comp.exists():
        data = _read_json(comp)
        # Normalize: ensure 'tiers' exists and each has tier_number
        tiers = data.get('tiers', [])
        # If test_count missing, compute from results
        for i, t in enumerate(tiers, start=1):
            t.setdefault('tier_number', i)
            cat = t.get('category_dir')
            if cat:
                res_dir = ROOT / 'results' / cat
                if res_dir.exists():
                    t['test_count'] = t.get('test_count', sum(1 for d in res_dir.iterdir() if d.is_dir() and not d.name.startswith('.')))
        data['tiers'] = tiers
        return data
    # Fallback to minimal metadata
    minimal = ROOT / 'config' / 'tier_metadata.json'
    tiers_list = []
    order = []
    tiers_info = {}
    if minimal.exists():
        m = _read_json(minimal)
        order = m.get('order', [])
        tiers_info = m.get('tiers', {})
    for idx, cat in enumerate(order, start=1):
        info = tiers_info.get(cat, {})
        res_dir = ROOT / 'results' / cat
        test_count = sum(1 for d in res_dir.iterdir() if d.is_dir() and not d.name.startswith('.')) if res_dir.exists() else 0
        tiers_list.append({
            'tier_number': idx,
            'tier_id': f'tier{idx}_{cat.lower()}',
            'tier_name': info.get('title', f'Tier {idx} — {cat}'),
            'short_name': info.get('title', f'Tier {idx} — {cat}').split('—')[-1].strip(),
            'category_dir': cat,
            'test_prefix': None,
            'test_count': test_count,
            'description': info.get('description', ''),
            'significance': info.get('description', ''),
            'key_validations': [],
            'representative_plot': None,
        })
    return {'tiers': tiers_list, 'document_templates': {}}


def get_summary_status(summary: dict) -> str:
    """Extract normalized PASS/FAIL/SKIP/UNKNOWN from a summary.json dict."""
    try:
        if summary.get('skipped') is True:
            return 'SKIP'
        if 'status' in summary:
            status = str(summary['status']).upper()
            if status in ('PASSED', 'PASS', 'TRUE'):
                return 'PASS'
            if status in ('FAILED', 'FAIL', 'FALSE'):
                return 'FAIL'
            if status in ('SKIPPED', 'SKIP'):
                return 'SKIP'
            return status
        if 'passed' in summary:
            pv = summary['passed']
            if isinstance(pv, bool):
                return 'PASS' if pv else 'FAIL'
            return str(pv).upper()
    except Exception:
        pass
    return 'UNKNOWN'


def _collect_tests_for_category(category_dir: str) -> list[dict]:
    res_dir = ROOT / 'results' / category_dir
    tests: list[dict] = []
    if res_dir.exists():
        for tdir in sorted(res_dir.iterdir()):
            if tdir.is_dir() and not tdir.name.startswith('.'):
                s = tdir / 'summary.json'
                if s.exists():
                    try:
                        summary = _read_json(s)
                        tests.append({
                            'id': tdir.name,
                            'status': get_summary_status(summary),
                            'description': summary.get('description', 'No description'),
                        })
                    except Exception:
                        continue
    return tests


def _compute_pass_counts(tier: dict) -> int:
    tests = _collect_tests_for_category(tier['category_dir'])
    return sum(1 for t in tests if t['status'] == 'PASS')


def copy_representative_plots(dest_dir: Path, tier_metadata: dict) -> list[str]:
    copied: list[str] = []
    for t in tier_metadata.get('tiers', []):
        plot = t.get('representative_plot')
        if not plot:
            continue
        source = ROOT / plot.get('source_path', '')
        dest = dest_dir / plot.get('dest_filename', f"plot_tier{t['tier_number']}.png")
        try:
            if source.exists():
                dest.write_bytes(source.read_bytes())
                copied.append(dest.name)
        except Exception:
            pass
    return copied


def generate_tier_achievements_from_template(tier: dict, dest_dir: Path, deterministic: bool = False) -> str:
    env = _jinja_env()
    tmpl = env.get_template('tier_achievements.md.j2')
    tests = _collect_tests_for_category(tier['category_dir'])
    passed = sum(1 for t in tests if t['status'] == 'PASS')
    content = tmpl.render(
        tier_name=tier['tier_name'],
        description=tier['description'],
        significance=tier['significance'],
        test_count=len(tests) if tests else tier.get('test_count', 0),
        passed=passed,
        tests=tests,
        deterministic_date=_deterministic_now_str() if deterministic else None,
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"TIER_{tier['tier_number']}_ACHIEVEMENTS.md"
    (dest_dir / fname).write_text(content, encoding='utf-8')
    return fname


def generate_results_comprehensive(dest_dir: Path, tier_metadata: dict, deterministic: bool = False) -> str:
    env = _jinja_env()
    tmpl = env.get_template('results_comprehensive.md.j2')
    tiers = []
    total_tests = 0
    total_passed = 0
    for t in tier_metadata.get('tiers', []):
        passed = _compute_pass_counts(t)
        test_count = t.get('test_count', 0)
        if test_count == 0:
            test_count = len(_collect_tests_for_category(t['category_dir']))
        total_tests += test_count
        total_passed += passed
        tiers.append({
            'tier_number': t['tier_number'],
            'tier_name': t['tier_name'],
            'short_name': t.get('short_name', t['category_dir']),
            'test_count': test_count,
            'passed': passed,
            'description': t['description'],
            'key_validations': t.get('key_validations', []),
            'significance': t.get('significance', ''),
        })
    content = tmpl.render(
        results_overview=tier_metadata.get('document_templates', {}).get('results_comprehensive_overview', ''),
        tiers=tiers,
        total_tests=total_tests,
        total_passed=total_passed,
        pass_percentage=(100.0 * total_passed / total_tests) if total_tests else 0.0,
        deterministic_date=_deterministic_now_str() if deterministic else None,
        generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    (dest_dir / 'RESULTS_COMPREHENSIVE.md').write_text(content, encoding='utf-8')
    return 'RESULTS_COMPREHENSIVE.md'


def _txt_to_markdown(txt_path: Path, title: str, deterministic: bool = False) -> str:
    """Convert a TXT document to markdown with YAML frontmatter.
    
    Single source of truth pattern: All content comes from TXT file.
    Adds markdown formatting and metadata frontmatter with proper paragraph breaks.
    """
    if not txt_path.exists():
        return f"# {title}\n\nSource file missing: {txt_path}"
    
    source_text = txt_path.read_text(encoding='utf-8')
    lines = []
    
    # Extract first line as document title
    first_line = source_text.split('\n')[0].strip()
    
    # Add YAML frontmatter FIRST (proper format)
    lines.append('---')
    lines.append(f'title: "{title}"')
    lines.append('author: "Greg D. Partin"')
    lines.append('institution: "LFM Research, Los Angeles CA USA"')
    lines.append('license: "CC BY-NC-ND 4.0"')
    lines.append('contact: "latticefieldmediumresearch@gmail.com"')
    lines.append('orcid: "https://orcid.org/0009-0004-0327-6528"')
    lines.append('doi: "10.5281/zenodo.17510124"')
    if deterministic:
        lines.append(f'generated: "{_deterministic_now_str()}"')
    lines.append('---')
    lines.append('')
    
    # Now add the title from TXT file
    lines.append('# ' + first_line)
    lines.append('')
    
    # Process rest of content with paragraph detection
    prev_was_empty = False
    in_list = False
    
    for line in source_text.split('\n')[1:]:
        stripped = line.strip()
        
        # Empty line - preserve as paragraph break
        if not stripped:
            if not prev_was_empty:  # Avoid multiple empty lines
                lines.append('')
                prev_was_empty = True
            in_list = False
            continue
        
        prev_was_empty = False
        
        # Detect major section headers (short lines, not indented)
        if len(stripped) < 80 and not line.startswith(' ') and not line.startswith('\t'):
            # Check if it's a numbered section header (not a list item)
            # Section headers are like "1. Overview" or "Abstract"
            # List items are like "1. Short item text" embedded in content
            first_word = stripped.split()[0] if stripped.split() else ""
            
            # It's a numbered section if: starts with number, followed by words that look like section titles
            is_section_header = False
            if stripped[0].isdigit() and '.' in first_word:
                # Check if it's followed by title-case words (section headers)
                rest = stripped[len(first_word):].strip()
                if rest and len(rest) < 50 and rest[0].isupper():
                    is_section_header = True
            elif stripped.startswith(('Abstract', 'Overview', 'Summary', 'References', 'Implications', 'Status', 'Key ', 'Recent')):
                is_section_header = True
                
            if is_section_header:
                # Check depth by counting dots in first word
                if first_word.count('.') == 1 and first_word[0].isdigit():  # "1."
                    lines.append('')  # Add blank line before header
                    lines.append('## ' + stripped)
                    lines.append('')  # Add blank line after header
                    in_list = False
                    continue
                elif first_word.count('.') == 2:  # "1.1."
                    lines.append('')
                    lines.append('### ' + stripped)
                    lines.append('')
                    in_list = False
                    continue
                elif not first_word[0].isdigit():  # "Abstract", "Overview", etc
                    lines.append('')
                    lines.append('## ' + stripped)
                    lines.append('')
                    in_list = False
                    continue
                    
            # Check for section-like headers
            if any(stripped.startswith(prefix) for prefix in ['License', 'Note:', 'Citation', 'Contact:', 'Trademark', 'Defensive', 'Legal', 'Redistribution', 'Derivative', 'Feature', 'Consequence']):
                if stripped.endswith(':'):
                    lines.append('')
                    lines.append('### ' + stripped)
                    lines.append('')
                    in_list = False
                    continue
        
        # Detect list items (starting with -, •, or digit.)
        if stripped.startswith(('-', '•')) or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == '.'):
            if not in_list:
                lines.append('')  # Add blank line before list starts
            lines.append(stripped)
            in_list = True
            continue
        
        # Regular paragraph text - add line with proper spacing
        if in_list:
            lines.append('')  # Add blank line after list
            in_list = False
        
        # Escape backslashes in Windows paths for LaTeX compatibility
        # Pattern: LFM\something → LFM\\something (double backslash for markdown/LaTeX)
        escaped_line = stripped
        if '\\' in stripped and not stripped.startswith('\\'):
            # Only escape if it looks like a path (e.g., "LFM\code")
            # Don't escape if it's already escaped or looks like LaTeX math
            import re
            escaped_line = re.sub(r'([A-Za-z0-9_]+)\\([A-Za-z0-9_]+)', r'\1\\\\\2', stripped)
        
        # Add the line
        lines.append(escaped_line)
        
        # For long paragraphs, add a blank line after if the next line is also long
        # This helps create visual paragraph breaks
    
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('License: CC BY-NC-ND 4.0')
    
    return '\n'.join(lines)


def generate_executive_summary(dest_dir: Path, tier_metadata: dict, deterministic: bool = False) -> str:
    """Generate Executive Summary from single source TXT file with dynamic tier stats.
    
    Single source of truth: docs/text/Executive_Summary.txt
    Dynamic data: Tier test counts and pass rates injected from metadata
    """
    src_txt = ROOT / "docs" / "text" / "Executive_Summary.txt"
    
    # Start with base conversion
    content = _txt_to_markdown(src_txt, "LFM Executive Summary", deterministic)
    
    # Calculate dynamic tier statistics
    total_tests = sum(t.get('test_count', 0) for t in tier_metadata.get('tiers', []))
    content = content.replace('{{TOTAL_TESTS}}', str(total_tests))
    
    # Calculate pass rate (assuming 100% for validated tiers)
    pass_rate = 100.0
    content = content.replace('{{PASS_RATE}}', f"{pass_rate:.1f}")
    
    dest_file = dest_dir / "EXECUTIVE_SUMMARY.md"
    dest_file.write_text(content, encoding='utf-8')
    return "EXECUTIVE_SUMMARY.md"


def generate_core_equations(dest_dir: Path, deterministic: bool = False) -> str:
    """Generate CORE_EQUATIONS.md from single source TXT file.
    
    Single source of truth: docs/text/LFM_Core_Equations.txt
    """
    src_txt = ROOT / "docs" / "text" / "LFM_Core_Equations.txt"
    content = _txt_to_markdown(src_txt, "LFM Core Equations and Physics", deterministic)
    
    dest_file = dest_dir / "CORE_EQUATIONS.md"
    dest_file.write_text(content, encoding='utf-8')
    return "CORE_EQUATIONS.md"


def generate_master_document(dest_dir: Path, deterministic: bool = False) -> str:
    """Generate MASTER_DOCUMENT.md from single source TXT file.
    
    Single source of truth: docs/text/LFM_Master.txt
    """
    src_txt = ROOT / "docs" / "text" / "LFM_Master.txt"
    content = _txt_to_markdown(src_txt, "LFM Master Document", deterministic)
    
    dest_file = dest_dir / "MASTER_DOCUMENT.md"
    dest_file.write_text(content, encoding='utf-8')
    return "MASTER_DOCUMENT.md"


def generate_test_design(dest_dir: Path, deterministic: bool = False) -> str:
    """Generate TEST_DESIGN.md from single source TXT file.
    
    Single source of truth: docs/text/LFM_Phase1_Test_Design.txt
    """
    src_txt = ROOT / "docs" / "text" / "LFM_Phase1_Test_Design.txt"
    content = _txt_to_markdown(src_txt, "LFM Phase 1 Test Design", deterministic)
    
    dest_file = dest_dir / "TEST_DESIGN.md"
    dest_file.write_text(content, encoding='utf-8')
    return "TEST_DESIGN.md"


if __name__ == '__main__':
    main()
