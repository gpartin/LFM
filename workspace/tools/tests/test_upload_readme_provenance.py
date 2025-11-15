# -*- coding: utf-8 -*-
"""Ensure upload README includes deterministic provenance.

Checks both OSF and Zenodo payloads (if present):
- Contains a 40-hex Git commit SHA
- Mentions deterministic mode (on/off)
- Contains a Generation Date line/value
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_OSF = ROOT / 'uploads' / 'osf'
UPLOAD_ZEN = ROOT / 'uploads' / 'zenodo'

SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")


def _check_readme(path: Path):
    assert path.exists(), f"README not found at {path}"
    txt = path.read_text(encoding='utf-8')
    # Git SHA present
    assert SHA_RE.search(txt), "Missing Git commit SHA in README"
    # Deterministic flag present
    assert ('deterministic: on' in txt.lower()) or ('deterministic: off' in txt.lower()), "Missing deterministic mode in README"
    # Generation date present
    assert ('Generation Date' in txt) or ('built' in txt) or ('generated' in txt.lower()), "Missing generation time in README"


def test_readme_provenance_blocks():
    if UPLOAD_OSF.exists():
        _check_readme(UPLOAD_OSF / 'README.md')
    if UPLOAD_ZEN.exists():
        _check_readme(UPLOAD_ZEN / 'README.md')
