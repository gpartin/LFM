# -*- coding: utf-8 -*-
"""Regression tests for canonical upload outputs.

Ensures:
- Canonical registry is consistent with MASTER_TEST_STATUS.csv
- Narrative docs (PREPRINT, PRIORITY_CLAIM, SCIENTIFIC_CLAIMS) contain canonical banner
- RESULTS_COMPREHENSIVE uses executed-only denominators

These tests do not run physics simulations and are safe for CI.
"""
from __future__ import annotations
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]  # workspace/
UPLOAD_OSF = ROOT / 'uploads' / 'osf'
UPLOAD_ZEN = ROOT / 'uploads' / 'zenodo'
RESULTS = ROOT / 'results'


def test_canonical_registry_consistency():
    from tools.verify_canonical_consistency import main as verify_main  # type: ignore
    # Returns 0 on success
    assert verify_main(strict=False) == 0


def _assert_banner_present(base: Path):
    banner_token = 'Canonical Validation Status'
    for name in ['PREPRINT_MANUSCRIPT.md', 'PRIORITY_CLAIM.md', 'SCIENTIFIC_CLAIMS.md']:
        p = base / name
        if p.exists():
            txt = p.read_text(encoding='utf-8')
            assert banner_token in txt, f"Missing canonical banner in {p}"


def test_narrative_docs_have_banner():
    if UPLOAD_OSF.exists():
        _assert_banner_present(UPLOAD_OSF)
    if UPLOAD_ZEN.exists():
        _assert_banner_present(UPLOAD_ZEN)


def test_comprehensive_uses_executed_only():
    for base in [UPLOAD_OSF, UPLOAD_ZEN]:
        p = base / 'RESULTS_COMPREHENSIVE.md'
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8')
        # Total line should mention executed-only and not show 105/106 style
        assert 'executed-only' in txt or 'executed only' in txt
        assert '105/106' not in txt, 'Found total 105/106 denominator; expected executed-only'
        # Per-tier pass rate should avoid 25/26 pattern
        assert '25/26' not in txt, 'Found tier denominator using defined tests; expected executed-only'

