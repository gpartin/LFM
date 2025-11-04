#!/usr/bin/env python3
# Audits evidence documents against current results and emits a review report.

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TEXT = ROOT / 'docs' / 'text'
UPLOAD = ROOT / 'docs' / 'upload'

DOCS = {
    'Executive_Summary.txt': TEXT / 'Executive_Summary.txt',
    'LFM_Master.txt': TEXT / 'LFM_Master.txt',
    'LFM_Core_Equations.txt': TEXT / 'LFM_Core_Equations.txt',
    'LFM_Phase1_Test_Design.txt': TEXT / 'LFM_Phase1_Test_Design.txt',
}

@dataclass
class TierSummary:
    name: str
    total: int
    passed: int

@dataclass
class ReviewIssue:
    severity: str  # CRITICAL | WARNING | INFO
    where: str
    message: str
    suggestion: str = ''


def discover_tiers() -> Dict[str, TierSummary]:
    tiers: Dict[str, TierSummary] = {}
    for tier_dir in sorted(RESULTS.iterdir()):
        if not tier_dir.is_dir():
            continue
        # Ignore temporary or deprecated tiers
        if tier_dir.name in {'.git', '__pycache__', 'Tier6', 'Tier6Demo'}:
            continue
        tier = tier_dir.name
        total = 0
        passed = 0
        for test_dir in sorted(tier_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            total += 1
            s = test_dir / 'summary.json'
            if s.exists():
                try:
                    data = json.loads(s.read_text(encoding='utf-8'))
                    p = data.get('passed')
                    if p is True or str(p).lower() in ['true', 'pass', 'passed']:
                        passed += 1
                except Exception:
                    pass
        tiers[tier] = TierSummary(name=tier, total=total, passed=passed)
    return tiers


def read_doc(path: Path) -> str:
    if not path.exists():
        return ''
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def audit_documents(tiers: Dict[str, TierSummary]) -> Tuple[List[ReviewIssue], str]:
    issues: List[ReviewIssue] = []
    em_present = 'Electromagnetic' in tiers and tiers['Electromagnetic'].total > 0
    em_rate_100 = em_present and tiers['Electromagnetic'].total > 0 and tiers['Electromagnetic'].passed == tiers['Electromagnetic'].total

    # Basic keyword checks when EM tier exists
    if em_present:
        for doc_name, path in DOCS.items():
            content = read_doc(path)
            if not content:
                issues.append(ReviewIssue('CRITICAL', doc_name, 'Document missing or unreadable', 'Restore the authoritative text file.'))
                continue
            if not re.search(r'electromagnetic|Maxwell|Lorentz force|Coulomb', content, re.IGNORECASE):
                issues.append(ReviewIssue('CRITICAL', doc_name, 'Electromagnetic achievements not referenced', 'Add a short paragraph noting Tier 5 validation and Maxwell-law reproduction.'))

    # Pass-rate awareness note for narrative docs + hard claims checks
    for doc_name, path in DOCS.items():
        content = read_doc(path)
        if content:
            if re.search(r'\b\d+\s*/\s*\d+\b', content):
                # If fixed X/Y appears in narrative, suggest to verify
                issues.append(ReviewIssue('WARNING', doc_name, 'Hard-coded pass counts detected', 'Verify counts match current results or replace with generic language.'))
            # Flag "100%" success claims if EM is not 100%
            if re.search(r'100%\s*(test\s*)?(success|success rate)', content, re.IGNORECASE) and not em_rate_100:
                issues.append(ReviewIssue('CRITICAL', doc_name, 'Absolute 100% success claim conflicts with current EM pass rate', 'Replace with dynamic token {{PASS_RATE:Electromagnetic}} or state current pass rate from results.'))

    # Evidence PDFs are no longer required as we generate PDFs from text. Keep optional info only.
    pdf_dir = ROOT / 'docs' / 'evidence' / 'pdf'
    if pdf_dir.exists():
        missing = []
        for pdf in ['Executive_Summary.pdf', 'LFM_Master.pdf', 'LFM_Core_Equations.pdf', 'LFM_Phase1_Test_Design.pdf']:
            if not (pdf_dir / pdf).exists():
                missing.append(pdf)
        if missing:
            issues.append(ReviewIssue('INFO', 'evidence/pdf', f'Optional legacy PDFs missing: {", ".join(missing)}', 'No action needed; PDFs are generated from text.'))

    # Check upload README for similar claims and DOI consistency
    upload_readme = ROOT / 'docs' / 'upload' / 'README.md'
    if upload_readme.exists():
        readme = read_doc(upload_readme)
        if re.search(r'100%\s*(test\s*)?(success|success rate)', readme, re.IGNORECASE) and not em_rate_100:
            issues.append(ReviewIssue('CRITICAL', 'README.md', 'Absolute 100% success claim conflicts with current EM pass rate', 'Revise README claims or make them dynamic to reflect results.'))

    # DOI consistency across documents
    doi_pattern = re.compile(r'doi:\s*"(10\.5281/zenodo\.[0-9]+)"', re.IGNORECASE)
    dois_found = set()
    for md in (ROOT / 'docs' / 'upload').rglob('*.md'):
        txt = read_doc(md)
        for m in doi_pattern.finditer(txt):
            dois_found.add(m.group(1))
    if len(dois_found) > 1:
        issues.append(ReviewIssue('WARNING', 'DOI', f'Multiple DOIs referenced across docs: {sorted(dois_found)}', 'Harmonize to the reserved DOI at release time; avoid hard-coded placeholders.'))

    # Anti-circumvention / overreach language
    if upload_readme.exists():
        readme = read_doc(upload_readme)
        if 'clean room' in readme.lower() or 'anti-circumvention' in readme.lower():
            issues.append(ReviewIssue('WARNING', 'README.md', 'Overbroad anti-circumvention/clean room language', 'Consider removing or softening; licenses cannot restrict independent re-implementation of ideas.'))

    # Defensive publication and trademark notices
    for md in (ROOT / 'docs' / 'upload').rglob('*.md'):
        txt = read_doc(md)
        if 'Defensive Publication Statement' in txt:
            issues.append(ReviewIssue('INFO', md.name, 'Defensive publication language present', 'Ensure you understand impact on your own patent filings (disclosure creates prior art).'))
        if 'Trademark Notice' in txt:
            issues.append(ReviewIssue('INFO', md.name, 'Trademark notice present', 'Verify marks are used correctly; consider TM/® usage and nominative fair use caveats.'))

    # Require THIRD_PARTY_LICENSES.md in upload
    tpl = ROOT / 'THIRD_PARTY_LICENSES.md'
    up_tpl = ROOT / 'docs' / 'upload' / 'THIRD_PARTY_LICENSES.md'
    if tpl.exists() and not up_tpl.exists():
        issues.append(ReviewIssue('WARNING', 'THIRD_PARTY_LICENSES.md', 'Third-party licenses summary not included in upload', 'Include THIRD_PARTY_LICENSES.md in upload package.'))

    # Build the review markdown
    lines: List[str] = []
    lines.append('---')
    lines.append('title: "Evidence Review — Source Docs vs Results"')
    lines.append('generated: "' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '"')
    lines.append('---')
    lines.append('')
    lines.append('# Evidence Review — Source Docs vs Results')
    lines.append('')
    lines.append('This review is generated during the upload build to help ensure that authoritative documents remain aligned with the current test results.')
    lines.append('')

    # Tier rollup
    lines.append('## Tier Summary (from results/*)')
    lines.append('')
    lines.append('| Tier | Tests | Passed | Pass rate |')
    lines.append('|------|-------|--------|-----------|')
    for tier in sorted(tiers.keys()):
        t = tiers[tier]
        rate = f"{(t.passed/t.total*100 if t.total else 0):.0f}%"
        lines.append(f"| {tier} | {t.total} | {t.passed} | {rate} |")
    lines.append('')

    # Issues
    lines.append('## Findings')
    lines.append('')
    if not issues:
        lines.append('All checks passed. No inconsistencies detected.')
    else:
        for i, iss in enumerate(issues, 1):
            lines.append(f"{i}. [{iss.severity}] {iss.where} — {iss.message}")
            if iss.suggestion:
                lines.append(f"   - Suggestion: {iss.suggestion}")
    lines.append('')

    # Suggested EM paragraph if missing
    if em_present:
        lines.append('## Suggested EM Paragraph (if needed)')
        lines.append('')
        lines.append('Electromagnetic validation (Tier 5) demonstrates that all four Maxwell equations (Gauss’s law, Gauss’s law for magnetism, Faraday’s law, and Ampère–Maxwell law) emerge within the LFM framework, with Coulomb’s law and Lorentz force verified through direct simulation. These results confirm classical electromagnetism as an emergent property of the discrete spacetime lattice.')
        lines.append('')

    return issues, '\n'.join(lines)


def main() -> int:
    tiers = discover_tiers()
    issues, report = audit_documents(tiers)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    out = UPLOAD / 'EVIDENCE_REVIEW.md'
    out.write_text(report, encoding='utf-8')

    # Return non-zero if critical issues to allow enforcement when desired
    if any(iss.severity == 'CRITICAL' for iss in issues):
        return 2
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
