#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import json
import re

class TextTemplateEngine:
    """Lightweight templating for LFM text sources with includes and dynamic tokens."""
    def __init__(self, project_root: Path):
        self.root = project_root
        self.results = self.root / 'results'

    def render(self, raw_text: str) -> str:
        text = self._expand_includes(raw_text)
        text = self._expand_dynamic_tokens(text)
        text = self._expand_discoveries_tokens(text)
        return text

    def _expand_includes(self, text: str) -> str:
        # Pattern: {{INCLUDE: relative/path.ext}}
        def repl(match):
            rel = match.group(1).strip()
            path = (self.root / rel).resolve()
            try:
                if path.exists():
                    return path.read_text(encoding='utf-8')
            except Exception:
                pass
            return ''
        return re.sub(r"\{\{\s*INCLUDE:\s*([^}]+)\s*\}\}", repl, text)

    def _expand_dynamic_tokens(self, text: str) -> str:
        # PASS_RATE:TIER_NAME → "X/Y (Z%)"
        def pass_rate_repl(match):
            tier = match.group(1).strip()
            passed, total = self._tier_counts(tier)
            pct = int(round((passed/total*100) if total else 0))
            return f"{passed}/{total} ({pct}%)"

        text = re.sub(r"\{\{\s*PASS_RATE:([^}]+)\}\}", pass_rate_repl, text)

        # TIER_SUMMARY_TABLE → Markdown table of tiers
        if '{{TIER_SUMMARY_TABLE}}' in text:
            table = self._tier_summary_table()
            text = text.replace('{{TIER_SUMMARY_TABLE}}', table)
        return text

    def _tier_counts(self, tier_name: str) -> tuple[int, int]:
        tdir = self.results / tier_name
        total = 0
        passed = 0
        if tdir.exists() and tdir.is_dir():
            for case in tdir.iterdir():
                if not case.is_dir():
                    continue
                total += 1
                s = case / 'summary.json'
                if s.exists():
                    try:
                        data = json.loads(s.read_text(encoding='utf-8'))
                        p = data.get('passed')
                        if p is True or str(p).lower() in ['true', 'pass', 'passed']:
                            passed += 1
                    except Exception:
                        pass
        return passed, total

    def _tier_summary_table(self) -> str:
        rows = ["| Tier | Tests | Passed | Pass rate |", "|------|-------|--------|-----------|"]
        ignore = {'.git', '__pycache__', 'Tier6', 'Tier6Demo'}
        if self.results.exists():
            for tier in sorted(d.name for d in self.results.iterdir() if d.is_dir() and d.name not in ignore):
                p, t = self._tier_counts(tier)
                pct = int(round((p/t*100) if t else 0))
                rows.append(f"| {tier} | {t} | {p} | {pct}% |")
        return "\n".join(rows)

    # === Discoveries support ===
    def _expand_discoveries_tokens(self, text: str) -> str:
        # Load registry
        reg = self.root / 'docs' / 'discoveries' / 'discoveries.json'
        try:
            discoveries = json.loads(reg.read_text(encoding='utf-8')) if reg.exists() else []
            if not isinstance(discoveries, list):
                discoveries = []
        except Exception:
            discoveries = []

        # DISCOVERY_SUMMARY_TABLE
        if '{{DISCOVERY_SUMMARY_TABLE}}' in text:
            table = self._discovery_table(discoveries)
            text = text.replace('{{DISCOVERY_SUMMARY_TABLE}}', table)

        # DISCOVERY_LIST or DISCOVERY_LIST:Tier
        def repl_list(m):
            tier = m.group(1)
            return self._discovery_list(discoveries, tier.strip() if tier else None)

        text = re.sub(r"\{\{\s*DISCOVERY_LIST(?::([^}]+))?\s*\}\}", repl_list, text)
        return text

    def _discovery_table(self, entries) -> str:
        rows = ["| Date | Tier | Title | Evidence |", "|------|------|-------|----------|"]
        for e in sorted(entries, key=lambda x: x.get('date','')):
            date = e.get('date','')[:10]
            tier = e.get('tier','')
            title = e.get('title','')
            evidence = e.get('evidence','')
            rows.append(f"| {date} | {tier} | {title} | {evidence} |")
        return "\n".join(rows) if len(rows) > 2 else "(No discoveries recorded)"

    def _discovery_list(self, entries, tier: Optional[str]) -> str:
        out = []
        for e in sorted(entries, key=lambda x: x.get('date','')):
            if tier and e.get('tier') != tier:
                continue
            date = e.get('date','')[:10]
            out.append(f"- {date} — {e.get('title','')} ({e.get('tier','')}) — {e.get('summary','')}")
        return "\n".join(out) if out else "(No discoveries recorded)"
