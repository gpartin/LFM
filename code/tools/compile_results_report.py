#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Compile an aggregate results report from results/* readme.txt and summary.json
into docs/upload/RESULTS_REPORT.md.

Usage:
  python tools/compile_results_report.py [results_root] [output_md]

Defaults:
  results_root = ./results
  output_md = ./docs/upload/RESULTS_REPORT.md
"""
import os
import sys
import json
from pathlib import Path

RESULTS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results')
OUTPUT_MD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('docs/upload/RESULTS_REPORT.md')


def gather_entries(root: Path):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        rel = d.relative_to(root)
        readme = d / 'readme.txt'
        summary = d / 'summary.json'
        item = {
            'rel': str(rel) if str(rel) != '.' else '/',
            'has_readme': readme.exists(),
            'has_summary': summary.exists(),
            'metrics': {},
        }
        if summary.exists():
            try:
                data = json.loads(summary.read_text(encoding='utf-8', errors='ignore'))
                # Keep only top-level scalars and common nested fields
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (str, int, float, bool)):
                            item['metrics'][k] = v
                        elif isinstance(v, dict):
                            for kk in ("status", "pass", "fail", "drift", "duration"):
                                if kk in v and isinstance(v[kk], (str, int, float, bool)):
                                    item['metrics'][f"{k}.{kk}"] = v[kk]
            except Exception:
                pass
        entries.append(item)
    return entries


def render_report(entries):
    lines = []
    lines.append('# LFM Results Report')
    lines.append('')
    lines.append('This report summarizes the contents of the results/ tree at build time and aggregates simple metrics from summary.json files when available.')
    lines.append('')
    lines.append('## Index')
    lines.append('')
    lines.append('| Path | README | Summary | Key Metrics |')
    lines.append('|------|--------|---------|-------------|')
    for e in sorted(entries, key=lambda x: x['rel']):
        metrics_preview = ", ".join(f"{k}={v}" for k, v in list(e['metrics'].items())[:5]) if e['metrics'] else ''
        lines.append(f"| {e['rel']} | {'yes' if e['has_readme'] else 'no'} | {'yes' if e['has_summary'] else 'no'} | {metrics_preview} |")
    lines.append('')
    lines.append('## Notes')
    lines.append('- Per-directory README files were generated automatically.')
    lines.append('- Key plots and CSVs are documented in each directory README.')
    return "\n".join(lines) + "\n"


def main():
    entries = gather_entries(RESULTS_ROOT)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(render_report(entries), encoding='utf-8')
    print(f"Wrote {OUTPUT_MD}")


if __name__ == '__main__':
    main()
