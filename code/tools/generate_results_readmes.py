#!/usr/bin/env python3
"""
Generate readme.txt files within each results directory, summarizing available
summary.json, CSVs, and plot images.

Usage:
  python tools/generate_results_readmes.py [results_root]

Defaults to ./results
"""
import os
import sys
import json
import datetime
from pathlib import Path

RESULTS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results')
TEMPLATE_PATH = Path('docs/templates/RESULTS_README_TEMPLATE.txt')
README_NAME = 'readme.txt'

PLOT_EXTS = {'.png', '.jpg', '.jpeg', '.svg', '.gif'}
CSV_EXTS = {'.csv', '.tsv'}


def format_metrics(summary_path: Path) -> str:
    if not summary_path.exists():
        return "- (no summary.json found)\n"
    try:
        data = json.loads(summary_path.read_text(encoding='utf-8', errors='ignore'))
    except Exception as e:
        return f"- Failed to parse summary.json: {e}\n"

    # Flatten top-level scalar fields into bullet list
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)):
                lines.append(f"- {k}: {v}")
            elif isinstance(v, dict):
                # include a couple of common keys if present
                for kk in ("status", "pass", "fail", "drift", "duration", "cases"):
                    if kk in v and isinstance(v[kk], (str, int, float, bool)):
                        lines.append(f"- {k}.{kk}: {v[kk]}")
    if not lines:
        lines.append("- (summary.json present but no simple scalar metrics to display)")
    return "\n".join(lines) + "\n"


def list_files(root: Path) -> str:
    lines = []
    for p in sorted(root.iterdir(), key=lambda x: (x.is_dir(), x.name.lower())):
        try:
            if p.is_dir():
                continue
            size = p.stat().st_size
            lines.append(f"- {p.name} ({size} bytes)")
        except Exception:
            continue
    if not lines:
        lines.append("- (no files)")
    return "\n".join(lines) + "\n"


def main():
    template = TEMPLATE_PATH.read_text(encoding='utf-8') if TEMPLATE_PATH.exists() else None
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for dirpath, dirnames, filenames in os.walk(RESULTS_ROOT):
        root = Path(dirpath)
        rel = root.relative_to(RESULTS_ROOT)

        # Count plots and CSVs
        plot_count = 0
        csv_count = 0
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in PLOT_EXTS:
                plot_count += 1
            if ext in CSV_EXTS:
                csv_count += 1

        summary_path = root / 'summary.json'
        has_summary = 'yes' if summary_path.exists() else 'no'
        metrics_block = format_metrics(summary_path)
        files_block = list_files(root)

        content = (
            template.format(
                rel_path=str(rel) if str(rel) != '.' else '/',
                generated_at=now,
                has_summary=has_summary,
                csv_count=csv_count,
                plot_count=plot_count,
                metrics_block=metrics_block,
                files_block=files_block,
            ) if template else (
                f"# Test Results Summary\n\n"
                f"Directory: {rel}\nGenerated: {now}\n\n"
                f"## Overview\n- Contains summary.json: {has_summary}\n- CSV files: {csv_count}\n- Plot images: {plot_count}\n\n"
                f"## Key Metrics\n{metrics_block}\n"
                f"## Files\n{files_block}\n"
            )
        )

        readme_path = root / README_NAME
        try:
            readme_path.write_text(content, encoding='utf-8')
            print(f"Wrote {readme_path}")
        except Exception as e:
            print(f"WARN: Failed to write {readme_path}: {e}")


if __name__ == '__main__':
    main()
