#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Record a new LFM discovery into the registry')
    parser.add_argument('--tier', required=True, help='Tier name (e.g., Electromagnetic)')
    parser.add_argument('--title', required=True, help='Short title of the discovery')
    parser.add_argument('--summary', required=True, help='1-2 sentence summary')
    parser.add_argument('--evidence', required=True, help='Relative path to evidence (e.g., results/Tier/CASE)')
    parser.add_argument('--link', action='append', default=[], help='Optional additional links (repeatable)')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    reg = root / 'docs' / 'discoveries' / 'discoveries.json'
    reg.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(reg.read_text(encoding='utf-8')) if reg.exists() else []
        if not isinstance(data, list):
            raise ValueError('discoveries.json must be a JSON array')
    except Exception:
        data = []

    entry = {
        'date': datetime.utcnow().isoformat() + 'Z',
        'tier': args.tier,
        'title': args.title.strip(),
        'summary': args.summary.strip(),
        'evidence': args.evidence.strip(),
        'links': args.link or []
    }

    data.append(entry)
    reg.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Added discovery: {entry['title']} ({entry['tier']})")

if __name__ == '__main__':
    main()
