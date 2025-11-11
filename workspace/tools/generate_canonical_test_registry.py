# -*- coding: utf-8 -*-
"""Generate a single canonical test registry JSON consumed by:
  - Website (pass rate banner, tier listings)
  - Upload builders (templates, README, claims docs)
  - Future analytics (consistency verification)

Canonical semantics:
  executed_tests = tests with status in {PASS, FAIL}
  skipped_tests  = tests with status SKIP
  skip_exempt    = subset of skipped that are formally excluded from denominator
                   (physical invalidity for lattice model). Presently: GRAV-09.
  pass_rate_executed = passed / executed_tests
  public_pass_rate   = passed / (executed_tests)  (skip_exempt removed from totals everywhere)

File written to: workspace/results/test_registry_canonical.json

Input sources (authoritative order):
  1. MASTER_TEST_STATUS.csv (human+script generated summary)
  2. parallel_test_results.json (raw per-test entries; used for descriptions fallback)

If MASTER_TEST_STATUS.csv missing, we fall back to parallel_test_results.json statuses.

Idempotent: safe to run multiple times — overwrites file atomically.
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
MASTER = RESULTS_DIR / 'MASTER_TEST_STATUS.csv'
PARALLEL_JSON = RESULTS_DIR / 'parallel_test_results.json'
OUT_PATH = RESULTS_DIR / 'test_registry_canonical.json'

SKIP_EXEMPT = { 'GRAV-09': 'Discrete lattice cannot represent required continuous potential gradient resolution (documented physical invalidity).' }

def load_master():
    data: dict[str, dict] = {}
    if not MASTER.exists():
        return data
    lines = MASTER.read_text(encoding='utf-8').splitlines()
    collecting = False
    section: list[str] = []
    for line in lines:
        if line.startswith('Test_ID,'):
            collecting = True
            section = [line]
            continue
        if collecting:
            if line.startswith('TIER'):
                # parse accumulated section
                reader = csv.DictReader(section)
                for row in reader:
                    tid = (row.get('Test_ID') or '').strip()
                    if not tid:
                        continue
                    data[tid] = {
                        'id': tid,
                        'description': (row.get('Description') or '').strip(),
                        'status': (row.get('Status') or '').strip().upper(),
                        'energy_drift': row.get('Energy_Drift')
                    }
                section = []
                collecting = False
            else:
                section.append(line)
    # final section if file ended
    if section:
        reader = csv.DictReader(section)
        for row in reader:
            tid = (row.get('Test_ID') or '').strip()
            if not tid:
                continue
            data[tid] = {
                'id': tid,
                'description': (row.get('Description') or '').strip(),
                'status': (row.get('Status') or '').strip().upper(),
                'energy_drift': row.get('Energy_Drift')
            }
    return data

def load_parallel():
    if not PARALLEL_JSON.exists():
        return {}
    try:
        raw = json.loads(PARALLEL_JSON.read_text(encoding='utf-8'))
    except Exception:
        return {}
    tests = {}
    for entry in raw.get('tests', []):
        tid = entry.get('id') or entry.get('test_id')
        if not tid:
            continue
        tid = tid.strip()
        tests[tid] = {
            'id': tid,
            'description': entry.get('description') or entry.get('name') or '',
            'status': (str(entry.get('status') or '')).upper() or 'UNKNOWN'
        }
    return tests

def merge(master: dict, parallel: dict):
    merged = {}
    for tid, m in master.items():
        merged[tid] = m.copy()
    for tid, p in parallel.items():
        if tid not in merged:
            merged[tid] = p.copy()
        else:
            # backfill description if missing
            if not merged[tid].get('description') and p.get('description'):
                merged[tid]['description'] = p['description']
            # prefer master status
    return merged

def tier_of(tid: str) -> int:
    prefix = tid.split('-')[0]
    mapping = {
        'REL': 1,
        'GRAV': 2,
        'ENER': 3,
        'QUAN': 4,
        'EM': 5,
        'COUP': 6,
        'THERM': 7
    }
    return mapping.get(prefix, 0)

def build_registry():
    master = load_master()
    parallel = load_parallel()
    merged = merge(master, parallel)
    tiers: dict[int, dict] = {}
    passed = failed = skipped = exempt = 0
    for tid, info in merged.items():
        status = info.get('status', 'UNKNOWN').upper()
        tnum = tier_of(tid)
        tiers.setdefault(tnum, {'tests': []})
        is_exempt = tid in SKIP_EXEMPT and status == 'SKIP'
        record = {
            'id': tid,
            'description': info.get('description', ''),
            'status': status,
            'skip_exempt': is_exempt,
            'energy_drift': info.get('energy_drift')
        }
        tiers[tnum]['tests'].append(record)
        if status == 'PASS':
            passed += 1
        elif status == 'FAIL':
            failed += 1
        elif status == 'SKIP':
            skipped += 1
            if is_exempt:
                exempt += 1
    executed = passed + failed  # excludes all skips
    public_total = executed  # denominator excluding all skips
    pass_rate_executed = (passed / executed * 100) if executed else 0.0
    registry = {
        'generatedAt': datetime.utcnow().isoformat(),
        'sourceFiles': [str(MASTER), str(PARALLEL_JSON)],
        'skipExempt': SKIP_EXEMPT,
        'summary': {
            'total_including_skips': passed + failed + skipped,
            'skipped': skipped,
            'skip_exempt_count': exempt,
            'executed': executed,
            'passed': passed,
            'failed': failed,
            'public_total': public_total,
            'public_pass_rate': round(pass_rate_executed, 1),
            'formula': 'public_pass_rate = passed / (executed) ; executed excludes SKIP including exempt'
        },
        'tiers': {}
    }
    for tnum, data in tiers.items():
        # compute per-tier stats
        tp_pass = sum(1 for r in data['tests'] if r['status'] == 'PASS')
        tp_fail = sum(1 for r in data['tests'] if r['status'] == 'FAIL')
        tp_skip = sum(1 for r in data['tests'] if r['status'] == 'SKIP')
        registry['tiers'][str(tnum)] = {
            'executed': tp_pass + tp_fail,
            'passed': tp_pass,
            'failed': tp_fail,
            'skipped': tp_skip,
            'pass_rate': round((tp_pass / (tp_pass + tp_fail) * 100) if (tp_pass + tp_fail) else 0, 1),
            'tests': data['tests']
        }
    return registry

def main():
    reg = build_registry()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(reg, indent=2), encoding='utf-8')
    tmp.replace(OUT_PATH)
    print(f"✅ Canonical test registry written: {OUT_PATH}")
    print(f"   Pass Rate (executed): {reg['summary']['public_pass_rate']}%  (passed={reg['summary']['passed']}, executed={reg['summary']['executed']})")
    return 0

if __name__ == '__main__':
    sys.exit(main())
