#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression Evidence Guard

Purpose:
  Automated regression check combining physics validation metrics and
  evidence artifact completeness to prevent silent drift prior to
  upload (OSF / Zenodo) or release.

Features:
  1. Runs a specified set of tests (or tiers) via run_parallel_suite.py
     with GPU enabled (MANDATORY) and optional fused backend.
  2. Validates evidence artifacts using existing validate_evidence.py.
  3. Compares key physics metrics (primary metric + energy drift) against
     stored baselines. Fails if regression beyond tolerance.
  4. Baseline file auto-initialized on first run.
  5. Optional --refresh-manifests to rewrite artifact manifests so role
     classification (required/optional) reflects any schema changes.

Baseline Storage:
  workspace/results/_regression_baselines/baseline.json
  Structure:
    {
      "REL-11": {
        "primary_metric": 0.002735,
        "energy_drift": 4.40e-04,
        "timestamp": "2025-11-14T23:39:18Z"
      },
      ...
    }

CLI Usage Examples (PowerShell):
  cd c:\LFM\workspace\src
  & c:\LFM\.venv\Scripts\python.exe ..\tools\regression_evidence_guard.py --tests "REL-11,QUAN-10" --backend fused --update-baseline --refresh-manifests

  # Full representative sweep (quick baseline) across tiers (adjust list as needed):
  & c:\LFM\.venv\Scripts\python.exe ..\tools\regression_evidence_guard.py --tests "REL-01,GRAV-12,ENER-01,QUAN-10,EM-01,COUP-01,THERM-01" --backend fused

Exit Codes:
  0 = PASS (no regressions, evidence complete)
  2 = Evidence incomplete for at least one test
  3 = Physics regression (metric exceeded tolerance)
  4 = Internal error (unexpected exception)

Tolerances:
  - energy drift: must not exceed 2.0 * recorded baseline drift
  - primary metric: must not exceed max(baseline * 1.25, baseline + small_eps)
  - If baseline metric is zero, any non-zero below small_eps=1e-12 accepted.

Assumptions:
  - run_parallel_suite.py available in src directory
  - validate_evidence.py in tools directory
  - summary.json contains validation.primary metrics

"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

SMALL_EPS = 1e-12
BASELINE_DIR = Path(__file__).parent.parent / 'results' / '_regression_baselines'
BASELINE_PATH = BASELINE_DIR / 'baseline.json'
SRC_DIR = Path(__file__).parent.parent / 'src'
TOOLS_DIR = Path(__file__).parent


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def _run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding='utf-8')


def _parse_tests_arg(arg: str) -> List[str]:
    return [t.strip().upper() for t in arg.split(',') if t.strip()]


def _extract_metrics(summary_path: Path) -> Dict[str, float]:
    data = _load_json(summary_path)
    if not isinstance(data, dict):
        return {}
    primary_block = data.get('validation', {}).get('primary', {})
    return {
        'primary_metric': float(primary_block.get('value', data.get('primary_metric', 0.0)) or 0.0),
        'energy_drift': float(data.get('energy_drift') or data.get('validation', {}).get('energy', {}).get('drift', 0.0) or 0.0)
    }


def _validate_evidence_for_tests(test_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Invoke validate_evidence.py individually to obtain artifact completeness per test."""
    results: Dict[str, Dict[str, Any]] = {}
    validator = TOOLS_DIR / 'validate_evidence.py'
    for tid in test_ids:
        cmd = [sys.executable, str(validator), '--test', tid]
        proc = _run_cmd(cmd, SRC_DIR)
        out = proc.stdout
        missing = []
        complete = '✅ ALL TESTS HAVE COMPLETE EVIDENCE' in out or 'Complete Evidence: 1 (100.0%)' in out
        if not complete:
            # crude extraction of missing lines
            for line in out.splitlines():
                if line.strip().startswith('- '):
                    missing.append(line.strip().lstrip('- ').strip())
        results[tid] = {
            'complete': complete,
            'missing': missing,
            'raw': out,
            'exit_code': proc.returncode,
        }
    return results


def _refresh_manifests(test_ids: List[str]) -> None:
    # Re-run emit_summary_artifacts for each result dir to update role classification after schema changes.
    try:
        from utils.evidence import emit_summary_artifacts  # type: ignore
    except Exception:
        return
    for tid in test_ids:
        # Attempt multiple category guesses (directory naming differs per tier)
        root = BASELINE_DIR.parent / 'results'
        candidates = []
        if root.exists():
            for cat in root.iterdir():
                p = cat / tid
                if p.exists():
                    candidates.append(p)
        seen = set()
        for c in candidates:
            if c.exists() and c not in seen:
                try:
                    emit_summary_artifacts(c, None)
                except Exception:
                    pass
                seen.add(c)


def main(argv: List[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description='Regression Evidence Guard')
    parser.add_argument('--tests', type=str, help='Comma-separated test IDs (e.g. REL-01,QUAN-10)')
    parser.add_argument('--tiers', type=str, help='Comma-separated tier numbers (overrides --tests if provided)')
    parser.add_argument('--backend', type=str, default='fused', help='Physics backend (baseline|fused)')
    parser.add_argument('--max-concurrent', type=int, default=8)
    parser.add_argument('--update-baseline', action='store_true', help='Update baseline after successful run')
    parser.add_argument('--refresh-manifests', action='store_true', help='Regenerate artifact manifests post-schema change')
    parser.add_argument('--tolerances', type=str, default='', help='Optional JSON string overriding tolerances')
    args = parser.parse_args(argv)

    if not (args.tests or args.tiers):
        print('ERROR: Specify --tests or --tiers')
        return 4

    # Derive test IDs: if tiers provided we run full tier(s) and collect IDs from results after run
    test_ids: List[str] = []
    if args.tests:
        test_ids = _parse_tests_arg(args.tests)

    # Run parallel suite
    suite_script = SRC_DIR / 'run_parallel_suite.py'
    cmd: List[str]
    if args.tiers:
        cmd = [sys.executable, str(suite_script), '--tiers', args.tiers, '--backend', args.backend, '--max-concurrent', str(args.max_concurrent)]
    else:
        cmd = [sys.executable, str(suite_script), '--tests', ','.join(test_ids), '--backend', args.backend, '--max-concurrent', str(args.max_concurrent)]
    print('=== Running test suite for regression check ===')
    proc = _run_cmd(cmd, SRC_DIR)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f'ERROR: parallel suite returned {proc.returncode}')
        return 4

    # If tiers used, attempt to enumerate test IDs from MASTER_TEST_STATUS.csv
    if args.tiers:
        status_csv = SRC_DIR / 'MASTER_TEST_STATUS.csv'
        if status_csv.exists():
            lines = status_csv.read_text(encoding='utf-8').splitlines()
            for ln in lines:
                if ln.startswith('REL-') or ln.startswith('GRAV-') or ln.startswith('ENER-') or ln.startswith('QUAN-') or ln.startswith('EM-') or ln.startswith('COUP-') or ln.startswith('THERM-'):
                    parts = ln.split(',')
                    test_ids.append(parts[0].strip())
            test_ids = sorted(set(test_ids))
        else:
            print('WARN: MASTER_TEST_STATUS.csv not found; baseline comparison may be incomplete.')

    # Evidence validation
    evidence_results = _validate_evidence_for_tests(test_ids)
    incomplete = [tid for tid, r in evidence_results.items() if not r['complete']]
    if incomplete:
        print('EVIDENCE INCOMPLETE for tests:', ', '.join(incomplete))
        if args.refresh_manifests:
            print('Attempting manifest refresh...')
            _refresh_manifests(incomplete)
            evidence_results = _validate_evidence_for_tests(incomplete)
            incomplete = [tid for tid, r in evidence_results.items() if not r['complete']]
        if incomplete:
            print('FAIL: Evidence completeness regression detected.')
            return 2

    # Collect metrics
    metrics_map: Dict[str, Dict[str, float]] = {}
    results_root = SRC_DIR.parent / 'results'
    for tid in test_ids:
        # Search for summary.json under any category folder
        summary_paths = list(results_root.glob(f'*/*{tid}/summary.json'))
        for sp in summary_paths:
            metrics_map[tid] = _extract_metrics(sp)
            break
        if tid not in metrics_map:
            print(f'WARN: No summary.json found for {tid}')
            metrics_map[tid] = {'primary_metric': 0.0, 'energy_drift': 0.0}

    baseline = _load_json(BASELINE_PATH) or {}

    # Compare metrics
    regressions: Dict[str, Dict[str, float]] = {}
    for tid, m in metrics_map.items():
        base = baseline.get(tid)
        if not base:
            continue  # new test; allow baseline update later
        # Energy drift tolerance
        allowed_drift = 2.0 * float(base.get('energy_drift', 0.0)) + SMALL_EPS
        if m['energy_drift'] > allowed_drift:
            regressions.setdefault(tid, {})['energy_drift'] = m['energy_drift']
        # Primary metric tolerance
        base_pm = float(base.get('primary_metric', 0.0))
        if base_pm <= SMALL_EPS:
            if m['primary_metric'] > SMALL_EPS * 10:
                regressions.setdefault(tid, {})['primary_metric'] = m['primary_metric']
        else:
            allowed_pm = max(base_pm * 1.25, base_pm + SMALL_EPS)
            if m['primary_metric'] > allowed_pm:
                regressions.setdefault(tid, {})['primary_metric'] = m['primary_metric']

    if regressions:
        print('Physics regressions detected:')
        for tid, vals in regressions.items():
            print(f'  {tid}: ' + ', '.join(f'{k}={v:.3e}' for k, v in vals.items()))
        return 3

    # Update baseline if requested
    if args.update_baseline:
        for tid, m in metrics_map.items():
            baseline[tid] = {
                'primary_metric': m['primary_metric'],
                'energy_drift': m['energy_drift'],
                'timestamp': _now_iso()
            }
        _write_json(BASELINE_PATH, baseline)
        print(f'Baseline updated: {BASELINE_PATH}')

    # Optional manifest refresh for all tests (post-schema adjustments)
    if args.refresh_manifests:
        _refresh_manifests(test_ids)
        print('Manifests refreshed.')

    print('Regression Evidence Guard PASS')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print('Interrupted by user')
        sys.exit(4)
    except Exception as e:
        print(f'Unexpected error: {type(e).__name__}: {e}')
        sys.exit(4)
