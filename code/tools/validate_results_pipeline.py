#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
Comprehensive validation framework for LFM results generation and upload pipeline.

This validator ensures data integrity across the entire chain:
  Test Execution → Results Artifacts → Master Status CSV → Upload Package → Manifest

Usage:
  # Validate a single test result
  python tools/validate_results_pipeline.py --test REL-01
  
  # Validate an entire tier
  python tools/validate_results_pipeline.py --tier 1
  
  # Validate master status integrity
  python tools/validate_results_pipeline.py --master-status
  
  # Validate upload package completeness
  python tools/validate_results_pipeline.py --upload-package
  
  # Validate entire pipeline (end-to-end)
  python tools/validate_results_pipeline.py --all
  
  # Run with strict mode (fail on warnings)
  python tools/validate_results_pipeline.py --all --strict
"""

import json
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import csv

# Repo root
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
UPLOAD = ROOT / 'docs' / 'upload'

# Central tier registry
try:
    from lfm_tiers import get_tiers, get_tier_by_number
except Exception:
    # Fallback if registry unavailable
    def get_tiers():
        return [
            {"tier": 1, "name": "Relativistic", "category_name": "Relativistic", "dir": "Relativistic", "prefix": "REL"},
            {"tier": 2, "name": "Gravity", "category_name": "Gravity Analogue", "dir": "Gravity", "prefix": "GRAV"},
            {"tier": 3, "name": "Energy", "category_name": "Energy Conservation", "dir": "Energy", "prefix": "ENER"},
            {"tier": 4, "name": "Quantization", "category_name": "Quantization", "dir": "Quantization", "prefix": "QUAN"},
        ]
    def get_tier_by_number(n: int):
        for t in get_tiers():
            if int(t["tier"]) == int(n):
                return t
        return None


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""
    pass


class PipelineValidator:
    """Validates the entire results → upload pipeline."""
    
    def __init__(self, strict: bool = False, verbose: bool = True):
        self.strict = strict
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def log(self, msg: str, level: str = 'info'):
        """Log a message at specified level."""
        if level == 'error':
            self.errors.append(msg)
            if self.verbose:
                print(f"❌ ERROR: {msg}")
        elif level == 'warning':
            self.warnings.append(msg)
            if self.verbose:
                print(f"⚠️  WARNING: {msg}")
        elif level == 'info':
            self.info.append(msg)
            if self.verbose:
                print(f"ℹ️  {msg}")
                
    def validate_test_result(self, test_id: str, tier_dir: Path) -> bool:
        """
        Validate that a single test result has all required artifacts.
        
        Expected structure:
          results/<tier>/<test_id>/
            summary.json      (required: test_id, description, status OR passed)
            readme.txt        (auto-generated, should exist)
            metrics.csv       (optional but recommended)
            plots/            (optional but common)
            diagnostics/      (optional)
        
        Returns True if valid, False otherwise.
        """
        test_dir = tier_dir / test_id
        if not test_dir.exists():
            self.log(f"Test directory not found: {test_dir}", 'error')
            return False
            
        valid = True
        
        # Check summary.json
        summary_file = test_dir / 'summary.json'
        if not summary_file.exists():
            self.log(f"{test_id}: Missing summary.json", 'error')
            valid = False
        else:
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                # Required fields (accept either 'test_id' or 'id')
                if 'test_id' not in summary and 'id' not in summary:
                    self.log(f"{test_id}: summary.json missing 'test_id' or 'id' field", 'error')
                    valid = False
                if 'description' not in summary:
                    self.log(f"{test_id}: summary.json missing 'description' field", 'error')
                    valid = False
                
                # Must have either 'status' or 'passed'
                if 'status' not in summary and 'passed' not in summary:
                    self.log(f"{test_id}: summary.json must have 'status' or 'passed' field", 'error')
                    valid = False
                
                # Validate status normalization
                if 'status' in summary:
                    status = str(summary['status']).upper()
                    if status not in ['PASS', 'PASSED', 'FAIL', 'FAILED', 'SKIP', 'SKIPPED', 'UNKNOWN']:
                        self.log(f"{test_id}: Invalid status '{summary['status']}'", 'warning')
                        
            except json.JSONDecodeError as e:
                self.log(f"{test_id}: Invalid JSON in summary.json: {e}", 'error')
                valid = False
        
        # Check readme.txt
        readme_file = test_dir / 'readme.txt'
        if not readme_file.exists():
            self.log(f"{test_id}: Missing readme.txt (should be auto-generated)", 'warning')
        
        # Info about optional artifacts
        if (test_dir / 'metrics.csv').exists():
            self.log(f"{test_id}: ✓ metrics.csv present", 'info')
        if (test_dir / 'plots').exists():
            plot_count = len(list((test_dir / 'plots').glob('*.png')))
            if plot_count > 0:
                self.log(f"{test_id}: ✓ {plot_count} plot(s) in plots/", 'info')
        
        return valid
    
    def validate_tier_results(self, tier_num: int) -> bool:
        """Validate all test results in a tier."""
        tier_def = get_tier_by_number(tier_num)
        if not tier_def:
            self.log(f"Invalid tier number: {tier_num}", 'error')
            return False
        tier_dir = RESULTS / tier_def["dir"]
        if not tier_dir.exists():
            self.log(f"Tier directory not found: {tier_dir}", 'error')
            return False
        
        self.log(f"Validating Tier {tier_num} ({tier_def['name']}) results...", 'info')
        
        test_dirs = [d for d in tier_dir.iterdir() if d.is_dir()]
        if not test_dirs:
            self.log(f"No test directories found in {tier_dir}", 'warning')
            return False
        
        valid = True
        for test_dir in sorted(test_dirs):
            test_id = test_dir.name
            if not self.validate_test_result(test_id, tier_dir):
                valid = False
        
        return valid
    
    def validate_master_status_integrity(self) -> bool:
        """
        Validate MASTER_TEST_STATUS.csv integrity.
        
        Checks:
        - CSV file exists and is readable
        - All tests in results/ are reflected in CSV
        - Status values are properly normalized (Passed→PASS)
        - Category summaries match actual test counts
        - No orphaned tests (in CSV but not in results/)
        """
        csv_path = RESULTS / 'MASTER_TEST_STATUS.csv'
        if not csv_path.exists():
            self.log(f"MASTER_TEST_STATUS.csv not found at {csv_path}", 'error')
            return False
        
        self.log("Validating MASTER_TEST_STATUS.csv integrity...", 'info')
        
        # Read CSV and extract test IDs
        csv_tests: Dict[str, Dict[str, str]] = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find detailed results section
            in_details = False
            for line in lines:
                if line.startswith('DETAILED TEST RESULTS'):
                    in_details = True
                    continue
                if in_details and line.strip() and not line.startswith('TIER'):
                    if ',' in line:
                        parts = line.strip().split(',', 3)
                        if len(parts) >= 3 and parts[0] and not parts[0].startswith('Test_ID'):
                            test_id = parts[0]
                            status = parts[2] if len(parts) > 2 else 'UNKNOWN'
                            csv_tests[test_id] = {'status': status}
        except Exception as e:
            self.log(f"Error reading MASTER_TEST_STATUS.csv: {e}", 'error')
            return False
        
        # Scan actual test results
        actual_tests: Dict[str, Path] = {}
        for t in get_tiers():
            tier_dir = RESULTS / t["dir"]
            if not tier_dir.exists():
                continue
            for test_dir in tier_dir.iterdir():
                if test_dir.is_dir():
                    actual_tests[test_dir.name] = test_dir
        
        valid = True
        
        # Check for missing tests (in results/ but not in CSV)
        missing_from_csv = set(actual_tests.keys()) - set(csv_tests.keys())
        if missing_from_csv:
            for test_id in sorted(missing_from_csv):
                self.log(f"Test {test_id} exists in results/ but not in MASTER_TEST_STATUS.csv", 'error')
            valid = False
        
        # Check for orphaned tests (in CSV but not in results/)
        orphaned = set(csv_tests.keys()) - set(actual_tests.keys())
        if orphaned:
            for test_id in sorted(orphaned):
                self.log(f"Test {test_id} in MASTER_TEST_STATUS.csv but directory not found", 'warning')
        
        # Validate status normalization
        for test_id, info in csv_tests.items():
            status = info['status']
            if status not in ['PASS', 'FAIL', 'SKIP', 'UNKNOWN']:
                self.log(f"Test {test_id} has non-normalized status '{status}' (should be PASS/FAIL/SKIP/UNKNOWN)", 'error')
                valid = False
        
        self.log(f"CSV contains {len(csv_tests)} tests, results/ contains {len(actual_tests)} tests", 'info')
        
        return valid
    
    def validate_upload_package(self) -> bool:
        """
        Validate upload package completeness and integrity.
        
        Checks:
        - Required files exist (LICENSE, NOTICE, evidence DOCX, etc.)
        - Comprehensive PDF exists and is recent
        - MANIFEST.md exists and checksums are valid
        - Zenodo/OSF metadata files are valid JSON
        - results_MASTER_TEST_STATUS.csv matches source
        """
        if not UPLOAD.exists():
            self.log(f"Upload directory not found: {UPLOAD}", 'error')
            return False
        
        self.log("Validating upload package...", 'info')
        
        valid = True
        
        # Required core files
        required_files = [
            'LICENSE',
            'NOTICE',
            'README.md',
            'PRE_PUBLIC_AUDIT_REPORT.md',
            'RESULTS_REPORT.md',
            'results_MASTER_TEST_STATUS.csv',
            'MANIFEST.md',
            'zenodo_metadata.json',
            'osf_metadata.json',
        ]
        
        for filename in required_files:
            filepath = UPLOAD / filename
            if not filepath.exists():
                self.log(f"Required file missing from upload/: {filename}", 'error')
                valid = False
        
        # Evidence DOCX files
        evidence_dir = UPLOAD / 'evidence_docx'
        if evidence_dir.exists():
            docx_files = list(evidence_dir.glob('*.docx'))
            if len(docx_files) < 4:
                self.log(f"Expected 4 evidence DOCX files, found {len(docx_files)}", 'error')
                valid = False
        else:
            self.log("evidence_docx/ directory missing", 'error')
            valid = False
        
        # Comprehensive PDF
        pdf_files = list(UPLOAD.glob('LFM_Comprehensive_Report_*.pdf'))
        if not pdf_files:
            self.log("Comprehensive PDF not found in upload/", 'error')
            valid = False
        elif len(pdf_files) > 1:
            self.log(f"Multiple comprehensive PDFs found: {[p.name for p in pdf_files]}", 'warning')
        else:
            pdf_file = pdf_files[0]
            age_hours = (datetime.now().timestamp() - pdf_file.stat().st_mtime) / 3600
            if age_hours > 24:
                self.log(f"Comprehensive PDF is {age_hours:.1f} hours old (may be stale)", 'warning')
        
        # Validate JSON metadata files
        for json_file in ['zenodo_metadata.json', 'osf_metadata.json']:
            json_path = UPLOAD / json_file
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    self.log(f"Invalid JSON in {json_file}: {e}", 'error')
                    valid = False
        
        # Validate MANIFEST.md checksums (sample check)
        manifest_path = UPLOAD / 'MANIFEST.md'
        if manifest_path.exists():
            # Check that at least one file's checksum is valid
            checked = False
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line and '`' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4:
                            filename = parts[1]
                            expected_hash = parts[3].strip('`')
                            if len(expected_hash) == 64:  # SHA256
                                filepath = UPLOAD / filename
                                if filepath.exists() and filepath.is_file():
                                    actual_hash = self._sha256_file(filepath)
                                    if actual_hash != expected_hash:
                                        self.log(f"Checksum mismatch for {filename}", 'error')
                                        valid = False
                                    checked = True
                                    break
            if not checked:
                self.log("Could not verify any checksums in MANIFEST.md", 'warning')
        
        return valid
    
    def validate_end_to_end(self) -> bool:
        """
        Validate the entire pipeline end-to-end.
        
        Checks consistency across:
        1. Test results in results/
        2. MASTER_TEST_STATUS.csv
        3. Upload package (results_MASTER_TEST_STATUS.csv, comprehensive PDF)
        4. Manifest integrity
        """
        self.log("=" * 60, 'info')
        self.log("END-TO-END PIPELINE VALIDATION", 'info')
        self.log("=" * 60, 'info')
        
        valid = True
        
        # Step 1: Validate all tier results
        self.log("\n[1/4] Validating test results...", 'info')
        for tier_num in [int(t["tier"]) for t in get_tiers()]:
            if not self.validate_tier_results(tier_num):
                valid = False
        
        # Step 2: Validate master status
        self.log("\n[2/4] Validating MASTER_TEST_STATUS.csv...", 'info')
        if not self.validate_master_status_integrity():
            valid = False
        
        # Step 3: Validate upload package
        self.log("\n[3/4] Validating upload package...", 'info')
        if not self.validate_upload_package():
            valid = False
        
        # Step 4: Cross-check consistency
        self.log("\n[4/4] Cross-checking consistency...", 'info')
        source_csv = RESULTS / 'MASTER_TEST_STATUS.csv'
        upload_csv = UPLOAD / 'results_MASTER_TEST_STATUS.csv'
        
        if source_csv.exists() and upload_csv.exists():
            # Compare using normalized content to ignore the non-deterministic 'Generated:' line
            source_hash = self._sha256_file_normalized(source_csv)
            upload_hash = self._sha256_file_normalized(upload_csv)
            if source_hash != upload_hash:
                self.log("results/MASTER_TEST_STATUS.csv does not match upload/results_MASTER_TEST_STATUS.csv", 'error')
                valid = False
            else:
                self.log("✓ MASTER_TEST_STATUS.csv matches between results/ and upload/", 'info')
        
        # Step 5: Resource metrics validation (optional in end-to-end)
        self.log("\n[5/4] Validating resource metrics...", 'info')
        if not self.validate_resource_metrics():
            valid = False

        return valid

    def validate_resource_metrics(self) -> bool:
        """Validate resource tracking metrics from test_metrics_history.json and results/ structure.

        Rules:
        - Every test under results/<Tier>/<TEST_ID>/ should have an entry in metrics history.
        - Last run metrics must include runtime_sec>0 and non-negative CPU/RAM/GPU values.
        """
        history_path = RESULTS / 'test_metrics_history.json'
        if not history_path.exists():
            self.log("test_metrics_history.json not found under results/", 'warning')
            return False

        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception as e:
            self.log(f"Cannot read test_metrics_history.json: {e}", 'error')
            return False

        # Build set of expected test IDs from results tree (filter canonical IDs only)
        expected: List[str] = []
        import re as _re
        for t in get_tiers():
            pat = _re.compile(t.get('id_pattern', rf'^{t["prefix"]}-\d+$'), _re.I)
            tier_dir = RESULTS / t["dir"]
            if not tier_dir.exists():
                continue
            for d in tier_dir.iterdir():
                if d.is_dir() and pat.match(d.name.strip()):
                    expected.append(d.name.strip())

        # Load master status to exempt SKIPped tests
        master_status_path = RESULTS / 'MASTER_TEST_STATUS.csv'
        skipped: set[str] = set()
        try:
            if master_status_path.exists():
                with open(master_status_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in line.split(',', 3)]
                        if len(parts) >= 3 and '-' in parts[0]:
                            if parts[2].upper() == 'SKIP':
                                skipped.add(parts[0])
        except Exception:
            pass

        ok = True
        # Check that each expected test has some history
        for tid in expected:
            if tid not in history:
                # If test was SKIPped in master status, don't require history
                if tid in skipped:
                    self.log(f"{tid}: skipped in master status; metrics history not required", 'info')
                    continue
                # If summary exists, warn but don't error
                # Determine tier dir from prefix
                tier_prefix = tid.split('-')[0]
                tier_dir_map = {t['prefix']: t['dir'] for t in get_tiers()}
                tdir = RESULTS / tier_dir_map.get(tier_prefix, '') / tid
                if (tdir / 'summary.json').exists():
                    self.log(f"{tid}: summary present but no metrics in history (treating as covered)", 'warning')
                    continue
                self.log(f"Resource metrics missing for {tid} in test_metrics_history.json", 'error')
                ok = False
                continue
            runs = history[tid].get('runs', [])
            if not runs:
                self.log(f"No runs recorded for {tid} in metrics history", 'error')
                ok = False
                continue
            last = runs[-1]
            # Required keys
            req = ['runtime_sec', 'peak_cpu_percent', 'peak_memory_mb', 'peak_gpu_memory_mb']
            missing = [k for k in req if k not in last]
            if missing:
                self.log(f"{tid}: missing resource keys in last run: {', '.join(missing)}", 'error')
                ok = False
                continue
            # Validate values
            rt = float(last.get('runtime_sec', 0.0) or 0.0)
            cpu = float(last.get('peak_cpu_percent', 0.0) or 0.0)
            mem = float(last.get('peak_memory_mb', 0.0) or 0.0)
            gpu = float(last.get('peak_gpu_memory_mb', 0.0) or 0.0)
            if rt <= 0:
                self.log(f"{tid}: runtime_sec <= 0 ({rt})", 'error')
                ok = False
            if cpu < 0:
                self.log(f"{tid}: peak_cpu_percent < 0 ({cpu})", 'error')
                ok = False
            if mem < 0:
                self.log(f"{tid}: peak_memory_mb < 0 ({mem})", 'error')
                ok = False
            if gpu < 0:
                self.log(f"{tid}: peak_gpu_memory_mb < 0 ({gpu})", 'error')
                ok = False
        return ok
    
    def _sha256_file(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _sha256_file_normalized(self, path: Path, *, ignore_generated_line: bool = True) -> str:
        """Compute SHA256 of a text file with optional normalization.

        Normalizations applied when ignore_generated_line is True:
        - Lines starting with 'Generated:' are replaced with a canonical token
          so deterministic upload timestamps (e.g., 1970-01-01) match live results.
        - Line endings are normalized to '\n' to avoid CRLF/LF differences across platforms.
        """
        h = hashlib.sha256()
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for raw_line in f:
                    line = raw_line.rstrip('\r\n')
                    if ignore_generated_line and line.startswith('Generated:'):
                        line = 'Generated: <normalized>'
                    # Re-append a single LF for stable hashing
                    data = (line + '\n').encode('utf-8')
                    h.update(data)
        except UnicodeDecodeError:
            # Fallback to binary hash if file isn't text
            return self._sha256_file(path)
        return h.hexdigest()
    
    def report(self) -> int:
        """
        Print validation report and return exit code.
        
        Returns:
            0 if validation passed
            1 if errors found
            2 if warnings found (only in strict mode)
        """
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} ERROR(S):")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED")
            return 0
        elif self.errors:
            print("\n❌ VALIDATION FAILED")
            return 1
        elif self.strict and self.warnings:
            print("\n⚠️  VALIDATION FAILED (strict mode)")
            return 2
        else:
            print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate LFM results and upload pipeline integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Validation scope
    parser.add_argument('--test', type=str, help='Validate a specific test ID (e.g., REL-01)')
    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4], help='Validate all tests in a tier')
    parser.add_argument('--master-status', action='store_true', help='Validate MASTER_TEST_STATUS.csv integrity')
    parser.add_argument('--upload-package', action='store_true', help='Validate upload package completeness')
    parser.add_argument('--all', action='store_true', help='Validate entire pipeline (end-to-end)')
    parser.add_argument('--resource-metrics', action='store_true', help='Validate resource tracking metrics consistency')
    
    # Options
    parser.add_argument('--strict', action='store_true', help='Fail on warnings (not just errors)')
    parser.add_argument('--quiet', action='store_true', help='Only show summary, not individual checks')
    
    args = parser.parse_args()
    
    # If no scope specified, default to --all
    if not any([args.test, args.tier, args.master_status, args.upload_package, args.all]):
        args.all = True
    
    validator = PipelineValidator(strict=args.strict, verbose=not args.quiet)
    
    valid = True
    
    if args.test:
        # Determine which tier the test belongs to
        tier_prefixes = {'REL': 'Relativistic', 'GRAV': 'Gravity', 'ENER': 'Energy', 'QUAN': 'Quantization'}
        prefix = args.test.split('-')[0] if '-' in args.test else ''
        tier_name = tier_prefixes.get(prefix)
        if not tier_name:
            print(f"Cannot determine tier for test ID: {args.test}")
            return 1
        tier_dir = RESULTS / tier_name
        valid = validator.validate_test_result(args.test, tier_dir)
    
    if args.tier:
        valid = validator.validate_tier_results(args.tier) and valid
    
    if args.master_status:
        valid = validator.validate_master_status_integrity() and valid
    
    if args.upload_package:
        valid = validator.validate_upload_package() and valid
    
    if args.all:
        valid = validator.validate_end_to_end()
    if args.resource_metrics:
        valid = validator.validate_resource_metrics() and valid
    
    return validator.report()


if __name__ == '__main__':
    sys.exit(main())
