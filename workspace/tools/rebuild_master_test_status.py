# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Rebuild MASTER_TEST_STATUS.csv from actual test results.
Scans workspace/results/ directories for test summaries AND config files for skipped tests.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

def load_config_tests(config_dir: Path) -> Dict[str, Dict]:
    """Load test definitions from config files to find skipped tests."""
    config_tests = {}
    
    # Config file mappings
    config_files = {
        "Tier 1": "config_tier1_relativistic.json",
        "Tier 2": "config_tier2_gravityanalogue.json",
        "Tier 3": "config_tier3_energy.json",
        "Tier 4": "config_tier4_quantization.json",
        "Tier 5": "config_tier5_electromagnetic.json",
        "Tier 6": "config_tier6_coupling.json",
        "Tier 7": "config_tier7_thermodynamics.json"
    }
    
    tier_categories = {
        "Tier 1": "Relativistic",
        "Tier 2": "Gravity",
        "Tier 3": "Energy",
        "Tier 4": "Quantization",
        "Tier 5": "Electromagnetic",
        "Tier 6": "Coupling",
        "Tier 7": "Thermodynamics"
    }
    
    for tier_label, config_file in config_files.items():
        config_path = config_dir / config_file
        if not config_path.exists():
            continue
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Handle both "variants" and "tests" schema
            tests = config.get('tests', config.get('variants', []))
            
            for test in tests:
                test_id = test.get('test_id', test.get('id'))
                if not test_id:
                    continue
                
                skip = test.get('skip', False)
                skip_reason = test.get('skip_reason', '')
                description = test.get('description', '')
                
                if skip:
                    config_tests[test_id] = {
                        'tier': tier_label,
                        'category': tier_categories[tier_label],
                        'status': 'SKIP',
                        'skip_reason': skip_reason,
                        'description': description
                    }
        except Exception as e:
            print(f"Warning: Could not read {config_path}: {e}")
    
    return config_tests

def scan_test_results(results_dir: Path, config_tests: Dict[str, Dict]) -> Dict[str, Dict]:
    """Scan results directories for test summaries and merge with skipped tests from configs."""
    test_data = {}
    
    # First, add all skipped tests from configs
    test_data.update(config_tests)
    
    # Tier mappings
    tier_dirs = {
        "Relativistic": "Tier 1",
        "Gravity": "Tier 2",
        "Energy": "Tier 3",
        "Quantization": "Tier 4",
        "Electromagnetic": "Tier 5",
        "Coupling": "Tier 6",
        "Thermodynamics": "Tier 7"
    }
    
    for tier_dir_name, tier_label in tier_dirs.items():
        tier_path = results_dir / tier_dir_name
        if not tier_path.exists():
            continue
            
        # Scan for test directories
        for test_dir in tier_path.iterdir():
            if not test_dir.is_dir():
                continue
            
            test_id = test_dir.name
            summary_file = test_dir / "summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    # Check if test is marked as skipped in summary
                    if summary.get('skipped', False):
                        status = 'SKIP'
                        skip_reason = summary.get('skip_reason', '')
                    else:
                        # Handle both "passed": true/false and "status": "Passed"/"Failed"
                        passed = summary.get('passed', False)
                        status_str = summary.get('status', '').lower()
                        if status_str in ['passed', 'pass']:
                            passed = True
                        elif status_str in ['failed', 'fail']:
                            passed = False
                        status = 'PASS' if passed else 'FAIL'
                        skip_reason = ''
                    
                    # Override config entry if test actually ran (or update skip info)
                    test_data[test_id] = {
                        'tier': tier_label,
                        'category': tier_dir_name,
                        'status': status,
                        'skip_reason': skip_reason,
                        'description': summary.get('description', ''),
                        'runtime_sec': summary.get('runtime_sec', 0.0),
                        'timestamp': summary.get('timestamp', '')
                    }
                except Exception as e:
                    print(f"Warning: Could not read {summary_file}: {e}")
    
    return test_data

def generate_master_csv(test_data: Dict[str, Dict], output_file: Path):
    """Generate MASTER_TEST_STATUS.csv from test data."""
    
    # Count tests by tier
    tier_stats = {
        "Tier 1": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Relativistic"},
        "Tier 2": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Gravity Analogue"},
        "Tier 3": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Energy Conservation"},
        "Tier 4": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Quantization"},
        "Tier 5": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Electromagnetic"},
        "Tier 6": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Coupling"},
        "Tier 7": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "category": "Thermodynamics & Statistical Mechanics"}
    }
    
    for test_id, data in test_data.items():
        tier = data['tier']
        status = data.get('status', 'UNKNOWN')
        tier_stats[tier]['total'] += 1
        if status == 'PASS':
            tier_stats[tier]['passed'] += 1
        elif status == 'FAIL':
            tier_stats[tier]['failed'] += 1
        elif status == 'SKIP':
            tier_stats[tier]['skipped'] += 1
    
    # Generate CSV
    lines = []
    lines.append("MASTER TEST STATUS REPORT - LFM Lattice Field Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Validation Rule: Suite marked NOT RUN if any test missing from CSV")
    lines.append("")
    lines.append("CATEGORY SUMMARY")
    lines.append("Tier,Category,Total_Tests,Passed,Failed,Skipped,Pass_Rate")
    
    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        total = stats['total']
        passed = stats['passed']
        failed = stats['failed']
        skipped = stats['skipped']
        runnable = total - skipped
        pass_rate = f"{100*passed/runnable:.1f}%" if runnable > 0 else "0%"
        lines.append(f"{tier},{stats['category']},{total},{passed},{failed},{skipped},{pass_rate}")
    
    lines.append("")
    lines.append("DETAILED TEST RESULTS")
    lines.append("Test_ID,Tier,Category,Status,Description,Skip_Reason,Runtime_Sec,Timestamp")
    
    for test_id in sorted(test_data.keys()):
        data = test_data[test_id]
        status = data.get('status', 'UNKNOWN')
        desc = data['description'].replace(',', ';')  # Escape commas
        skip_reason = data.get('skip_reason', '').replace(',', ';')  # Escape commas
        runtime = f"{data.get('runtime_sec', 0.0):.3f}"
        timestamp = data.get('timestamp', '')
        
        lines.append(f"{test_id},{data['tier']},{data['category']},{status},{desc},{skip_reason},{runtime},{timestamp}")
    
    # Write file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Generated {output_file}")
    print(f"   Total tests: {len(test_data)}")
    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        if stats['total'] > 0:
            parts = [f"{stats['passed']}/{stats['total']} passed"]
            if stats['failed'] > 0:
                parts.append(f"{stats['failed']} failed")
            if stats['skipped'] > 0:
                parts.append(f"{stats['skipped']} skipped")
            print(f"   {tier}: {', '.join(parts)}")

if __name__ == "__main__":
    workspace_root = Path(__file__).parent.parent
    results_dir = workspace_root / "results"
    config_dir = workspace_root / "config"
    output_file = results_dir / "MASTER_TEST_STATUS.csv"
    
    print("Loading test configurations...")
    config_tests = load_config_tests(config_dir)
    print(f"Found {len(config_tests)} skipped tests in configs")
    
    print("Scanning test results...")
    test_data = scan_test_results(results_dir, config_tests)
    
    print(f"Total tests tracked: {len(test_data)}")
    
    print("Generating MASTER_TEST_STATUS.csv...")
    generate_master_csv(test_data, output_file)
    
    print("\n✅ Done!")
