# -*- coding: utf-8 -*-
"""
Rebuild MASTER_TEST_STATUS.csv from actual test results.
Scans workspace/results/ directories for test summaries.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def scan_test_results(results_dir: Path) -> Dict[str, Dict]:
    """Scan results directories for test summaries."""
    test_data = {}
    
    # Tier mappings
    tier_dirs = {
        "Relativistic": "Tier 1",
        "Gravity": "Tier 2",
        "Energy": "Tier 3",
        "Quantization": "Tier 4",
        "Electromagnetic": "Tier 5",
        "Coupling": "Tier 6"
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
                    
                    # Handle both "passed": true/false and "status": "Passed"/"Failed"
                    passed = summary.get('passed', False)
                    status = summary.get('status', '').lower()
                    if status in ['passed', 'pass']:
                        passed = True
                    elif status in ['failed', 'fail']:
                        passed = False
                    
                    test_data[test_id] = {
                        'tier': tier_label,
                        'category': tier_dir_name,
                        'passed': passed,
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
        "Tier 1": {"total": 0, "passed": 0, "category": "Relativistic"},
        "Tier 2": {"total": 0, "passed": 0, "category": "Gravity Analogue"},
        "Tier 3": {"total": 0, "passed": 0, "category": "Energy Conservation"},
        "Tier 4": {"total": 0, "passed": 0, "category": "Quantization"},
        "Tier 5": {"total": 0, "passed": 0, "category": "Electromagnetic"},
        "Tier 6": {"total": 0, "passed": 0, "category": "Coupling"}
    }
    
    for test_id, data in test_data.items():
        tier = data['tier']
        tier_stats[tier]['total'] += 1
        if data['passed']:
            tier_stats[tier]['passed'] += 1
    
    # Generate CSV
    lines = []
    lines.append("MASTER TEST STATUS REPORT - LFM Lattice Field Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Validation Rule: Suite marked NOT RUN if any test missing from CSV")
    lines.append("")
    lines.append("CATEGORY SUMMARY")
    lines.append("Tier,Category,Total_Tests,Passed,Failed,Pass_Rate")
    
    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        total = stats['total']
        passed = stats['passed']
        failed = total - passed
        pass_rate = f"{100*passed/total:.1f}%" if total > 0 else "0%"
        lines.append(f"{tier},{stats['category']},{total},{passed},{failed},{pass_rate}")
    
    lines.append("")
    lines.append("DETAILED TEST RESULTS")
    lines.append("Test_ID,Tier,Category,Status,Description,Runtime_Sec,Timestamp")
    
    for test_id in sorted(test_data.keys()):
        data = test_data[test_id]
        status = "PASS" if data['passed'] else "FAIL"
        desc = data['description'].replace(',', ';')  # Escape commas
        runtime = f"{data['runtime_sec']:.3f}"
        timestamp = data['timestamp']
        
        lines.append(f"{test_id},{data['tier']},{data['category']},{status},{desc},{runtime},{timestamp}")
    
    # Write file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Generated {output_file}")
    print(f"   Total tests: {len(test_data)}")
    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        if stats['total'] > 0:
            print(f"   {tier}: {stats['passed']}/{stats['total']} passed")

if __name__ == "__main__":
    workspace_root = Path(__file__).parent.parent
    results_dir = workspace_root / "results"
    output_file = results_dir / "MASTER_TEST_STATUS.csv"
    
    print("Scanning test results...")
    test_data = scan_test_results(results_dir)
    
    print(f"Found {len(test_data)} tests with results")
    
    print("Generating MASTER_TEST_STATUS.csv...")
    generate_master_csv(test_data, output_file)
    
    print("\n✅ Done!")
