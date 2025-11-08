#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""
Promote an experiment from experiments/ to experiments/candidates/

This represents passing Gate 1: Reproducibility
The experiment must run without manual intervention and produce consistent results.
"""

import argparse
import os
import shutil
import json
from datetime import datetime
from pathlib import Path


def promote_experiment(investigation_name: str, workspace_root: str = ".") -> None:
    """
    Promote an experiment to candidate status (Gate 1: Reproducibility)
    
    Args:
        investigation_name: Name of the investigation to promote
        workspace_root: Root directory of the workspace
    """
    # Paths
    exp_dir = Path(workspace_root) / "experiments" / investigation_name
    candidate_dir = Path(workspace_root) / "experiments" / "candidates" / investigation_name
    
    # Validate source exists
    if not exp_dir.exists():
        print(f"❌ Error: Experiment '{investigation_name}' not found at {exp_dir}")
        print(f"\nAvailable experiments:")
        exp_root = Path(workspace_root) / "experiments"
        for item in exp_root.iterdir():
            if item.is_dir() and item.name not in ['candidates', 'archive']:
                print(f"  - {item.name}")
        return
    
    # Check if already promoted
    if candidate_dir.exists():
        print(f"⚠️  Warning: Candidate already exists at {candidate_dir}")
        response = input("Overwrite existing candidate? [y/N]: ")
        if response.lower() != 'y':
            print("Promotion cancelled.")
            return
        shutil.rmtree(candidate_dir)
    
    # Checklist for Gate 1: Reproducibility
    print(f"\n{'='*60}")
    print(f"Promoting: {investigation_name}")
    print(f"Gate 1: Reproducibility Checklist")
    print(f"{'='*60}\n")
    
    checklist = [
        ("Script runs without manual intervention", None),
        ("Results reproduce within acceptable variance (< 5%)", None),
        ("Dependencies documented (requirements.txt or notes)", None),
        ("Random seeds controlled or variance quantified", None),
        ("Output files generated consistently", None),
        ("Computation time documented", None),
    ]
    
    print("Please confirm each criterion has been met:\n")
    
    all_passed = True
    for i, (criterion, _) in enumerate(checklist, 1):
        response = input(f"[{i}/6] {criterion}\n      Met? [y/N]: ")
        if response.lower() != 'y':
            all_passed = False
            print(f"      ❌ Not met\n")
        else:
            print(f"      ✓ Met\n")
    
    if not all_passed:
        print("\n❌ Promotion cancelled: Not all criteria met.")
        print("Please address the failing criteria and try again.")
        return
    
    # Copy experiment to candidates
    print(f"\n{'='*60}")
    print("Copying experiment to candidates/...")
    shutil.copytree(exp_dir, candidate_dir)
    print(f"✓ Copied to {candidate_dir}")
    
    # Create promotion metadata
    metadata = {
        "investigation_name": investigation_name,
        "promoted_to_candidate": datetime.now().isoformat(),
        "gate": "Gate 1: Reproducibility",
        "checklist_passed": True,
        "promoted_by": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }
    
    metadata_file = candidate_dir / "promotion_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Created promotion metadata")
    
    # Create README if it doesn't exist
    readme_file = candidate_dir / "README.md"
    if not readme_file.exists():
        readme_content = f"""# {investigation_name}

## Status: Candidate (Gate 1: Reproducibility)

Promoted: {datetime.now().strftime('%Y-%m-%d')}

## Reproducibility Confirmation

This experiment has passed Gate 1 and is confirmed to:
- Run without manual intervention
- Produce consistent results
- Have documented dependencies
- Control randomness appropriately

## Next Steps

To promote to validated test (Gate 2):
1. Document theoretical basis
2. Perform convergence tests
3. Analyze errors and uncertainties
4. Compare to known limits
5. Write pytest-compatible test
6. Run: `python tools/promote_candidate.py {investigation_name}`

## Original Notes

[Include original experiment notes or link to them]
"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✓ Created README.md")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: Promoted to candidates/")
    print(f"{'='*60}\n")
    
    print(f"Location: {candidate_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run reproducibility tests on different machines/GPUs")
    print(f"  2. Verify statistical consistency across multiple runs")
    print(f"  3. Document computational requirements")
    print(f"  4. When ready for validation, run:")
    print(f"     python tools/promote_candidate.py {investigation_name}")
    
    # Optionally keep or remove original
    print(f"\n")
    response = input(f"Archive original experiment (move to experiments/archive/)? [Y/n]: ")
    if response.lower() != 'n':
        archive_dir = Path(workspace_root) / "experiments" / "archive" / investigation_name
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.move(str(exp_dir), str(archive_dir))
        print(f"✓ Archived to experiments/archive/{investigation_name}")
    else:
        print(f"Original experiment remains at experiments/{investigation_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote experiment to candidate (Gate 1: Reproducibility)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/promote_experiment.py circular_orbit_investigation
  python tools/promote_experiment.py em_wave_dispersion --workspace ../workspace

This tool validates that an experiment meets Gate 1 criteria:
  - Reproducible execution
  - Consistent results
  - Documented dependencies
  - Controlled randomness
        """
    )
    
    parser.add_argument(
        "investigation_name",
        help="Name of the investigation to promote (directory name in experiments/)"
    )
    
    parser.add_argument(
        "--workspace",
        default=".",
        help="Path to workspace root (default: current directory)"
    )
    
    args = parser.parse_args()
    
    promote_experiment(args.investigation_name, args.workspace)


if __name__ == "__main__":
    main()
