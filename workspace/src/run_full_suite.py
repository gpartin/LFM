#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full LFM test suite (excluding known problematic tests)

Excludes:
- GRAV-09: Too slow (per user request "skip grav-09!")
- GRAV-23: Requires optional chi_field_equation module
"""

import subprocess
import sys
from pathlib import Path

# All test IDs across 7 tiers (manually constructed)
ALL_TEST_IDS = [
    # Tier 1: Relativistic (17 tests)
    "REL-01", "REL-02", "REL-03", "REL-04", "REL-05", "REL-06", "REL-07",
    "REL-08", "REL-09", "REL-10", "REL-11", "REL-12", "REL-13", "REL-14",
    "REL-15", "REL-16", "REL-17",
    
    # Tier 2: Gravity Analogue (24 tests, excluding GRAV-09 and GRAV-23)
    "GRAV-01", "GRAV-02", "GRAV-03", "GRAV-04", "GRAV-05", "GRAV-06",
    "GRAV-07", "GRAV-08", # "GRAV-09" EXCLUDED (too slow)
    "GRAV-10", "GRAV-11", "GRAV-12", "GRAV-13", "GRAV-14", "GRAV-15",
    "GRAV-16", "GRAV-17", "GRAV-18", "GRAV-19", "GRAV-20", "GRAV-21",
    "GRAV-22", # "GRAV-23" EXCLUDED (requires optional module)
    "GRAV-24", "GRAV-25", "GRAV-26",
    
    # Tier 3: Energy Conservation (11 tests)
    "ENER-01", "ENER-02", "ENER-03", "ENER-04", "ENER-05", "ENER-06",
    "ENER-07", "ENER-08", "ENER-09", "ENER-10", "ENER-11",
    
    # Tier 4: Quantization (16 tests)
    "QUAN-01", "QUAN-02", "QUAN-03", "QUAN-04", "QUAN-05", "QUAN-06",
    "QUAN-07", "QUAN-08", "QUAN-09", "QUAN-10", "QUAN-11", "QUAN-12",
    "QUAN-13", "QUAN-14", "QUAN-15", "QUAN-16",
    
    # Tier 5: Electromagnetic (21 tests)
    "EM-01", "EM-02", "EM-03", "EM-04", "EM-05", "EM-06", "EM-07",
    "EM-08", "EM-09", "EM-10", "EM-11", "EM-12", "EM-13", "EM-14",
    "EM-15", "EM-16", "EM-17", "EM-18", "EM-19", "EM-20", "EM-21",
    
    # Tier 6: Coupling (12 tests)
    "COUP-01", "COUP-02", "COUP-03", "COUP-04", "COUP-05", "COUP-06",
    "COUP-07", "COUP-08", "COUP-09", "COUP-10", "COUP-11", "COUP-12",
    
    # Tier 7: Thermodynamics (5 tests)
    "THERM-01", "THERM-02", "THERM-03", "THERM-04", "THERM-05",
]

print(f"Full Test Suite Execution")
print(f"=" * 60)
print(f"Total tests: {len(ALL_TEST_IDS)}")
print(f"Excluded: GRAV-09 (slow), GRAV-23 (optional module)")
print(f"=" * 60)
print()

# Build test list string
test_list = ",".join(ALL_TEST_IDS)

# Run parallel suite
src_dir = Path(__file__).resolve().parent
python_exe = sys.executable
cmd = [
    python_exe,
    str(src_dir / "run_parallel_suite.py"),
    "--tests", test_list,
    "--max-concurrent", "8",
]

print(f"Running command:")
print(f"  {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd=str(src_dir))
sys.exit(result.returncode)
