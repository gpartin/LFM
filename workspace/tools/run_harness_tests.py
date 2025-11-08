# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

import json
import sys
from pathlib import Path
# Bootstrap: ensure 'workspace' is on sys.path to import path_utils
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))
from path_utils import add_workspace_to_sys_path, get_workspace_dir
WS = get_workspace_dir(__file__)
from run_tier5_electromagnetic import Tier5ElectromagneticHarness

config_path = WS / "config" / "config_tier5_electromagnetic.json"
if not config_path.exists():
    raise SystemExit(f"Config not found: {config_path}")

harness = Tier5ElectromagneticHarness(str(config_path))

# Helper to run single test and print summary
def run_and_print(test_id):
    for t in harness.config.get("tests", []):
        if t.get("id") == test_id and t.get("enabled", True):
            print(f"Running {test_id} via harness...")
            res = harness.run_test(t)
            print(f"Result: {res.test_id}, passed={res.passed}")
            print("Metrics summary:")
            for k,v in list(res.metrics.items())[:10]:
                print(f"  {k}: {v}")
            return res
    raise SystemExit(f"Test {test_id} not found or not enabled in config")

if __name__ == '__main__':
    # Run EM-21 with energy work term
    res21 = run_and_print('EM-21')
    print('Done')
