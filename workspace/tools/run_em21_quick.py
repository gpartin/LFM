# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

import json
import sys
from pathlib import Path
# Bootstrap to import path_utils from the workspace root
_WS_DIR = Path(__file__).resolve().parents[1]
if str(_WS_DIR) not in sys.path:
    sys.path.insert(0, str(_WS_DIR))
from path_utils import add_workspace_to_sys_path, get_workspace_dir
WS = get_workspace_dir(__file__)
from run_tier5_electromagnetic import test_dynamic_chi_em_pulse

cfg_path = WS / "config" / "config_tier5_electromagnetic.json"
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# Short test config for quick run
test_cfg = {
    'chi_wave_frequency': 0.02,
    'chi_wave_amplitude': 0.02,
    'gate_center_fraction': 0.75,
    'gate_width_fraction': 0.15,
    'pulse_amplitude': 0.08,
    'pulse_duration': 2.0,
    'steps_override': 800,
    'boundary': 'mur',
    'boundary_ab_compare': True,
    'pml_cells': 8,
    'pml_order': 3,
    'pml_sigma_max': 0.8,
    'enable_dt_ladder': True
}

out_dir = WS / "results" / "Electromagnetic" / "EM-21_quick"
out_dir.mkdir(parents=True, exist_ok=True)

res = test_dynamic_chi_em_pulse(cfg, test_cfg, out_dir)
print('EM-21 quick result:', res.test_id, 'passed=', res.passed)
print('metrics:', res.metrics)
