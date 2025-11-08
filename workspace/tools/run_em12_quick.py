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
from em_analytical_framework import ChiFieldCoupling
cfg_path = WS / "config" / "config_tier5_electromagnetic.json"
with open(cfg_path, 'r', encoding='utf-8') as _f:
    cfg = json.load(_f)
# small test_config: short total_time to keep runtime low
test_cfg = {
    'chi_wave_frequency': 0.01,
    'chi_wave_amplitude': 0.02,
    'em_probe_frequency': 0.03,
    'interaction_duration': 10.0,
    # Use a small settle_time and a preregistered window so the quick run has enough data
    'settle_time': 0.5,
    'preregister_windows': True,
    'prereg_window_start': 20,
    'prereg_window_length': 512,
    'enable_multitaper': False,
    'enable_dt_ladder': True,
    'multitaper_time_bandwidth': 3.0,
    'multitaper_tapers': 3,
    'use_empirical_calibration': False
}
res = ChiFieldCoupling.dynamic_chi_em_response_fdtd({'time':5.0}, test_cfg, cfg)
print('error, expected, computed:', res['error'], res['expected'], res['computed'])
print('dt_ladder in diagnostics:', res['_diagnostics'].get('dt_ladder'))
