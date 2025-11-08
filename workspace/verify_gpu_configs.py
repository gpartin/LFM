# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

import json

configs = [
    ('Tier 1', 'config/config_tier1_relativistic.json'),
    ('Tier 2', 'config/config_tier2_gravityanalogue.json'),
    ('Tier 3', 'config/config_tier3_energy.json'),
    ('Tier 4', 'config/config_tier4_quantization.json'),
    ('Tier 5', 'config/config_tier5_electromagnetic.json'),
]

print("GPU Configuration Status:")
print("-" * 60)

for name, path in configs:
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    
    # Check both possible locations for GPU settings
    hardware = cfg.get('hardware', {})
    run_settings = cfg.get('run_settings', {})
    
    gpu_enabled = hardware.get('gpu_enabled', run_settings.get('use_gpu', False))
    fused_cuda = hardware.get('use_fused_cuda', run_settings.get('use_fused_cuda', False))
    
    gpu_status = "✓ ENABLED" if gpu_enabled else "✗ DISABLED"
    fused_status = "✓ ENABLED" if fused_cuda else "✗ DISABLED"
    
    print(f"{name}: GPU {gpu_status}, Fused CUDA {fused_status}")

print("-" * 60)
print("\nReady to run optimal parallel tests!")
