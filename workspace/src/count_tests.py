# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0

"""Count total tests across all tiers."""
import json

tiers = [
    ('config_tier1_relativistic.json', 1),
    ('config_tier2_gravityanalogue.json', 2),
    ('config_tier3_energy.json', 3),
    ('config_tier4_quantization.json', 4),
    ('config_tier5_electromagnetic.json', 5),
    ('config_tier6_coupling.json', 6),
    ('config_tier7_thermodynamics.json', 7)
]

total = 0
for fname, t in tiers:
    with open(f'../config/{fname}', encoding='utf-8') as f:
        c = json.load(f)
    n = len(c.get('variants', c.get('tests', [])))
    print(f'Tier {t}: {n:2d} tests')
    total += n

print(f'\nTOTAL: {total} tests')
