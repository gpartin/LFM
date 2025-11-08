#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © 2025 Emergent Physics Lab. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# Non-commercial use with attribution only; no distribution of modified material; commercial use requires prior written permission.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0


import json
from pathlib import Path

# Test config loading
config_path = "config/config_tier5_electromagnetic.json"
print(f"Checking config file: {config_path}")
print(f"File exists: {Path(config_path).exists()}")

if Path(config_path).exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Output dir: {config.get('output_dir')}")
    print(f"Output dir type: {type(config.get('output_dir'))}")
    print(f"Number of tests: {len(config.get('tests', []))}")
    
    # Check first test
    if config.get('tests'):
        first_test = config['tests'][0]
        print(f"First test ID: {first_test.get('id')}")
        print(f"First test type: {first_test.get('type')}")